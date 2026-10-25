"""Sync the per-field metadata indexes of the loaded profiles with the database.

    uv run python -m agintis.cli.indexes --profiles case/<name>/profiles            # plan
    uv run python -m agintis.cli.indexes --profiles case/<name>/profiles --apply    # create

Reads the profiles, computes the expression index every `indexed` scalar field needs (derived from
the SQL the filters emit, so the planner can use it), diffs it against the database and prints
the plan. Nothing changes without `--apply`, which runs `CREATE INDEX CONCURRENTLY IF NOT EXISTS`
outside a transaction (reads and writes keep working; the build can take a while on big tables).
It is idempotent: a second run is a no-op.

Only range / ordering filters use these indexes (equality is served by the table-wide GIN):
flag `indexed` the fields compared with gt/gte/lt/lte or sorted on.

Indexes are never dropped by default. `--prune --confirm-prune` also drops (and rebuilds, when
expected) the unusable ones: orphans nobody declares any more, stale ones built from another
expression, unmanaged ones (our name, but built by hand or without our fingerprint comment) and
invalid leftovers of an interrupted concurrent build. An index still being built is never touched.
Note that an orphan is relative to the profiles you pass: point `--profiles` at the directory the
deployment really loads.

One run at a time: the sync takes an advisory lock and waits at most `--lock-timeout` seconds for
another run to finish. If a run fails midway, the indexes it had already created or dropped are
printed; nothing is rolled back (re-run to continue).

Only the database and the profile files are used (no embedding service). Exit codes: 0 plan
printed or applied; 2 bad arguments or incompatible declarations; 3 the run was aborted (database
unavailable, invalid profiles, a failed index build).
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from agintis.bootstrap.container import Container
from agintis.bootstrap.settings import get_settings
from agintis.modules.knowledge.api import (
    ApplyResult,
    ConflictingIndexError,
    IndexPlan,
    IndexStatus,
    IndexSyncError,
)

DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ABORTED = 3
logger = logging.getLogger("agintis.cli.indexes")

_ACTIONS = {
    IndexStatus.MISSING: "create",
    IndexStatus.PRESENT: "-",
    IndexStatus.STALE: "needs rebuild (--prune --confirm-prune)",
    IndexStatus.INVALID: "needs rebuild (--prune --confirm-prune)",
    IndexStatus.UNMANAGED: "not created by this tool (--prune --confirm-prune rebuilds it)",
    IndexStatus.BUILDING: "being built by another session: left alone",
    IndexStatus.ORPHAN: "unused (--prune --confirm-prune drops it)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.indexes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profiles", type=Path, required=True, help="directory of YAML profiles")
    parser.add_argument("--apply", action="store_true", help="create the missing indexes")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="also drop orphan, stale and invalid indexes (needs --confirm-prune)",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=DEFAULT_LOCK_TIMEOUT_SECONDS,
        help="seconds to wait for another sync run (default: %(default)s)",
    )
    parser.add_argument(
        "--confirm-prune", action="store_true", help="explicit confirmation that --prune may DROP"
    )
    return parser


def format_plan(plan: IndexPlan) -> str:
    if not plan.entries:
        return "no indexed metadata fields: nothing to do"
    width = max(len(e.name) for e in plan.entries)
    lines = [f"{'index':{width}}  {'status':9}  action"]
    lines += [f"{e.name:{width}}  {e.status.value:9}  {_ACTIONS[e.status]}" for e in plan.entries]
    return "\n".join(lines)


def format_pending_sql(plan: IndexPlan) -> list[str]:
    return [
        f"{e.spec.create_sql};" for e in plan.with_status(IndexStatus.MISSING) if e.spec is not None
    ]


async def run(args: argparse.Namespace) -> tuple[IndexPlan, ApplyResult | None]:
    settings = get_settings()
    settings = settings.model_copy(
        update={
            "knowledge": settings.knowledge.model_copy(update={"profiles_dir": args.profiles}),
            # Agent specs are validated against the profiles of the RUNNING app, not these.
            "agents": settings.agents.model_copy(update={"dir": None}),
        }
    )
    # Not entered as `async with`: the startup checks need the embedding service, which an
    # index sync does not use.
    container = Container.build(settings)
    try:
        fields = [f for profile in container.profiles.all() for f in profile.metadata_fields]
        plan = await container.index_sync.plan(fields)
        print(format_plan(plan))
        if not args.apply:
            for statement in format_pending_sql(plan):
                print(f"  {statement}")
            return plan, None
        return plan, await container.index_sync.apply(
            plan, prune=args.prune, lock_timeout_seconds=args.lock_timeout
        )
    finally:
        await container.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prune and not args.confirm_prune:
        print("--prune drops indexes: pass --confirm-prune as well", file=sys.stderr)
        return EXIT_USAGE
    if args.lock_timeout <= 0:
        print("--lock-timeout must be greater than 0", file=sys.stderr)
        return EXIT_USAGE
    if args.confirm_prune and not args.prune:
        print("--confirm-prune only makes sense with --prune", file=sys.stderr)
        return EXIT_USAGE
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        plan, result = asyncio.run(run(args))
    except ConflictingIndexError as exc:
        print(f"conflict: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except IndexSyncError as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        _print_result(exc.result, header="done before the failure")
        return EXIT_ABORTED
    except Exception as exc:  # database down, invalid profiles, a failed concurrent build
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        return EXIT_ABORTED
    if result is None:
        pending = len(plan.with_status(IndexStatus.MISSING))
        print(f"dry run: {pending} index(es) to create; pass --apply to create them")
    else:
        _print_result(result)
    return EXIT_OK


def _print_result(result: ApplyResult, *, header: str | None = None) -> None:
    if header:
        print(f"{header}:")
    print(f"created: {', '.join(result.created) or '-'}")
    print(f"dropped: {', '.join(result.dropped) or '-'}")
    if result.unmarked:
        print(
            f"created WITHOUT fingerprint comment (unmanaged until rebuilt with --prune): "
            f"{', '.join(result.unmarked)}"
        )
    if result.skipped:
        print(f"skipped (being built elsewhere): {', '.join(result.skipped)}")


if __name__ == "__main__":
    sys.exit(main())
