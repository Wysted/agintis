"""Batch ingest of a case's local documents.

    uv run python -m agintis.cli.ingest --profiles case/spensiones/profiles \
        --index case/spensiones/index/documents.jsonl --limit 10

Reads the index rows, keeps those whose file is already on disk (`<samples>/_manifest.json`;
nothing is downloaded), routes each to a profile from the profiles' `routing` rules and ingests
it through the container's `IngestBatch`. One line is printed per document AS IT FINISHES, then a
summary by outcome. Database and embedding service settings come from the usual environment /
`.env`.

Exit codes: 0 every document was handled (indexed, replaced, skipped or unrouted); 1 at least one
document failed (the batch went on); 2 bad arguments; 3 the run was aborted (startup check, an
unavailable database or embedding service): what finished before it is still printed.

A row that cannot be turned into a document (malformed JSON, invalid metadata, unreadable file)
is reported as `failed` and the run continues.

Re-routing: when a profile's routing changes, a source that moved leaves its READY version under
the old profile (served next to the new one). Such sources are flagged in the output; pass
`--cleanup-rerouted` to delete the old versions.

Pruning: a source that disappeared from the index keeps its READY version (it is served forever).
`--prune` removes, after a batch WITHOUT failures, the READY documents of this tenant under the
profiles of `--profiles` whose source is not in the batch (each removal is logged). It is skipped
with a warning when any document failed or any index row had no file on disk (a partial fetch
must not delete good documents), when nothing was ingested (an empty index), and it cannot be
combined with `--limit`. "In the batch" means in THIS index file: use ONE index per set of
profiles (two index files over the same profiles would prune each other's documents) and do not
run two ingests concurrently over the same profiles. A source that is now unrouted keeps its old
READY version: neither `--prune` nor `--cleanup-rerouted` removes it.

Tokenizer: the chunkers count tokens with the tokenizer of `EMBEDDINGS_TOKENIZER` (default: the
model id, e.g. `BAAI/bge-m3`). A hub id is DOWNLOADED from the Hugging Face Hub on first use
(then cached under ~/.cache/huggingface); point `EMBEDDINGS_TOKENIZER` at a local
`tokenizer.json` for a run that never touches the network. `--offline` (or a local tokenizer
path) sets `HF_HUB_OFFLINE=1`; the Rust loader behind a hub id may not honour it, so a local
file is the only hard guarantee.
"""

import argparse
import asyncio
import collections
import logging
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from agintis.bootstrap.container import Container
from agintis.bootstrap.settings import get_settings
from agintis.cli.local_case import iter_index_entries, load_manifest, source_for
from agintis.modules.knowledge.api import DEFAULT_PROFILE, BatchItem, BatchOutcome
from agintis.shared_kernel.errors import DomainError
from agintis.shared_kernel.ids import TenantId

DEFAULT_TENANT = "default"
EXIT_OK = 0
EXIT_DOCUMENT_FAILED = 1
EXIT_USAGE = 2
EXIT_ABORTED = 3
logger = logging.getLogger("agintis.cli.ingest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.ingest",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profiles", type=Path, required=True, help="directory of YAML profiles")
    parser.add_argument("--index", type=Path, required=True, help="JSON Lines index of documents")
    parser.add_argument(
        "--samples",
        type=Path,
        help="directory with the downloaded files and _manifest.json "
        "(default: <index dir>/../samples)",
    )
    parser.add_argument("--limit", type=int, help="ingest at most this many local documents")
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="tenant id (default: %(default)s)")
    parser.add_argument(
        "--cleanup-rerouted",
        action="store_true",
        help="delete the READY versions a source keeps under profiles it is no longer routed to",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="after a batch without failures, delete the READY documents of these profiles "
        "whose source is not in THIS index (one index per profile set, no concurrent ingest; "
        "incompatible with --limit; skipped when a failure, an empty index or an index row "
        "without a local file is found)",
    )
    parser.add_argument(
        "--offline", action="store_true", help="set HF_HUB_OFFLINE=1 (see the tokenizer note)"
    )
    return parser


def local_items(
    index: Path, samples: Path, limit: int | None, missing: list[str] | None = None
) -> Iterator[BatchItem]:
    """Index rows whose file is on disk, in index order, at most `limit` of them; lazily.

    The rows skipped for lack of a local file are appended to `missing` (when given).

    A line or row that cannot become a document is yielded as a ready-made `failed` outcome (it
    does not count towards `limit`), so one bad row never ends the run.
    """
    files = load_manifest(samples)
    produced = 0
    for entry in iter_index_entries(index):
        if limit is not None and produced >= limit:
            return
        place = f"{index.name}:{entry.line}"
        if entry.row is None:
            yield BatchOutcome(place, "failed", reason=f"unreadable index line: {entry.error}")
            continue
        try:
            source = source_for(entry.row, files)
        except (DomainError, OSError) as exc:
            local = files.get(str(entry.row.get("id")))
            uri = local.url if local is not None else str(entry.row.get("url") or place)
            yield BatchOutcome(uri, "failed", reason=f"{type(exc).__name__}: {exc}")
            continue
        if source is None:
            if missing is not None:
                missing.append(place)
            continue
        produced += 1
        yield source


def format_outcome(outcome: BatchOutcome) -> str:
    detail = f"{outcome.chunk_count} chunks" if outcome.reason is None else outcome.reason
    if outcome.rerouted_from:
        action = "removed" if outcome.removed_rerouted else "STILL READY, use --cleanup-rerouted"
        detail += f"; also under {', '.join(outcome.rerouted_from)} ({action})"
    if outcome.title_refreshed:
        detail += "; title refreshed"
    return f"{outcome.status:9} {outcome.profile or '-':34} {outcome.uri}  ({detail})"


def _configure_offline(requested: bool, tokenizer_source: str) -> None:
    """A local tokenizer file needs no network: say so to the Hugging Face libraries too."""
    if requested or Path(tokenizer_source).is_file():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


async def run(args: argparse.Namespace, emit: Callable[[BatchOutcome], None]) -> int | None:
    """Ingest the case, handing every outcome to `emit` as soon as it is known.

    Returns the number of pruned documents; None when `--prune` was not given or was skipped.
    """
    settings = get_settings()
    settings = settings.model_copy(
        update={
            "knowledge": settings.knowledge.model_copy(update={"profiles_dir": args.profiles}),
            # Ingest has no conversations: no checkpointer pool, no checkpoint-table boot check.
            "checkpointer": settings.checkpointer.model_copy(update={"enabled": False}),
            # Agent specs are validated against the profiles of the RUNNING app, not these.
            "agents": settings.agents.model_copy(update={"dir": None}),
        }
    )
    _configure_offline(args.offline, settings.embeddings.tokenizer_source)
    samples: Path = args.samples or args.index.parent.parent / "samples"
    missing: list[str] = []
    items = local_items(args.index, samples, args.limit, missing)
    async with Container.build(settings, for_ingest=True) as container:
        tenant = TenantId(args.tenant)
        # Unchanged documents are skipped before add_many: an existing tenant's missing BM25
        # index is built here.
        await container.chunks.repair_bm25_index(tenant)
        outcomes: list[BatchOutcome] = []
        async for outcome in container.ingest_batch.stream(
            tenant, items, cleanup_rerouted=args.cleanup_rerouted
        ):
            outcomes.append(outcome)
            emit(outcome)
        if not args.prune:
            return None
        if missing:
            logger.warning(
                "Prune skipped: %d index rows have no local file (partial fetch)", len(missing)
            )
            return None
        names = [p.name for p in container.profiles.all() if p.name != DEFAULT_PROFILE.name]
        # shortcut: prune scope = this index file's outcomes, no cross-run registry; upgrade to a
        # per-profile-set manifest if several index files must feed the same profiles.
        removed = await container.ingest_batch.prune(tenant, names, outcomes)
        return None if removed is None else len(removed)


def _summary(counts: collections.Counter[str]) -> str:
    """Status counts; `titles_refreshed` is a side count of skipped documents, not a status."""
    return ", ".join(f"{status}={n}" for status, n in sorted(counts.items())) or "empty"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        print("--limit must be greater than 0", file=sys.stderr)
        return EXIT_USAGE
    if args.prune and args.limit is not None:
        print("--prune cannot be combined with --limit", file=sys.stderr)
        return EXIT_USAGE
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Every embedding request is an INFO line of httpx: thousands per run.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    counts: collections.Counter[str] = collections.Counter()

    def emit(outcome: BatchOutcome) -> None:
        counts[outcome.status] += 1
        if outcome.title_refreshed:
            counts["titles_refreshed"] += 1
        print(format_outcome(outcome), flush=True)

    try:
        pruned = asyncio.run(run(args, emit))
    except Exception as exc:  # startup checks, database or embedding service down, bad profiles
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        print("summary (partial):", _summary(counts), flush=True)
        return EXIT_ABORTED
    summary = _summary(counts)
    if args.prune:
        summary += f", pruned={pruned}" if pruned is not None else ", prune skipped (see warnings)"
    print("summary:", summary)
    return EXIT_DOCUMENT_FAILED if counts["failed"] else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
