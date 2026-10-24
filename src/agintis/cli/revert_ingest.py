"""Revert an ingest batch: delete the documents an index file lists, for one tenant.

    uv run python -m agintis.cli.revert_ingest --profiles case/x/profiles \
        --index case/x/index/documents.jsonl --tenant acme            # dry run
    uv run python -m agintis.cli.revert_ingest ... --apply            # delete

Reads the SAME index file `agintis.cli.ingest` consumes and resolves each row's source URI exactly
as ingest did: the url of the row's entry in `<samples>/_manifest.json` (`--samples`, default
`<index dir>/../samples`; the index `url` may differ). Rows without a local file were never
ingested and are skipped. Every version of those sources (any profile, any status)
of `--tenant` is selected; chunks go with their document. The default is a DRY RUN that prints
what would be deleted and the counts; `--apply` deletes (each deletion is logged). Re-ingesting
afterwards restores the documents. `--apply` re-resolves the documents when it runs, so a
concurrent ingest between a dry run and the apply can change the set. The dry run opens the
database read-only. After a bulk revert, HNSW results may differ slightly from
before: see the README note on re-measuring or `REINDEX`.

Exit codes: 0 ok (also when nothing matched: a warning is printed); 2 bad arguments or an index
without usable rows; 3 aborted (startup check, unavailable database, a concurrent ingest): the
documents already deleted are listed first.
"""

import argparse
import asyncio
import logging
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from agintis.bootstrap.container import Container
from agintis.bootstrap.settings import get_settings
from agintis.cli.local_case import source_uris
from agintis.modules.knowledge.api import Document
from agintis.shared_kernel.ids import TenantId

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ABORTED = 3
logger = logging.getLogger("agintis.cli.revert_ingest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.revert_ingest",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profiles", type=Path, required=True, help="directory of YAML profiles")
    parser.add_argument("--index", type=Path, required=True, help="JSON Lines index of documents")
    parser.add_argument(
        "--samples",
        type=Path,
        help="directory with _manifest.json (default: <index dir>/../samples)",
    )
    parser.add_argument("--tenant", required=True, help="tenant id whose documents are reverted")
    parser.add_argument(
        "--apply", action="store_true", help="delete (default: dry run, nothing changes)"
    )
    return parser


def format_document(document: Document) -> str:
    return f"{document.status.value:10} {document.profile:34} {document.source_uri}"


async def run(
    args: argparse.Namespace, uris: Sequence[str], on_removed: Callable[[Document], None]
) -> list[Document]:
    settings = get_settings()
    settings = settings.model_copy(
        update={
            "knowledge": settings.knowledge.model_copy(update={"profiles_dir": args.profiles}),
            # A dry run cannot write, whatever a bug does.
            "database": settings.database.model_copy(update={"read_only": not args.apply}),
            "checkpointer": settings.checkpointer.model_copy(update={"enabled": False}),
            "agents": settings.agents.model_copy(update={"dir": None}),
        }
    )
    async with Container.build(settings, for_ingest=True) as container:
        return await container.ingest_batch.revert(
            TenantId(args.tenant), uris, apply=args.apply, on_removed=on_removed
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    samples: Path = args.samples or args.index.parent.parent / "samples"
    uris, skipped = source_uris(args.index, samples)
    if skipped:
        print(f"warning: {skipped} index rows have no local file (never ingested): skipped")
    if not uris:
        print(f"{args.index} has no rows with a local file", file=sys.stderr)
        return EXIT_USAGE
    verb = "deleted" if args.apply else "would delete"
    documents: list[Document] = []

    def show(document: Document) -> None:
        documents.append(document)
        print(f"{verb} {format_document(document)}", flush=True)

    try:
        found = asyncio.run(run(args, uris, show))
    except Exception as exc:  # startup checks, database down, a concurrent ingest
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        if args.apply:
            print(f"summary (partial): deleted {len(documents)} documents before the abort")
        return EXIT_ABORTED
    if not args.apply:
        for document in found:
            show(document)
    if not found:
        print(
            f"warning: no document of tenant {args.tenant!r} matches these sources "
            "(check the tenant and the manifest)"
        )
    by_profile = Counter(d.profile for d in documents)
    detail = ", ".join(f"{name}={n}" for name, n in sorted(by_profile.items())) or "none"
    mode = "" if args.apply else " (dry run, pass --apply to delete)"
    print(f"summary: {verb} {len(documents)} documents of {len(set(uris))} sources: {detail}{mode}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
