"""Revert an ingest batch: delete the documents an index file lists, for one tenant.

    uv run python -m agintis.cli.revert_ingest --profiles case/x/profiles \
        --index case/x/index/documents.jsonl --tenant acme            # dry run
    uv run python -m agintis.cli.revert_ingest ... --apply            # delete

Reads the SAME index file `agintis.cli.ingest` consumes (the `url` of each row is the document's
source URI; no local files are needed). Every version of those sources (any profile, any status)
of `--tenant` is selected; chunks go with their document. The default is a DRY RUN that prints
what would be deleted and the counts; `--apply` deletes (each deletion is logged). Re-ingesting
afterwards restores the documents. After a bulk revert, HNSW results may differ slightly from
before: see the README note on re-measuring or `REINDEX`.

Exit codes: 0 ok; 2 bad arguments or an index without usable rows; 3 aborted (startup check,
unavailable database).
"""

import argparse
import asyncio
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from agintis.bootstrap.container import Container
from agintis.bootstrap.settings import get_settings
from agintis.cli.local_case import iter_index
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
    parser.add_argument("--tenant", required=True, help="tenant id whose documents are reverted")
    parser.add_argument(
        "--apply", action="store_true", help="delete (default: dry run, nothing changes)"
    )
    return parser


def index_uris(index: Path) -> list[str]:
    """Source URIs of the index rows (their `url`); rows without one are skipped."""
    return [str(row["url"]) for row in iter_index(index) if row.get("url")]


def format_document(document: Document) -> str:
    return f"{document.status.value:10} {document.profile:34} {document.source_uri}"


async def run(args: argparse.Namespace, uris: Sequence[str]) -> list[Document]:
    settings = get_settings()
    settings = settings.model_copy(
        update={
            "knowledge": settings.knowledge.model_copy(update={"profiles_dir": args.profiles}),
            "checkpointer": settings.checkpointer.model_copy(update={"enabled": False}),
            "agents": settings.agents.model_copy(update={"dir": None}),
        }
    )
    async with Container.build(settings, for_ingest=True) as container:
        return await container.ingest_batch.revert(TenantId(args.tenant), uris, apply=args.apply)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    uris = index_uris(args.index)
    if not uris:
        print(f"{args.index} has no rows with a url", file=sys.stderr)
        return EXIT_USAGE
    try:
        documents = asyncio.run(run(args, uris))
    except Exception as exc:  # startup checks, database down, bad profiles
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        return EXIT_ABORTED
    verb = "deleted" if args.apply else "would delete"
    for document in documents:
        print(f"{verb} {format_document(document)}", flush=True)
    by_profile = Counter(d.profile for d in documents)
    detail = ", ".join(f"{name}={n}" for name, n in sorted(by_profile.items())) or "none"
    mode = "" if args.apply else " (dry run, pass --apply to delete)"
    print(f"summary: {verb} {len(documents)} documents of {len(set(uris))} sources: {detail}{mode}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
