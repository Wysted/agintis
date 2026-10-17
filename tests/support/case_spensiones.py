"""Runs the spensiones case over its local sample documents (no DB, no embedding service).

Samples and the crawler index are raw case data (gitignored): `samples_available()` tells
whether they exist on this machine. Each sample goes through the real pipeline: profile routing
by its index row, parser, cleaners, chunker, metadata resolution and `conform_metadata`
(inside `IngestDocument`), with in-memory repositories and a fake embedding.
"""

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.cli.local_case import iter_index, load_manifest, source_for
from agintis.modules.knowledge.api import (
    IngestDocument,
    MetadataResolver,
    ProfileRouter,
    load_profiles_dir,
)
from agintis.modules.knowledge.domain import Chunk
from agintis.modules.knowledge.infrastructure.chunkers import TokenCounter
from agintis.modules.knowledge.infrastructure.parsers import ParseLimits
from agintis.modules.knowledge.infrastructure.registry import (
    build_chunkers,
    build_cleaners,
    build_extractors,
    build_parsers,
    build_transforms,
)
from agintis.shared_kernel.ids import TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    NullIngestLock,
)

CASE_DIR = Path(__file__).resolve().parents[2] / "case" / "spensiones"
PROFILES_DIR = CASE_DIR / "profiles"
INDEX_FILE = CASE_DIR / "index" / "documents.jsonl"
SAMPLES_DIR = CASE_DIR / "samples"
# The reference validation (validation_report.md) ran on this day: future-date rejection uses it.
REFERENCE_INGEST_DATE = dt.datetime(2026, 10, 15, tzinfo=dt.UTC)
TENANT = TenantId("case")


def samples_available() -> bool:
    return INDEX_FILE.is_file() and (SAMPLES_DIR / "_manifest.json").is_file()


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    profile: str | None
    chunks: tuple[Chunk, ...]
    error: str | None = None


async def ingest_samples(counter: TokenCounter) -> dict[str, SampleResult]:
    """Ingest every downloaded sample with the profile the router picks; results by sample id."""
    limits = ParseLimits()
    profiles = load_profiles_dir(PROFILES_DIR)
    router = ProfileRouter(profiles)
    manifest = load_manifest(SAMPLES_DIR)
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_index(INDEX_FILE):
        rows.setdefault(str(row["id"]), row)

    results: dict[str, SampleResult] = {}
    for sample_id in manifest:
        source = source_for(rows[sample_id], manifest)
        assert source is not None
        selection = router.select(source.document_metadata)
        if selection.profile is None:
            results[sample_id] = SampleResult(sample_id, None, (), selection.reason)
            continue
        chunks = InMemoryChunkRepository()
        documents = InMemoryDocumentRepository()
        ingest = IngestDocument(
            documents,
            chunks,
            build_parsers(
                max_bytes=limits.max_bytes,
                max_pages=limits.max_pages,
                max_uncompressed_bytes=limits.max_uncompressed_bytes,
                max_compression_ratio=limits.max_compression_ratio,
            ),
            build_chunkers(counter),
            DeterministicFakeEmbedding(size=8),
            FixedClock(REFERENCE_INGEST_DATE),
            NullIngestLock(),
            cleaners=build_cleaners(),
            metadata=MetadataResolver(
                build_extractors(), build_transforms(), FixedClock(REFERENCE_INGEST_DATE)
            ),
        )
        await ingest.execute(TENANT, selection.profile, source)
        ordered = sorted((c for c, _ in chunks.items), key=lambda c: c.ordinal)
        results[sample_id] = SampleResult(sample_id, selection.profile.name, tuple(ordered))
    return results
