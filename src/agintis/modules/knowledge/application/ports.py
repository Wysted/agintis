"""Ports implemented by infrastructure adapters (parsing, chunking and reranking strategies)."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import ClassVar

from agintis.modules.knowledge.application.dto import (
    ChunkDraft,
    DocumentMetadata,
    ParsedDocument,
    SourceFile,
)
from agintis.modules.knowledge.domain import ParamValue, RagProfile, ScoredChunk
from agintis.shared_kernel.ids import TenantId


class Parser(ABC):
    # Strategy implementation version, part of the profile fingerprint (see `RagProfile`).
    # Bump it when a change to the code or its defaults can change the output: documents
    # ingested with the old behavior are then re-indexed. 1 = baseline (not hashed). A string
    # may name what the output depends on beyond the code (e.g. an OCR engine and its model).
    version: ClassVar[int | str] = 1

    @abstractmethod
    async def parse(self, source: SourceFile) -> ParsedDocument: ...


class Chunker(ABC):
    version: ClassVar[int] = 1  # see `Parser.version`
    # Smallest `chunk_max_tokens` a profile may give this chunker (checked at boot).
    min_chunk_max_tokens: ClassVar[int] = 1

    def dependency_versions(self) -> Mapping[str, int]:
        """Versions of OTHER strategies this chunker delegates to, as `"<kind>:<key>": version`.

        They join the profile fingerprint, so a change in a delegate re-indexes the documents
        of every profile that reaches it (e.g. `whole_document` falling back to `structured`).
        """
        return {}

    @abstractmethod
    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]: ...


class Cleaner(ABC):
    """One step of a profile's cleaner pipeline: parsed text in, cleaned text out.

    Runs after parsing and before chunking. `params` are the profile's (already validated at
    boot) parameters for this key; the cleaner must not mutate its input.
    """

    version: ClassVar[int] = 1  # see `Parser.version`

    @abstractmethod
    async def clean(
        self, parsed: ParsedDocument, params: Mapping[str, ParamValue]
    ) -> ParsedDocument: ...


class Reranker(ABC):
    """Reorders retrieval candidates by relevance to the query (cross-encoder style)."""

    @abstractmethod
    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Return at most `top_n` chunks, best first, carrying the reranker's own scores.

        A reranker that did not rescore a chunk (no-op, degraded fallback) must return the
        very same `ScoredChunk` object it received: retrieval treats identity as "not scored".
        """


class IngestLock(ABC):
    """Serializes ingests of the SAME source: key = (tenant_id, profile, source_uri).

    Everything between "read current versions" and "swap to the new READY version" runs under
    the lock, including cleanup of stale PENDING/SUPERSEDED leftovers (only safe while held:
    without it a PENDING row may belong to a live concurrent ingest).

    Postgres adapter: a SESSION advisory lock (`pg_advisory_lock` / `pg_advisory_unlock` on a
    dedicated connection; `pg_advisory_xact_lock` only if the whole ingest runs in one
    transaction) keyed by a stable 64-bit hash of the key (e.g. first 8 bytes of sha256 of
    "tenant\x00profile\x00source_uri", signed). Locks must be released on cancellation.
    """

    @abstractmethod
    def hold(
        self, tenant_id: TenantId, profile: str, source_uri: str
    ) -> AbstractAsyncContextManager[None]: ...


@dataclass(frozen=True)
class ProfileSelection:
    """Outcome of routing one document: the chosen profile, or None plus why none applies."""

    profile: RagProfile | None
    reason: str


class ProfileSelector(ABC):
    """Picks the profile a document should be ingested with, from its document metadata."""

    @abstractmethod
    def select(self, document: DocumentMetadata) -> ProfileSelection: ...
