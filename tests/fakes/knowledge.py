"""Fakes for the knowledge module ports."""

import asyncio
import math
import re
import unicodedata
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from typing import Any

from agintis.modules.knowledge.application import (
    ChunkDraft,
    Chunker,
    Cleaner,
    IngestLock,
    ParsedDocument,
    ParsedSection,
    Parser,
    SourceFile,
)
from agintis.modules.knowledge.domain import (
    Chunk,
    ChunkRepository,
    Document,
    DocumentIdentityError,
    DocumentNotFoundError,
    DocumentRepository,
    DocumentStatus,
    DuplicateDocumentError,
    MetadataField,
    ParamValue,
    RagProfile,
    ScoredChunk,
    TenantMismatchError,
    chunk_filterable_fields,
    ensure_single_tenant,
)
from agintis.platform.db.criteria import MAX_IN_VALUES
from agintis.shared_kernel.criteria import (
    Criteria,
    Filter,
    FilterExpr,
    UnsupportedFilterError,
    ensure_fields,
    tenant_scoped,
)
from agintis.shared_kernel.criteria.in_memory import apply, matches
from agintis.shared_kernel.ids import DocumentId, TenantId


def _ensure_in_cap(expr: FilterExpr | None) -> None:
    """Fail like the SQL converter does when an IN list is over its cap."""
    if expr is None:
        return
    if isinstance(expr, Filter):
        if isinstance(expr.value, tuple) and len(expr.value) > MAX_IN_VALUES:
            raise UnsupportedFilterError(f"{expr.field!r} has {len(expr.value)} values")
        return
    for child in expr.exprs:
        _ensure_in_cap(child)


def _document_record(document: Document) -> dict[str, Any]:
    return asdict(document)


def _source_key(document: Document) -> tuple[str, str, str]:
    return (document.tenant_id, document.profile, document.source_uri)


def _chunk_record(chunk: Chunk) -> dict[str, Any]:
    core = {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "tenant_id": chunk.tenant_id,
        "profile": chunk.profile,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
    }
    # Core fields LAST so metadata can never shadow identity or tenant columns.
    return {**dict(chunk.metadata), **core}


_STOPWORDS = frozenset({"a", "de", "del", "el", "en", "la", "las", "los", "que", "un", "una", "y"})


def lexical_terms(text: str) -> list[str]:
    """Lowercase, accent-free word tokens without stopwords (a stand-in for `es_unaccent`)."""
    folded = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in folded if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+", plain) if t not in _STOPWORDS]


def lexical_score(query_terms: Sequence[str], text: str) -> float:
    """0.0 unless EVERY query term occurs in `text` (AND, like plainto_tsquery);
    otherwise the term frequency normalised by length (higher is better)."""
    terms = lexical_terms(text)
    if not query_terms or any(term not in terms for term in query_terms):
        return 0.0
    return sum(terms.count(term) for term in set(query_terms)) / len(terms)


def any_term_score(query_terms: Sequence[str], text: str) -> float:
    """0.0 unless SOME query term occurs in `text` (OR, a stand-in for BM25 without IDF);
    otherwise the term frequency normalised by length (higher is better)."""
    terms = lexical_terms(text)
    found = sum(terms.count(term) for term in set(query_terms))
    return found / len(terms) if found else 0.0


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryDocumentRepository(DocumentRepository):
    def __init__(self) -> None:
        self.items: dict[DocumentId, Document] = {}

    async def save(self, document: Document) -> None:
        existing = self.items.get(document.id)
        if existing is not None and existing.tenant_id != document.tenant_id:
            raise TenantMismatchError(document.id)
        if existing is not None and (existing.profile, existing.source_uri) != (
            document.profile,
            document.source_uri,
        ):
            raise DocumentIdentityError(document.id)
        if document.status is DocumentStatus.READY and self._ready_conflict(document):
            raise DuplicateDocumentError(document.source_uri)
        self.items[document.id] = document

    def _ready_conflict(self, document: Document) -> bool:
        return any(
            other.id != document.id
            and other.status is DocumentStatus.READY
            and _source_key(other) == _source_key(document)
            for other in self.items.values()
        )

    async def mark_ready(
        self, tenant_id: TenantId, document_id: DocumentId, chunk_count: int
    ) -> bool:
        # No await between check and write: atomic in the single-threaded event loop.
        pending = self.items.get(document_id)
        if pending is None or pending.tenant_id != tenant_id:
            return False
        if pending.status is not DocumentStatus.PENDING:
            return False
        for other in list(self.items.values()):
            if (
                other.id != pending.id
                and other.status is DocumentStatus.READY
                and _source_key(other) == _source_key(pending)
            ):
                self.items[other.id] = replace(other, status=DocumentStatus.SUPERSEDED)
        self.items[pending.id] = pending.mark_ready(chunk_count)
        return True

    async def update_title(self, tenant_id: TenantId, document_id: DocumentId, title: str) -> bool:
        found = self.items.get(document_id)
        if (
            found is None
            or found.tenant_id != tenant_id
            or found.status is not DocumentStatus.READY
        ):
            return False
        self.items[document_id] = replace(found, title=title)
        return True

    async def get(self, tenant_id: TenantId, document_id: DocumentId) -> Document | None:
        found = self.items.get(document_id)
        return found if found is not None and found.tenant_id == tenant_id else None

    async def find_by_source(
        self, tenant_id: TenantId, profile: str, source_uri: str
    ) -> list[Document]:
        criteria = Criteria.all_of(
            Filter.eq("profile", profile), Filter.eq("source_uri", source_uri)
        )
        return await self.matching(tenant_id, criteria)

    async def delete(self, tenant_id: TenantId, document_id: DocumentId) -> None:
        if await self.get(tenant_id, document_id) is not None:
            del self.items[document_id]

    async def matching(self, tenant_id: TenantId, criteria: Criteria) -> list[Document]:
        ensure_fields(criteria, self.FILTERABLE_FIELDS)
        _ensure_in_cap(criteria.filter)
        scoped = replace(criteria, filter=tenant_scoped(tenant_id, criteria.filter))
        return apply(scoped, self.items.values(), as_record=_document_record)


class InMemoryChunkRepository(ChunkRepository):
    """Chunks with vectors.

    Pass `documents` to mimic the real adapters: only chunks of READY documents are served and
    `add_many` requires the document to exist for the chunk's tenant and profile.
    Without it every stored chunk is searchable (use-case tests that exercise the READY check
    of `RetrieveForTopic` itself rely on that).
    """

    def __init__(self, documents: InMemoryDocumentRepository | None = None) -> None:
        self.items: list[tuple[Chunk, Sequence[float]]] = []
        self._documents = documents

    def _is_served(self, chunk: Chunk) -> bool:
        if self._documents is None:
            return True
        document = self._documents.items.get(chunk.document_id)
        return (
            document is not None
            and document.tenant_id == chunk.tenant_id
            and document.status is DocumentStatus.READY
        )

    async def add_many(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        ensure_single_tenant(chunks)
        if self._documents is not None:  # mimic the composite foreign key of the SQL schema
            for chunk in chunks:
                document = self._documents.items.get(chunk.document_id)
                if (
                    document is None
                    or document.tenant_id != chunk.tenant_id
                    or document.profile != chunk.profile
                ):
                    raise DocumentNotFoundError(chunk.document_id)
        self.items.extend(zip(chunks, vectors, strict=True))

    async def search(
        self,
        tenant_id: TenantId,
        query_vector: Sequence[float],
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        if k <= 0:
            raise ValueError("k must be greater than 0")
        ensure_fields(filter, chunk_filterable_fields(metadata_fields))
        expr = tenant_scoped(tenant_id, filter)
        hits = [
            ScoredChunk(chunk, cosine(query_vector, vector))
            for chunk, vector in self.items
            if self._is_served(chunk) and matches(expr, _chunk_record(chunk))
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[:k]

    async def search_lexical(
        self,
        tenant_id: TenantId,
        query_text: str,
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        return self._text_search(tenant_id, query_text, filter, k, metadata_fields, lexical_score)

    async def search_bm25(
        self,
        tenant_id: TenantId,
        query_text: str,
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        return self._text_search(tenant_id, query_text, filter, k, metadata_fields, any_term_score)

    def _text_search(
        self,
        tenant_id: TenantId,
        query_text: str,
        filter: FilterExpr | None,
        k: int,
        metadata_fields: Sequence[MetadataField],
        score_of: Callable[[Sequence[str], str], float],
    ) -> list[ScoredChunk]:
        if k <= 0:
            raise ValueError("k must be greater than 0")
        ensure_fields(filter, chunk_filterable_fields(metadata_fields))
        expr = tenant_scoped(tenant_id, filter)
        query_terms = lexical_terms(query_text)
        hits = [
            ScoredChunk(chunk, score)
            for chunk, _ in self.items
            if self._is_served(chunk)
            and matches(expr, _chunk_record(chunk))
            and (score := score_of(query_terms, chunk.text)) > 0.0
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[:k]

    async def delete_by_document(self, tenant_id: TenantId, document_id: DocumentId) -> None:
        self.items = [
            (c, v)
            for c, v in self.items
            if not (c.tenant_id == tenant_id and c.document_id == document_id)
        ]


class FakeParser(Parser):
    """Splits the UTF-8 content on blank lines; one section per paragraph, page = index + 1."""

    def __init__(self, title: str | None = None) -> None:
        self.calls = 0
        self.title = title

    async def parse(self, source: SourceFile) -> ParsedDocument:
        self.calls += 1
        paragraphs = [p for p in source.content.decode().split("\n\n") if p.strip()]
        sections = tuple(ParsedSection(text=p, page=i + 1) for i, p in enumerate(paragraphs))
        return ParsedDocument(title=self.title or f"title of {source.uri}", sections=sections)


class FakeChunker(Chunker):
    """One chunk per section, carrying the page as metadata."""

    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for section in parsed.sections:
            metadata: Mapping[str, str | int | float] = (
                {} if section.page is None else {"page": section.page}
            )
            drafts.append(ChunkDraft(text=section.text, metadata=metadata))
        return drafts


class FakeCleaner(Cleaner):
    """Drops the sections containing `params["drop"]` and records every call."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, ParamValue]] = []

    async def clean(
        self, parsed: ParsedDocument, params: Mapping[str, ParamValue]
    ) -> ParsedDocument:
        self.calls.append(params)
        marker = params.get("drop")
        sections = tuple(
            s for s in parsed.sections if not (isinstance(marker, str) and marker in s.text)
        )
        return replace(parsed, sections=sections)


class InMemoryIngestLock(IngestLock):
    """One asyncio.Lock per (tenant, profile, source_uri) key."""

    def __init__(self) -> None:
        self._locks: dict[tuple[str, str, str], asyncio.Lock] = {}

    @asynccontextmanager
    async def hold(self, tenant_id: TenantId, profile: str, source_uri: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault((tenant_id, profile, source_uri), asyncio.Lock())
        async with lock:
            yield


class NullIngestLock(IngestLock):
    """No serialization: used to prove the repositories alone protect against concurrent ingests."""

    @asynccontextmanager
    async def hold(self, tenant_id: TenantId, profile: str, source_uri: str) -> AsyncIterator[None]:
        yield
