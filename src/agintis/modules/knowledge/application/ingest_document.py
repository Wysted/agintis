"""Use case: parse, chunk, embed and store a document (idempotent, atomic from the caller's view).

Identity of a document = (tenant_id, profile, source_uri). The content hash only decides
whether re-ingesting the SAME source is a no-op; equal content from different sources is kept
as separate documents.

Flow (everything after step 1 runs under `IngestLock` for the source)
1. Resolve strategies and hash the content and the caller's document metadata.
2. Read the versions of the source. READY with the same content hash, profile fingerprint
   (profile plus the `version` of each strategy that will run) AND metadata hash -> skipped.
   A row stored before the metadata hash existed (NULL) never matches: it is reprocessed once.
   A skipped document still gets its title refreshed when the caller's title changed (the title
   is display data: no re-parse, no re-embedding).
3. Parse, run the profile's cleaners in order, chunk, resolve the declared metadata fields from
   their sources (regex groups, the caller's `document_metadata`, cleaner-collected values,
   constants, extractors), check chunk metadata against the declared types (coercing regex
   group strings to int/float, ISO dates; `InvalidMetadataError` otherwise) and embed BEFORE
   touching storage (the fallible, expensive part); an empty result raises EmptyDocumentError
   without side effects.
4. Delete leftovers (PENDING of crashed attempts, SUPERSEDED garbage); safe only under the lock.
5. Save the new document as PENDING, add chunks, `mark_ready` (conditional PENDING -> READY that
   atomically supersedes the previous READY version). If the transition does not apply,
   `ConcurrentIngestError`. On any failure the new document and its chunks are deleted
   (compensation, errors logged and never masking the original) and the error is re-raised.
6. Delete the superseded version (chunks + document). Best effort: a failure here is logged
   and the garbage is removed by the next ingest of the source.

The previous version keeps being served until step 5 swaps it: retrieval never sees a window
with zero versions, nor a READY document without its chunks.
"""

import asyncio
import hashlib
import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace

from langchain_core.embeddings import Embeddings

from agintis.modules.knowledge.application.dto import (
    ChunkDraft,
    IngestResult,
    IngestStatus,
    SourceFile,
    metadata_hash,
)
from agintis.modules.knowledge.application.errors import (
    ConcurrentIngestError,
    EmptyDocumentError,
    InvalidProfileError,
    UnknownStrategyError,
)
from agintis.modules.knowledge.application.metadata import MetadataResolver
from agintis.modules.knowledge.application.ports import Chunker, Cleaner, IngestLock, Parser
from agintis.modules.knowledge.domain import (
    Chunk,
    ChunkRepository,
    CleanerSpec,
    Document,
    DocumentRepository,
    DocumentStatus,
    InvalidMetadataError,
    RagProfile,
    chunk_id_for,
    conform_metadata,
    display_title,
)
from agintis.shared_kernel.criteria import Criteria, Filter
from agintis.shared_kernel.ids import DocumentId, TenantId, new_id
from agintis.shared_kernel.ports import Clock

DEFAULT_EMBED_BATCH_SIZE = 32
TITLE_METADATA_KEY = "title"

logger = logging.getLogger(__name__)


def _resolve[T](registry: Mapping[str, T], key: str, kind: str) -> T:
    try:
        return registry[key]
    except KeyError:
        raise UnknownStrategyError(f"No {kind} registered for key {key!r}") from None


def caller_title(document_metadata: Mapping[str, object]) -> str | None:
    """The caller's `document_metadata["title"]` as a display title (`display_title`); None when
    it is not a string or nothing is left of it."""
    value = document_metadata.get(TITLE_METADATA_KEY)
    if not isinstance(value, str):
        return None
    return display_title(value) or None


def _strategy_versions(
    profile: RagProfile,
    parser: Parser,
    chunker: Chunker,
    cleaners: Sequence[tuple[Cleaner, CleanerSpec]],
    resolver: MetadataResolver | None = None,
) -> dict[str, int | str]:
    """Implementation versions of the strategies about to run (feed the profile fingerprint)."""
    versions: dict[str, int | str] = {
        f"parser:{profile.parser_key}": parser.version,
        f"chunker:{profile.chunker_key}": chunker.version,
    }
    versions.update(chunker.dependency_versions())
    versions.update({f"cleaner:{spec.key}": cleaner.version for cleaner, spec in cleaners})
    if resolver is not None:
        versions.update(resolver.versions(profile))
    return versions


class IngestDocument:
    def __init__(
        self,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        parsers: Mapping[str, Parser],
        chunkers: Mapping[str, Chunker],
        embeddings: Embeddings,
        clock: Clock,
        lock: IngestLock,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        cleaners: Mapping[str, Cleaner] | None = None,
        metadata: MetadataResolver | None = None,
    ) -> None:
        if embed_batch_size <= 0:
            raise ValueError("embed_batch_size must be greater than 0")
        self._documents = documents
        self._chunks = chunks
        self._parsers = parsers
        self._chunkers = chunkers
        self._cleaners: Mapping[str, Cleaner] = cleaners if cleaners is not None else {}
        self._embeddings = embeddings
        self._clock = clock
        self._lock = lock
        self._metadata = metadata
        self._embed_batch_size = embed_batch_size

    async def execute(
        self, tenant_id: TenantId, profile: RagProfile, source: SourceFile
    ) -> IngestResult:
        # Resolve strategies first so a misconfigured profile fails fast, even on re-ingest.
        parser = _resolve(self._parsers, profile.parser_key, "parser")
        chunker = _resolve(self._chunkers, profile.chunker_key, "chunker")
        cleaners = [
            (_resolve(self._cleaners, spec.key, "cleaner"), spec) for spec in profile.cleaners
        ]
        if self._metadata is None and any(f.source is not None for f in profile.metadata_fields):
            raise InvalidProfileError(
                f"Profile {profile.name!r} declares metadata sources but this IngestDocument "
                "has no MetadataResolver: its fields would be silently dropped"
            )

        content_hash = hashlib.sha256(source.content).hexdigest()
        async with self._lock.hold(tenant_id, profile.name, source.uri):
            return await self._ingest_locked(
                tenant_id, profile, source, parser, chunker, cleaners, content_hash
            )

    async def find_rerouted(
        self, tenant_id: TenantId, source_uri: str, keep_profile: str
    ) -> list[Document]:
        """READY documents of `source_uri` that live under a profile other than `keep_profile`.

        When routing changes (a rule edited, a document re-classified), the previous profile's
        version stays READY and is served next to the new one: these are those orphans.
        """
        criteria = Criteria.all_of(
            Filter.eq("source_uri", source_uri),
            Filter.eq("status", DocumentStatus.READY.value),
            Filter.neq("profile", keep_profile),
        )
        return await self._documents.matching(tenant_id, criteria)

    async def find_stale(
        self, tenant_id: TenantId, profile: str, keep_uris: Collection[str]
    ) -> list[Document]:
        """READY documents under `profile` whose source is not in `keep_uris` (removed upstream)."""
        criteria = Criteria.all_of(
            Filter.eq("profile", profile), Filter.eq("status", DocumentStatus.READY.value)
        )
        return [
            d
            for d in await self._documents.matching(tenant_id, criteria)
            if d.source_uri not in keep_uris
        ]

    async def find_by_sources(
        self, tenant_id: TenantId, source_uris: Collection[str]
    ) -> list[Document]:
        """Every version (any status, any profile) of the given sources, for a batch revert."""
        criteria = Criteria.all_of(Filter.in_("source_uri", sorted(source_uris)))
        return await self._documents.matching(tenant_id, criteria)

    async def remove_rerouted(self, tenant_id: TenantId, document: Document) -> bool:
        """Delete an orphan found by `find_rerouted`/`find_stale` (chunks and document), under
        its source lock; False when it is already gone."""
        async with self._lock.hold(tenant_id, document.profile, document.source_uri):
            if await self._documents.get(tenant_id, document.id) is None:
                return False
            await self._remove(document)
            return True

    async def _ingest_locked(
        self,
        tenant_id: TenantId,
        profile: RagProfile,
        source: SourceFile,
        parser: Parser,
        chunker: Chunker,
        cleaners: Sequence[tuple[Cleaner, CleanerSpec]],
        content_hash: str,
    ) -> IngestResult:
        fingerprint = profile.fingerprint_for(
            _strategy_versions(profile, parser, chunker, cleaners, self._metadata)
        )
        versions = await self._documents.find_by_source(tenant_id, profile.name, source.uri)
        previous = next((d for d in versions if d.status is DocumentStatus.READY), None)
        source_metadata_hash = metadata_hash(source.document_metadata)
        if previous is not None and self._is_current(
            previous, content_hash, fingerprint, source_metadata_hash
        ):
            refreshed = await self._refresh_title(previous, source)
            return self._skipped(previous, title_refreshed=refreshed)

        parsed = replace(await parser.parse(source), document_metadata=source.document_metadata)
        for cleaner, spec in cleaners:
            parsed = await cleaner.clean(parsed, spec.params)
        drafts = await chunker.chunk(parsed, profile)
        if not drafts:
            raise EmptyDocumentError(f"No chunks produced for {source.uri!r}")
        if self._metadata is not None:
            drafts = await asyncio.to_thread(self._metadata.resolve, profile, parsed, drafts)
        drafts = self._conform_metadata(drafts, profile, source)
        vectors = await self._embed([draft.text for draft in drafts])

        # Leftovers are only safe to remove while holding the lock.
        for leftover in versions:
            if leftover.status is not DocumentStatus.READY:
                await self._remove(leftover)

        document = Document(
            id=DocumentId(new_id()),
            tenant_id=tenant_id,
            profile=profile.name,
            source_uri=source.uri,
            # Priority: the caller's title > the parser's title (URI fallback inside the parser),
            # both normalized by `display_title`. Applied after chunking, so chunks and the
            # fingerprint never depend on it.
            title=caller_title(source.document_metadata) or display_title(parsed.title),
            mime_type=source.mime_type,
            content_hash=content_hash,
            profile_fingerprint=fingerprint,
            created_at=self._clock.now(),
            metadata_hash=source_metadata_hash,
        )
        chunks = self._build_chunks(document, drafts)
        try:
            await self._documents.save(document)
            await self._chunks.add_many(chunks, vectors)
            if not await self._documents.mark_ready(tenant_id, document.id, len(chunks)):
                raise ConcurrentIngestError(
                    f"Document {document.id!r} for {source.uri!r} is no longer pending"
                )
        except BaseException:
            await self._compensate(document)
            raise

        if previous is not None:
            await self._remove_superseded(previous)
        status: IngestStatus = "replaced" if previous is not None else "indexed"
        return IngestResult(document_id=document.id, chunk_count=len(chunks), status=status)

    @staticmethod
    def _conform_metadata(
        drafts: Sequence[ChunkDraft], profile: RagProfile, source: SourceFile
    ) -> list[ChunkDraft]:
        conformed: list[ChunkDraft] = []
        for ordinal, draft in enumerate(drafts):
            try:
                metadata = conform_metadata(draft.metadata, profile.metadata_fields)
            except InvalidMetadataError as exc:
                raise InvalidMetadataError(
                    f"Chunk {ordinal} of {source.uri!r} (profile {profile.name!r}): {exc}"
                ) from exc
            conformed.append(replace(draft, metadata=metadata))
        return conformed

    @staticmethod
    def _is_current(
        document: Document, content_hash: str, fingerprint: str, source_metadata_hash: str
    ) -> bool:
        return (
            document.status is DocumentStatus.READY
            and document.content_hash == content_hash
            and document.profile_fingerprint == fingerprint
            and document.metadata_hash == source_metadata_hash
        )

    async def _refresh_title(self, document: Document, source: SourceFile) -> bool:
        title = caller_title(source.document_metadata)
        if title is None or title == document.title:
            return False
        if not await self._documents.update_title(document.tenant_id, document.id, title):
            return False
        logger.info(
            "Refreshed the title of document %s (profile %s): %.60r -> %.60r",
            document.id,
            document.profile,
            document.title,
            title,
        )
        return True

    @staticmethod
    def _skipped(document: Document, *, title_refreshed: bool = False) -> IngestResult:
        return IngestResult(
            document_id=document.id,
            chunk_count=document.chunk_count,
            status="skipped",
            title_refreshed=title_refreshed,
        )

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._embed_batch_size):
            batch = list(texts[start : start + self._embed_batch_size])
            embedded = await self._embeddings.aembed_documents(batch)
            if len(embedded) != len(batch):
                raise ValueError("Embeddings returned a different number of vectors than texts")
            vectors.extend(embedded)
        return vectors

    @staticmethod
    def _build_chunks(document: Document, drafts: Sequence[ChunkDraft]) -> list[Chunk]:
        return [
            Chunk(
                id=chunk_id_for(document.id, ordinal),
                document_id=document.id,
                tenant_id=document.tenant_id,
                profile=document.profile,
                ordinal=ordinal,
                text=draft.text,
                metadata=draft.metadata,
            )
            for ordinal, draft in enumerate(drafts)
        ]

    async def _remove(self, document: Document) -> None:
        await self._chunks.delete_by_document(document.tenant_id, document.id)
        await self._documents.delete(document.tenant_id, document.id)

    async def _remove_superseded(self, document: Document) -> None:
        try:
            await self._remove(document)
        except Exception:
            logger.exception(
                "Could not delete superseded document %s; next ingest cleans it", document.id
            )

    async def _compensate(self, document: Document) -> None:
        """Best-effort rollback; a failure here must never mask the original error."""
        try:
            await self._remove(document)
        except Exception:
            logger.exception("Compensation failed for document %s; row stays PENDING", document.id)
