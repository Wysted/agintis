"""Repository ports of the knowledge module.

Tenant contract: every operation is tenant-scoped, either through an explicit `tenant_id`
argument or, for `DocumentRepository.save` and `ChunkRepository.add_many`, through the
`tenant_id` carried by the entities. Adapters MUST inject the tenant into the query with
`tenant_scoped(tenant_id, expr)` (inside their Criteria converter) so no caller can read or
touch another tenant's data; the caller's expression stays ONE operand of an AND, so a
top-level AnyOf cannot escape the restriction.

Filterable fields are declared per port (`FILTERABLE_FIELDS`); filters or orders naming
anything else must raise `UnknownFieldError` (`ensure_fields`). Chunk filters are profile-aware:
the allowed set is the universal chunk fields UNION the built-in metadata fields UNION the
`filterable` metadata fields the caller's profile declares (`chunk_filterable_fields`).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import ClassVar

from agintis.modules.knowledge.domain.fusion import (
    DEFAULT_BRANCH_DEPTH,
    DEFAULT_RRF_K,
    FusedChunk,
    reciprocal_rank_fusion,
)
from agintis.modules.knowledge.domain.models import (
    BUILTIN_METADATA_FIELDS,
    RESERVED_CHUNK_FIELDS,
    Chunk,
    Document,
    MetadataField,
    ScoredChunk,
    chunk_metadata_schema,
)
from agintis.shared_kernel.criteria import Criteria, FilterExpr
from agintis.shared_kernel.ids import DocumentId, TenantId

DOCUMENT_FILTERABLE_FIELDS = frozenset(
    {
        "id",
        "tenant_id",
        "profile",
        "source_uri",
        "title",
        "mime_type",
        "content_hash",
        "profile_fingerprint",
        "created_at",
        "status",
        "chunk_count",
    }
)

# Universal chunk fields plus the built-in metadata fields (e.g. `page`). Profile-declared
# metadata fields are added per call, see `chunk_filterable_fields`.
CHUNK_FILTERABLE_FIELDS = RESERVED_CHUNK_FIELDS | frozenset(
    f.name for f in BUILTIN_METADATA_FIELDS if f.filterable
)


def chunk_filterable_fields(declared: Iterable[MetadataField]) -> frozenset[str]:
    """Fields a chunk filter may use for a profile declaring `declared` metadata fields."""
    metadata = frozenset(f.name for f in chunk_metadata_schema(declared).values() if f.filterable)
    return RESERVED_CHUNK_FIELDS | metadata


def ensure_single_tenant(chunks: Sequence[Chunk]) -> TenantId | None:
    """Return the shared tenant of `chunks` (None when empty); ValueError when mixed."""
    tenants = {chunk.tenant_id for chunk in chunks}
    if len(tenants) > 1:
        raise ValueError("add_many requires all chunks to belong to the same tenant")
    return next(iter(tenants), None)


class DocumentRepository(ABC):
    """Persistence of documents.

    Identity: a document is identified by (tenant_id, profile, source_uri). SQL adapters
    MUST enforce it with a PARTIAL unique index, so a new version can be built while the
    previous one is still served:

        CREATE UNIQUE INDEX ... ON documents (tenant_id, profile, source_uri)
        WHERE status = 'ready';

    PENDING and SUPERSEDED rows are not constrained. Violations raise
    `DuplicateDocumentError`. The content hash is NOT an identity: it only decides, per
    source, whether re-ingesting is a no-op.

    Status transitions are CONDITIONAL (`mark_ready`) and `save` is tenant-guarded; callers
    must never emulate a transition by re-saving a stale snapshot (that can resurrect a
    document whose chunks were already cleaned up by a concurrent ingest).
    """

    FILTERABLE_FIELDS: ClassVar[frozenset[str]] = DOCUMENT_FILTERABLE_FIELDS

    @abstractmethod
    async def save(self, document: Document) -> None:
        """Insert or update (upsert by id), tenant-guarded.

        An existing id owned by ANOTHER tenant raises `TenantMismatchError` and leaves the row
        untouched. SQL adapters: `INSERT ... ON CONFLICT (id) DO UPDATE SET ... WHERE
        documents.tenant_id = excluded.tenant_id` and treat 0 affected rows as the error.
        The identity columns (`tenant_id`, `profile`, `source_uri`) are immutable: re-saving an
        existing id with a different profile or source raises `DocumentIdentityError`.
        Saving a READY document while another READY one exists for the same source raises
        `DuplicateDocumentError`.
        """

    @abstractmethod
    async def mark_ready(
        self, tenant_id: TenantId, document_id: DocumentId, chunk_count: int
    ) -> bool:
        """Atomically promote a PENDING document to READY; return False when it did not apply.

        Conditional transition: only a PENDING row of `tenant_id` is promoted (SQL:
        `UPDATE ... WHERE id = ? AND tenant_id = ? AND status = 'pending'`). A missing,
        foreign, already READY or SUPERSEDED document returns False and changes nothing.

        In the SAME atomic operation (one statement/transaction) any other READY document of
        the same (tenant, profile, source_uri) becomes SUPERSEDED. That is what lets the
        partial unique index hold while keeping exactly one served version at all times.
        """

    @abstractmethod
    async def update_title(self, tenant_id: TenantId, document_id: DocumentId, title: str) -> bool:
        """Change the title of a READY document of `tenant_id`; False when nothing matched.

        The title is display metadata (it is read at query time for citations): it does not
        touch chunks, hashes or the status, and needs no re-embedding.
        """

    @abstractmethod
    async def get(self, tenant_id: TenantId, document_id: DocumentId) -> Document | None: ...

    @abstractmethod
    async def find_by_source(
        self, tenant_id: TenantId, profile: str, source_uri: str
    ) -> list[Document]:
        """Return every version (any status) ingested from `source_uri` under `profile`."""

    @abstractmethod
    async def delete(self, tenant_id: TenantId, document_id: DocumentId) -> None:
        """Delete the document (no-op when missing). Chunks are removed via ChunkRepository."""

    @abstractmethod
    async def matching(self, tenant_id: TenantId, criteria: Criteria) -> list[Document]: ...


class ChunkRepository(ABC):
    FILTERABLE_FIELDS: ClassVar[frozenset[str]] = CHUNK_FILTERABLE_FIELDS

    @abstractmethod
    async def add_many(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Persist chunks with their embedding vectors (same length and order).

        All chunks must share one `tenant_id` (`ensure_single_tenant`); mixed tenants raise
        ValueError. Every chunk must point to an existing document of ITS tenant and profile;
        otherwise `DocumentNotFoundError` (missing and foreign-tenant are indistinguishable on
        purpose). SQL adapters map the composite foreign key violation to it.
        """

    @abstractmethod
    async def search(
        self,
        tenant_id: TenantId,
        query_vector: Sequence[float],
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        """Return up to `k` most similar chunks (best first) matching `filter`.

        Only chunks of READY documents are returned (PENDING and SUPERSEDED versions are never
        served). `metadata_fields` is the metadata schema of the caller's profile: the filter
        may name core fields, built-in metadata and the declared `filterable` fields
        (`chunk_filterable_fields`); anything else raises `UnknownFieldError`. Adapters use the
        declared types to compare values.

        Scores are SIMILARITY, higher is better; adapters convert distances (cosine
        distance -> 1 - distance). May return fewer than `k` even when more matches exist
        (ANN post-filtering).
        """

    @abstractmethod
    async def search_lexical(
        self,
        tenant_id: TenantId,
        query_text: str,
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        """Return up to `k` chunks whose text matches `query_text` (full-text), best first.

        Same contract as `search` for tenant scoping, READY-only serving and filters; only the
        ranking differs: the score is a lexical relevance (higher is better, NOT comparable
        with the similarity of `search`) and ties break by chunk id. Terms are combined with AND
        (web-search syntax); a query without searchable terms returns an empty list.
        """

    @abstractmethod
    async def search_bm25(
        self,
        tenant_id: TenantId,
        query_text: str,
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        """Return up to `k` chunks ranked by BM25 against `query_text`, best first.

        Same contract as `search_lexical` for tenant scoping, READY-only serving, filters and
        blank queries; terms are combined with OR (BM25 ranks chunks matching ANY term) and the
        score is the BM25 relevance (higher is better, comparable only within one result list).
        """

    async def search_hybrid(
        self,
        tenant_id: TenantId,
        query_vector: Sequence[float],
        query_text: str,
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
        branch_depth: int = DEFAULT_BRANCH_DEPTH,
        rrf_k: int = DEFAULT_RRF_K,
        bm25: bool = False,
        lexical_weight: float = 1.0,
    ) -> list[FusedChunk]:
        """Dense + lexical search fused with Reciprocal Rank Fusion, best first.

        The lexical branch is `search_lexical`, or `search_bm25` when `bm25` is set, weighted by
        `lexical_weight` in the fusion (dense = 1). Each branch runs with the SAME tenant scope
        and `filter` (so a chunk excluded by a filter can never enter through the other branch)
        and fetches `max(branch_depth, k)` candidates; the fused list is cut to `k`, ordered by
        fused score then chunk id. Implemented once here, on top of the branch searches, so
        every adapter fuses the same way and only the branches are adapter-specific.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        depth = max(branch_depth, k)
        dense = await self.search(
            tenant_id, query_vector, filter, depth, metadata_fields=metadata_fields
        )
        lexical_search = self.search_bm25 if bm25 else self.search_lexical
        lexical = await lexical_search(
            tenant_id, query_text, filter, depth, metadata_fields=metadata_fields
        )
        return reciprocal_rank_fusion(
            dense, lexical, k=rrf_k, limit=k, lexical_weight=lexical_weight
        )

    async def repair_bm25_index(self, tenant_id: TenantId) -> None:
        """Rebuild what `search_bm25` needs for a tenant that already has chunks, where the
        backend keeps per-tenant structures (`add_many` builds them for new tenants). An ingest
        run calls it once, so documents skipped as unchanged still get it. Default: nothing."""
        return None

    @abstractmethod
    async def delete_by_document(self, tenant_id: TenantId, document_id: DocumentId) -> None: ...
