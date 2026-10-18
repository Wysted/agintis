"""Postgres adapters of the knowledge module (SQLAlchemy Core + pgvector + advisory locks)."""

from agintis.modules.knowledge.infrastructure.postgres.chunks import PgChunkRepository
from agintis.modules.knowledge.infrastructure.postgres.documents import PgDocumentRepository
from agintis.modules.knowledge.infrastructure.postgres.indexes import (
    ApplyResult,
    CatalogIndex,
    ConflictingIndexError,
    IndexEntry,
    IndexPlan,
    IndexStatus,
    IndexSyncError,
    MetadataIndexSpec,
    MetadataIndexSync,
    diff_indexes,
    expected_metadata_indexes,
    metadata_index_name,
    metadata_index_problems,
)
from agintis.modules.knowledge.infrastructure.postgres.ingest_lock import PgIngestLock
from agintis.modules.knowledge.infrastructure.postgres.text_search import (
    TextSearchConfigError,
    verify_pg_textsearch,
    verify_text_search_config,
)

__all__ = [
    "ApplyResult",
    "CatalogIndex",
    "ConflictingIndexError",
    "IndexEntry",
    "IndexPlan",
    "IndexStatus",
    "IndexSyncError",
    "MetadataIndexSpec",
    "MetadataIndexSync",
    "PgChunkRepository",
    "PgDocumentRepository",
    "PgIngestLock",
    "TextSearchConfigError",
    "diff_indexes",
    "expected_metadata_indexes",
    "metadata_index_name",
    "metadata_index_problems",
    "verify_pg_textsearch",
    "verify_text_search_config",
]
