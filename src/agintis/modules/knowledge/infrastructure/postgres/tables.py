"""SQLAlchemy Core tables of the knowledge module (Alembic autogenerates from these).

The `embedding` column is declared WITHOUT a dimension here on purpose: the width is a
deployment decision (`EMBEDDINGS_EXPECTED_DIM`, default 1024 for bge-m3) applied by the first
migration. Changing the embedding model to another dimension needs a NEW migration (alter the
column, drop/rebuild the HNSW index) plus re-embedding every chunk.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Column,
    Computed,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

from agintis.modules.knowledge.domain import DocumentStatus
from agintis.platform.db import metadata

# Text search configuration shared by the generated `tsv` column and lexical queries.
TEXT_SEARCH_CONFIG = "public.es_unaccent"

READY_SOURCE_INDEX = "uq_documents_ready_source"
DOCUMENT_FK = "fk_chunks_document_tenant"
METADATA_INDEX_PREFIX = "ix_chunks_meta_"
# BM25 index (pg_textsearch) on chunks.text, named by `to_bm25query`. Migration 0005 creates it
# only where the extension is available, so it is not declared on the table below.
BM25_INDEX = "ix_chunks_text_bm25"

_STATUSES = ", ".join(f"'{status.value}'" for status in DocumentStatus)

documents = Table(
    "documents",
    metadata,
    Column("id", Text, primary_key=True),
    Column("tenant_id", Text, nullable=False),
    Column("profile", Text, nullable=False),
    Column("source_uri", Text, nullable=False),
    Column("title", Text, nullable=False),
    Column("mime_type", Text, nullable=False),
    Column("content_hash", Text, nullable=False),
    Column("profile_fingerprint", Text, nullable=False),
    Column("metadata_hash", Text, nullable=True),
    Column("status", Text, nullable=False),
    Column("chunk_count", Integer, nullable=False, server_default=text("0")),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    # Target of the composite FK from chunks: a chunk can only point to a document of ITS tenant
    # AND profile (both immutable on a document, see PgDocumentRepository.save).
    UniqueConstraint("id", "tenant_id", "profile"),
    CheckConstraint(f"status IN ({_STATUSES})", name="status_valid"),
    CheckConstraint("chunk_count >= 0", name="chunk_count_non_negative"),
    # Identity of a source: only ONE ready version; pending/superseded are unconstrained.
    Index(
        READY_SOURCE_INDEX,
        "tenant_id",
        "profile",
        "source_uri",
        unique=True,
        postgresql_where=text("status = 'ready'"),
    ),
    Index("ix_documents_tenant_profile_source", "tenant_id", "profile", "source_uri"),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", Text, primary_key=True),
    Column("document_id", Text, nullable=False),
    Column("tenant_id", Text, nullable=False),
    Column("profile", Text, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("embedding", Vector(), nullable=False),
    Column(
        "tsv",
        TSVECTOR,
        Computed(f"to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, text)", persisted=True),
    ),
    ForeignKeyConstraint(
        ["document_id", "tenant_id", "profile"],
        ["documents.id", "documents.tenant_id", "documents.profile"],
        ondelete="CASCADE",
        name=DOCUMENT_FK,
    ),
    UniqueConstraint("document_id", "ordinal"),
    Index(
        "ix_chunks_embedding_hnsw",
        "embedding",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    ),
    Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    Index(
        "ix_chunks_metadata",
        "metadata",
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
    ),
    Index("ix_chunks_tenant_profile_document", "tenant_id", "profile", "document_id"),
)
