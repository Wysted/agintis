"""Postgres `ChunkRepository`: pgvector HNSW search, tenant-scoped, READY documents only."""

import logging
import math
import re
from collections.abc import Sequence
from typing import Any

from psycopg import errors as pg_errors
from sqlalchemy import (
    BindParameter,
    Float,
    RowMapping,
    Select,
    String,
    TextClause,
    and_,
    bindparam,
    cast,
    delete,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agintis.modules.knowledge.domain import (
    Chunk,
    ChunkRepository,
    DocumentNotFoundError,
    DocumentStatus,
    MetadataField,
    ScoredChunk,
    chunk_filterable_fields,
    ensure_single_tenant,
)
from agintis.modules.knowledge.infrastructure.postgres.fields import chunk_fields
from agintis.modules.knowledge.infrastructure.postgres.indexes import DEFAULT_LOCK_TIMEOUT_SECONDS
from agintis.modules.knowledge.infrastructure.postgres.tables import (
    DOCUMENT_FK,
    TEXT_SEARCH_CONFIG,
    bm25_index_name,
    chunks,
    documents,
)
from agintis.modules.knowledge.infrastructure.postgres.text_search import TextSearchConfigError
from agintis.platform.db import UnitOfWork, compile_filter
from agintis.shared_kernel.criteria import FilterExpr, ensure_fields, tenant_scoped
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId

logger = logging.getLogger(__name__)

# pgvector defaults, overridable per deployment (DATABASE_HNSW_EF_SEARCH / _MAX_SCAN_TUPLES).
DEFAULT_EF_SEARCH = 40
DEFAULT_MAX_SCAN_TUPLES = 20_000
MAX_EF_SEARCH = 1000  # pgvector's upper bound for hnsw.ef_search
# Server-side cap per SEARCH statement (ms). Applied per transaction, not per connection, so the
# ingest paths that share the pool (bulk inserts, mark_ready) are never affected.
DEFAULT_SEARCH_TIMEOUT_MS = 10_000

# A (schema-qualified) text search configuration name. It is bound as a parameter anyway; the
# pattern only rejects obviously wrong deployment settings early.
_TEXT_SEARCH_CONFIG = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?")

# Code-point order, like Python: the SQL tiebreak and the Python re-sort agree.
_ID_ORDER = chunks.c.id.collate("C")

_CHUNK_COLUMNS = (
    chunks.c.id,
    chunks.c.document_id,
    chunks.c.tenant_id,
    chunks.c.profile,
    chunks.c.ordinal,
    chunks.c.text,
    chunks.c.metadata,
)


def search_settings_statement() -> TextClause:
    """Transaction-local (`set_config(..., true)` = SET LOCAL) knobs of the ANN scan.

    pgvector filters AFTER the ANN scan, so iterative scans keep walking the graph until k rows
    survive the filters, bounded by `max_scan_tuples`. `relaxed_order` may emit rows slightly out
    of order (the search sorts them again). `statement_timeout` caps the search statements that
    follow in the same transaction. Bound parameters make the values injection-proof.
    """
    return text(
        "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true), "
        "set_config('hnsw.ef_search', :ef_search, true), "
        "set_config('hnsw.max_scan_tuples', :max_scan_tuples, true), "
        "set_config('statement_timeout', :statement_timeout, true)"
    )


def lexical_settings_statement() -> TextClause:
    """Transaction-local statement timeout of the full-text and BM25 branches."""
    return text("SELECT set_config('statement_timeout', :statement_timeout, true)")


# First ingest of a tenant: is its BM25 index missing while pg_textsearch is installed AND loaded?
# The same statement hands the DO block below its inputs as transaction-local settings: a DO block
# takes no bind parameters, and the tenant id is never spliced into SQL in Python.
# shortcut: a plain CREATE INDEX scans the whole chunks table and blocks the writes of EVERY
# tenant while it runs; move to CREATE INDEX CONCURRENTLY (outside a transaction) when chunks
# gets large. And one partial index per tenant: planning and insert cost grow with the tenant
# count (every index predicate is checked) and the index of a tenant left without chunks is
# never dropped; partition chunks by tenant beyond a few hundred tenants.
_BM25_INDEX_NEEDED = text(
    "SELECT set_config('agintis.bm25_index', :index, true), "
    "set_config('agintis.bm25_tenant', :tenant, true), "
    "to_regclass(:index) IS NULL "
    "AND current_setting('pg_textsearch.library_version', true) IS NOT NULL "
    "AND EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch') AS needed"
)
_BM25_SKIP_REASON = "agintis.bm25_skip_reason"
# The build may outlast a role's default statement_timeout (a timeout would fail every retry),
# but waiting for the SHARE lock stays bounded: a queued lock request blocks every later writer
# of chunks. Past `lock_timeout` the ingest fails with a retryable OperationalError.
_BUILD_TIMEOUTS = text(
    "SELECT set_config('statement_timeout', '0', true), "
    "set_config('lock_timeout', :lock_timeout, true)"
)
_CREATE_BM25_INDEX = text(f"""
DO $$
BEGIN
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON chunks USING bm25 (text) '
        'WITH (text_config = %L) WHERE tenant_id = %L',
        current_setting('agintis.bm25_index'), '{TEXT_SEARCH_CONFIG}',
        current_setting('agintis.bm25_tenant'));
EXCEPTION
    -- A concurrent first ingest of the same tenant created it first.
    WHEN duplicate_table OR unique_violation THEN NULL;
    -- Not the owner of chunks: ingest goes on (reported in Python, psycopg drops NOTICEs).
    WHEN insufficient_privilege THEN
        PERFORM set_config('{_BM25_SKIP_REASON}',
            format('role %s does not own chunks', current_user), true);
END
$$
""")
_SKIP_REASON = text(f"SELECT current_setting('{_BM25_SKIP_REASON}', true)")
_BM25_INDEX_AND_CHUNKS = text(
    "SELECT to_regclass(:index) IS NOT NULL AS has_index, "
    "EXISTS (SELECT 1 FROM chunks WHERE tenant_id = :tenant) AS has_chunks"
)


def _sql_literal(value: str) -> BindParameter[str]:
    """`value` rendered as a quoted SQL literal at execution (the dialect escapes it)."""
    return bindparam(None, value, type_=String, unique=True, literal_execute=True)


def _is_document_fk_violation(exc: IntegrityError) -> bool:
    orig = exc.orig
    return (
        isinstance(orig, pg_errors.ForeignKeyViolation) and orig.diag.constraint_name == DOCUMENT_FK
    )


def _order_key(row: RowMapping) -> tuple[bool, float, str]:
    """(distance, id) with NaN distances (zero vectors) last, like Postgres sorts them."""
    distance = float(row["distance"])
    return (math.isnan(distance), 0.0 if math.isnan(distance) else distance, str(row["id"]))


def _to_chunk(row: RowMapping) -> Chunk:
    return Chunk(
        id=ChunkId(row["id"]),
        document_id=DocumentId(row["document_id"]),
        tenant_id=TenantId(row["tenant_id"]),
        profile=row["profile"],
        ordinal=row["ordinal"],
        text=row["text"],
        metadata=row["metadata"],
    )


class PgChunkRepository(ChunkRepository):
    """Chunks + embeddings.

    Only chunks of READY documents are searched, via a JOIN on `documents` (not a denormalized
    status): promoting/superseding a version is then one cheap UPDATE of a single document row
    instead of rewriting every chunk row and its index entries, and the status can never be
    stale. The JOIN also pins the document's tenant to the chunk's.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        ef_search: int = DEFAULT_EF_SEARCH,
        max_scan_tuples: int = DEFAULT_MAX_SCAN_TUPLES,
        text_search_config: str = TEXT_SEARCH_CONFIG,
        search_timeout_ms: int = DEFAULT_SEARCH_TIMEOUT_MS,
        bm25_indexes: bool = False,
        bm25_lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        """`bm25_indexes`: create a tenant's BM25 index at its first ingest (set when a profile
        uses `bm25_hybrid`; otherwise ingest never pays for the index build), waiting at most
        `bm25_lock_timeout_seconds` for the lock on chunks."""
        if not 1 <= ef_search <= MAX_EF_SEARCH:
            raise ValueError(f"ef_search must be between 1 and {MAX_EF_SEARCH}")
        if search_timeout_ms <= 0:
            raise ValueError("search_timeout_ms must be greater than 0")
        if max_scan_tuples <= 0:
            raise ValueError("max_scan_tuples must be greater than 0")
        if _TEXT_SEARCH_CONFIG.fullmatch(text_search_config) is None:
            raise ValueError(f"Invalid text search configuration {text_search_config!r}")
        if bm25_lock_timeout_seconds <= 0:
            raise ValueError("bm25_lock_timeout_seconds must be greater than 0")
        self._factory = session_factory
        self._text_search_config = text_search_config
        self._ef_search = ef_search
        self._max_scan_tuples = max_scan_tuples
        self._search_timeout = str(search_timeout_ms)
        self.creates_bm25_indexes = bm25_indexes
        self._bm25_lock_timeout = str(max(1, int(bm25_lock_timeout_seconds * 1000)))
        self._bm25_skips_logged: set[TenantId] = set()

    async def add_many(self, chunks_: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks_) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        tenant_id = ensure_single_tenant(chunks_)
        if tenant_id is None:
            return
        if self.creates_bm25_indexes:
            await self._ensure_bm25_index(tenant_id)
        rows: list[dict[str, Any]] = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "tenant_id": chunk.tenant_id,
                "profile": chunk.profile,
                "ordinal": chunk.ordinal,
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
                "embedding": list(vector),
            }
            for chunk, vector in zip(chunks_, vectors, strict=True)
        ]
        try:
            async with UnitOfWork(self._factory) as uow:
                await uow.session.execute(insert(chunks), rows)
        except IntegrityError as exc:
            if _is_document_fk_violation(exc):
                # One error for "missing" and "foreign tenant/profile": no existence oracle.
                raise DocumentNotFoundError(chunks_[0].document_id) from exc
            raise

    async def _ensure_bm25_index(self, tenant_id: TenantId) -> None:
        """Create the tenant's BM25 index (see `tables.bm25_index_name`) before its first chunks.

        A no-op without pg_textsearch, so every other retrieval mode keeps working on any
        Postgres. Checked by name first: once the index exists no lock is taken. Its own short
        transaction, before the insert: CREATE INDEX blocks writes to chunks while it runs.
        """
        index = bm25_index_name(tenant_id)
        async with UnitOfWork(self._factory) as uow:
            found = await uow.session.execute(
                _BM25_INDEX_NEEDED, {"index": index, "tenant": tenant_id}
            )
            if not found.mappings().one()["needed"]:
                return
            await uow.session.execute(_BUILD_TIMEOUTS, {"lock_timeout": self._bm25_lock_timeout})
            await uow.session.execute(_CREATE_BM25_INDEX)
            reason = await uow.session.scalar(_SKIP_REASON)
        if reason and tenant_id not in self._bm25_skips_logged:
            self._bm25_skips_logged.add(tenant_id)  # once per tenant, not once per document
            logger.warning(
                "BM25 index %s of tenant %r not created (%s): bm25_hybrid searches of this "
                "tenant fail until the owner of chunks creates it (see revision 0006)",
                index,
                tenant_id,
                reason,
            )

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
        where = compile_filter(tenant_scoped(tenant_id, filter), chunk_fields(metadata_fields))

        distance = chunks.c.embedding.cosine_distance(list(query_vector)).label("distance")
        candidates = (
            select(*_CHUNK_COLUMNS, distance)
            .join(
                documents,
                and_(
                    documents.c.id == chunks.c.document_id,
                    documents.c.tenant_id == chunks.c.tenant_id,
                ),
            )
            .where(documents.c.status == DocumentStatus.READY.value, where)
            .order_by(distance, _ID_ORDER)
            .limit(k)
            # relaxed_order may emit rows slightly out of order: materialize, then sort exactly.
            .cte("candidates")
            .prefix_with("MATERIALIZED")
        )
        # `+ 0` keeps the planner from dropping this "redundant" sort (Postgres 17+ knows the
        # CTE is already ordered by `distance`); see the pgvector README on relaxed_order.
        stmt = select(candidates).order_by(candidates.c.distance + 0, candidates.c.id.collate("C"))

        async with UnitOfWork(self._factory) as uow:
            await uow.session.execute(search_settings_statement(), self._search_settings(k))
            rows = (await uow.session.execute(stmt)).mappings().all()
        # Defense in depth: never trust the order the database handed back.
        ordered = sorted(rows, key=_order_key)
        return [ScoredChunk(_to_chunk(row), 1.0 - float(row["distance"])) for row in ordered]

    def _search_settings(self, k: int) -> dict[str, str]:
        """Bound parameters of `search_settings_statement` for a search of `k` rows.

        `ef_search` is raised to `k` (up to pgvector's limit): an ANN scan returns at most
        `ef_search` candidates per pass, so a hybrid branch asking for more than the configured
        value would otherwise rely on iterative scans alone.
        """
        return {
            "ef_search": str(min(max(self._ef_search, k), MAX_EF_SEARCH)),
            "max_scan_tuples": str(self._max_scan_tuples),
            "statement_timeout": self._search_timeout,
        }

    async def search_lexical(
        self,
        tenant_id: TenantId,
        query_text: str,
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        """Full-text branch: `tsv @@ plainto_tsquery(config, q)` ranked by `ts_rank_cd`.

        `plainto_tsquery` ANDs the terms and treats everything else as text: the query is
        user input, so `OR`, `-term` and `"phrases"` are not operators. The GIN index on the
        generated `tsv` serves the match. The query configuration is a deployment setting
        (default `public.es_unaccent`; the boot check verifies it is the one `tsv` is built with);
        the query text and the configuration name are bound parameters. READY-only through the same
        JOIN as the dense branch, tenant + filter in the same WHERE.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        ensure_fields(filter, chunk_filterable_fields(metadata_fields))
        if not query_text.strip():
            return []
        where = compile_filter(tenant_scoped(tenant_id, filter), chunk_fields(metadata_fields))
        query = func.plainto_tsquery(cast(self._text_search_config, REGCONFIG), query_text)
        rank = func.ts_rank_cd(chunks.c.tsv, query).label("rank")
        stmt = (
            select(*_CHUNK_COLUMNS, rank)
            .join(
                documents,
                and_(
                    documents.c.id == chunks.c.document_id,
                    documents.c.tenant_id == chunks.c.tenant_id,
                ),
            )
            .where(
                documents.c.status == DocumentStatus.READY.value,
                chunks.c.tsv.bool_op("@@")(query),
                where,
            )
            .order_by(rank.desc(), _ID_ORDER)
            .limit(k)
        )
        async with UnitOfWork(self._factory) as uow:
            await uow.session.execute(
                lexical_settings_statement(), {"statement_timeout": self._search_timeout}
            )
            rows = (await uow.session.execute(stmt)).mappings().all()
        # Defense in depth, like the dense branch: best rank first, ties by id (code points).
        ordered = sorted(rows, key=lambda r: (-float(r["rank"]), str(r["id"])))
        return [ScoredChunk(_to_chunk(row), float(row["rank"])) for row in ordered]

    async def search_bm25(
        self,
        tenant_id: TenantId,
        query_text: str,
        filter: FilterExpr | None,
        k: int,
        *,
        metadata_fields: Sequence[MetadataField] = (),
    ) -> list[ScoredChunk]:
        """BM25 branch (pg_textsearch): `text <@> to_bm25query(q, tenant's index)`, best first.

        Each tenant has its own partial index (`tables.bm25_index_name`), so BM25 statistics are
        the tenant's own. `<@>` returns the NEGATED BM25 score (ascending = best first), which
        lets the index serve the ORDER BY ... LIMIT (an incremental sort adds the id tiebreak);
        the other filters are applied after scoring and the scan keeps going until LIMIT rows
        survive. The tenant and the index name are rendered as SQL literals: a partial index
        serves a plan only when the planner can prove `tenant_id = '<tenant>'`, which a generic
        plan (a statement prepared after a few runs) cannot do from a bound parameter. Only chunks
        matching some query term are kept, in Python: a non-matching row scores 0 when the
        planner does not use the index, and 0 sorts after every match, so dropping it after the
        LIMIT is the same result. Not in SQL: a `<@>` condition in WHERE is evaluated per
        scanned row (EXPLAIN shows it as a Filter re-scoring every row the index returns). The
        selected score is still computed per RETURNED row: pg_textsearch reuses the index's
        cached score only for an ORDER BY expression that is not selected.
        The index fixes the text configuration; tenant, filter, READY-only and the statement
        timeout as in `search_lexical`. A tenant without an index returns [] when it has no
        chunks and raises `TextSearchConfigError` when it has some (ingested before pg_textsearch
        was loaded, or by a role that does not own chunks).
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        ensure_fields(filter, chunk_filterable_fields(metadata_fields))
        if not query_text.strip():
            return []
        where = compile_filter(tenant_scoped(tenant_id, filter), chunk_fields(metadata_fields))
        index = bm25_index_name(tenant_id)
        # shortcut: the index covers ALL the tenant's rows, so IDF counts every profile and the
        # pending/superseded versions too, and profile/metadata/READY filters apply after
        # scoring (bounded by the tenant's rows and statement_timeout). Move to indexes per
        # (tenant, profile) or partitioning when a tenant's profiles differ a lot in vocabulary.
        distance = chunks.c.text.op("<@>", return_type=Float)(
            func.to_bm25query(query_text, _sql_literal(index))
        )
        score = distance.label("score")
        stmt = (
            select(*_CHUNK_COLUMNS, score)
            .join(
                documents,
                and_(
                    documents.c.id == chunks.c.document_id,
                    documents.c.tenant_id == chunks.c.tenant_id,
                ),
            )
            .where(
                documents.c.status == DocumentStatus.READY.value,
                chunks.c.tenant_id == _sql_literal(tenant_id),
                where,
            )
            .order_by(score, _ID_ORDER)
            .limit(k)
        )
        try:
            rows = await self._bm25_rows(stmt)
        except DBAPIError as exc:
            # to_bm25query fails at plan time on a missing index (undefined_object), rows or not.
            if not isinstance(exc.orig, pg_errors.UndefinedObject):
                raise
            async with UnitOfWork(self._factory) as uow:
                found = await uow.session.execute(
                    _BM25_INDEX_AND_CHUNKS, {"index": index, "tenant": tenant_id}
                )
                state = found.mappings().one()
            if state["has_index"]:  # a first ingest committed it meanwhile: retry once
                rows = await self._bm25_rows(stmt)
            elif state["has_chunks"]:
                quoted = tenant_id.replace("'", "''")
                raise TextSearchConfigError(
                    f"Tenant {tenant_id!r} has chunks but no BM25 index {index} (ingested before "
                    "a profile used bm25_hybrid or pg_textsearch was loaded, or by a role that "
                    "does not own chunks): as the owner of chunks run `CREATE INDEX IF NOT "
                    f"EXISTS {index} ON chunks USING bm25 (text) WITH (text_config = "
                    f"'{TEXT_SEARCH_CONFIG}') WHERE tenant_id = '{quoted}'`, or re-ingest a "
                    "changed document while a profile uses bm25_hybrid"
                ) from exc
            else:
                return []
        # Defense in depth: best (most negative) first, ties by id (code points).
        matching = [row for row in rows if float(row["score"]) < 0]
        ordered = sorted(matching, key=lambda r: (float(r["score"]), str(r["id"])))
        return [ScoredChunk(_to_chunk(row), -float(row["score"])) for row in ordered]

    async def _bm25_rows(self, stmt: Select[Any]) -> Sequence[RowMapping]:
        async with UnitOfWork(self._factory) as uow:
            await uow.session.execute(
                lexical_settings_statement(), {"statement_timeout": self._search_timeout}
            )
            rows: Sequence[RowMapping] = (await uow.session.execute(stmt)).mappings().all()
        return rows

    async def delete_by_document(self, tenant_id: TenantId, document_id: DocumentId) -> None:
        stmt = delete(chunks).where(
            chunks.c.tenant_id == tenant_id, chunks.c.document_id == document_id
        )
        async with UnitOfWork(self._factory) as uow:
            await uow.session.execute(stmt)
