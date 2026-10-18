"""knowledge: one partial BM25 index per tenant instead of the global one (pg_textsearch)

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-17

BM25 statistics (IDF, average length) are per index: the global `ix_chunks_text_bm25` of revision
0005 let a big tenant shift another tenant's term weights. This revision drops it and creates
`ix_chunks_bm25_<first 16 hex of sha256(tenant_id)>` (`tables.bm25_index_name`, computed here in
SQL) for every tenant already in chunks, with `WHERE tenant_id = '<tenant>'`. Later tenants get
theirs at their first ingest (`PgChunkRepository.add_many`). Like 0005, a NO-OP (logged as a
WARNING) where the extension is not available, where the server does not preload its library and
where the migration role cannot create it. To add the indexes later, as a privileged role on a
server that preloads the library, create the extension and rerun this revision
(`alembic downgrade 0005 && alembic upgrade head`).
"""

import logging
from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")

# Same transaction-local channel as revision 0005: psycopg drops server NOTICEs.
SKIP_REASON_SETTING = "agintis.bm25_skip_reason"

CREATE_TENANT_INDEXES = f"""
DO $$
DECLARE
    tenant text;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_textsearch') THEN
        PERFORM set_config('{SKIP_REASON_SETTING}',
            'the pg_textsearch extension is not available', true);
        RETURN;
    END IF;
    IF current_setting('pg_textsearch.library_version', true) IS NULL THEN
        PERFORM set_config('{SKIP_REASON_SETTING}',
            'pg_textsearch is not in shared_preload_libraries', true);
        RETURN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch') THEN
        BEGIN
            CREATE EXTENSION pg_textsearch;
        EXCEPTION WHEN insufficient_privilege THEN
            PERFORM set_config('{SKIP_REASON_SETTING}',
                format('role %s cannot CREATE EXTENSION pg_textsearch', current_user), true);
            RETURN;
        END;
    END IF;
    -- An array, not a query loop: an open cursor on chunks blocks CREATE INDEX on it.
    FOREACH tenant IN ARRAY ARRAY(SELECT DISTINCT tenant_id FROM chunks) LOOP
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON chunks USING bm25 (text) '
            'WITH (text_config = %L) WHERE tenant_id = %L',
            'ix_chunks_bm25_' || left(encode(sha256(convert_to(tenant, 'UTF8')), 'hex'), 16),
            'public.es_unaccent', tenant);
    END LOOP;
    -- Last: ACCESS EXCLUSIVE on chunks only for this brief drop, not for the builds above.
    DROP INDEX IF EXISTS ix_chunks_text_bm25;
END
$$
"""

# Back to revision 0005: drop every per-tenant bm25 index, recreate the global one where the
# extension is installed and loaded.
DROP_TENANT_INDEXES = """
DO $$
DECLARE
    index_name text;
BEGIN
    FOREACH index_name IN ARRAY ARRAY(
        SELECT c.relname::text FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_am am ON am.oid = c.relam
        WHERE i.indrelid = to_regclass('chunks') AND am.amname = 'bm25'
            AND starts_with(c.relname, 'ix_chunks_bm25_')
    ) LOOP
        EXECUTE format('DROP INDEX %I', index_name);
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch')
            AND current_setting('pg_textsearch.library_version', true) IS NOT NULL THEN
        CREATE INDEX IF NOT EXISTS ix_chunks_text_bm25 ON chunks
            USING bm25 (text) WITH (text_config = 'public.es_unaccent');
    END IF;
END
$$
"""


def upgrade() -> None:
    op.execute(CREATE_TENANT_INDEXES)
    if context.is_offline_mode():  # `--sql`: nothing ran, nothing to report
        return
    reason = (
        op.get_bind()
        .execute(text("SELECT current_setting(:name, true)"), {"name": SKIP_REASON_SETTING})
        .scalar()
    )
    if reason:
        logger.warning(
            "Per-tenant BM25 indexes skipped (%s): retrieval_mode bm25_hybrid will refuse to "
            "boot; other modes are unaffected. See revision 0006 to add them later.",
            reason,
        )


def downgrade() -> None:
    op.execute(DROP_TENANT_INDEXES)
