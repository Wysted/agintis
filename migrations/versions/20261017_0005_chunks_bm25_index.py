"""knowledge: BM25 index on chunks.text (pg_textsearch), only where the extension is available

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-17

Serves `retrieval_mode: bm25_hybrid`. pg_textsearch (Timescale, PostgreSQL license) ships with
the project's Postgres image (docker/postgres) but not with managed Postgres (RDS, Cloud SQL), so
this revision is a NO-OP (logged as a WARNING) where the extension is not available, where the
server does not preload its library, and where the migration role cannot create it. Profiles on
`bm25_hybrid` then fail at boot with a clear error; the other retrieval modes never need it. A
compose DB started before the image existed must be rebuilt first
(`docker compose up -d --build db`), otherwise this revision is skipped. To add it later,
as a privileged role on a server that preloads the library
(`shared_preload_libraries = 'pg_textsearch'`):

    CREATE EXTENSION IF NOT EXISTS pg_textsearch;
    CREATE INDEX IF NOT EXISTS ix_chunks_text_bm25 ON chunks
        USING bm25 (text) WITH (text_config = 'public.es_unaccent');

The text configuration is the one `chunks.tsv` uses (revision 0001), so both lexical legs
tokenize alike. Revision 0006 replaces this index with one per tenant.
"""

import logging
from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger("alembic.runtime.migration")

# The DO block reports why it skipped through this transaction-local setting: psycopg drops
# server NOTICEs, so the reason is logged from Python.
SKIP_REASON_SETTING = "agintis.bm25_skip_reason"

# One index over every tenant mixes their BM25 statistics: revision 0006 replaces it with one
# partial index per tenant.
CREATE_BM25_INDEX = f"""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_textsearch') THEN
        PERFORM set_config('{SKIP_REASON_SETTING}',
            'the pg_textsearch extension is not available', true);
        RETURN;
    END IF;
    -- Set by the library once loaded: the same test the extension's install script fails on.
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
    CREATE INDEX IF NOT EXISTS ix_chunks_text_bm25 ON chunks
        USING bm25 (text) WITH (text_config = 'public.es_unaccent');
END
$$
"""


def upgrade() -> None:
    op.execute(CREATE_BM25_INDEX)
    if context.is_offline_mode():  # `--sql`: nothing ran, nothing to report
        return
    reason = (
        op.get_bind()
        .execute(text("SELECT current_setting(:name, true)"), {"name": SKIP_REASON_SETTING})
        .scalar()
    )
    if reason:
        logger.warning(
            "BM25 index skipped (%s): retrieval_mode bm25_hybrid will refuse to boot; other "
            "modes are unaffected. See revision 0005 to add it later.",
            reason,
        )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_text_bm25")
    # The extension is kept, like vector/unaccent in 0001: shared infrastructure.
