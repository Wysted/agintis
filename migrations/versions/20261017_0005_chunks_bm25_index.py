"""knowledge: BM25 index on chunks.text (pg_textsearch), only where the extension is available

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-17

Serves `retrieval_mode: bm25_hybrid`. pg_textsearch (Timescale, PostgreSQL license) ships with
the project's Postgres image (docker/postgres) but not with managed Postgres (RDS, Cloud SQL), so
this revision is a NO-OP (NOTICE) where the extension is not available, where the server does not
preload its library, and where the migration role cannot create it. Profiles on `bm25_hybrid`
then fail at boot with a clear error; the other retrieval modes never need it. To add it later,
as a privileged role on a server that preloads the library
(`shared_preload_libraries = 'pg_textsearch'`):

    CREATE EXTENSION IF NOT EXISTS pg_textsearch;
    CREATE INDEX IF NOT EXISTS ix_chunks_text_bm25 ON chunks
        USING bm25 (text) WITH (text_config = 'public.es_unaccent');

The text configuration is the one `chunks.tsv` uses (revision 0001), so both lexical legs
tokenize alike.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# shortcut: one index over every tenant and profile, so BM25 IDF statistics are global (a big
# tenant shifts another tenant's term weights; filters apply after scoring); move to partial
# indexes per tenant (or partitioning) when tenants with different vocabularies share a database.
CREATE_BM25_INDEX = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_textsearch') THEN
        RAISE NOTICE 'pg_textsearch is not available: skipping the BM25 index '
            '(retrieval_mode bm25_hybrid will refuse to boot)';
        RETURN;
    END IF;
    -- Set by the library once loaded: the same test the extension's install script fails on.
    IF current_setting('pg_textsearch.library_version', true) IS NULL THEN
        RAISE NOTICE 'pg_textsearch is not in shared_preload_libraries: skipping the BM25 index';
        RETURN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch') THEN
        BEGIN
            CREATE EXTENSION pg_textsearch;
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE NOTICE 'role % cannot CREATE EXTENSION pg_textsearch: skipping the BM25 index',
                current_user;
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


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_text_bm25")
    # The extension is kept, like vector/unaccent in 0001: shared infrastructure.
