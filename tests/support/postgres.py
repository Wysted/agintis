"""Helpers to run tests against a real, disposable Postgres database.

The test databases (`agintis_test`, `agintis_test_e2e`, ...) are dropped, recreated and migrated
with Alembic once per test session, so schema changes are always exercised and dev data is never
touched. They live on the server of `TEST_DATABASE_URL` (env or `.env`) or, when unset, on the
dev server of `DATABASE_URL`. Only a database whose name ends with `_test` or contains `_test_`
can ever be dropped (`assert_disposable_name`).
"""

import re
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agintis.bootstrap.settings import get_settings
from agintis.platform.db import create_engine, create_unpooled_engine

TEST_DATABASE_NAME = "agintis_test"
# Contract vectors are 2-dimensional; also proves the migration honors the configured width.
TEST_EMBEDDING_DIM = 2
# End-to-end database: bge-m3 width, for tests that embed with the real TEI service.
E2E_DATABASE_NAME = "agintis_test_e2e"
E2E_EMBEDDING_DIM = 1024
# 64 dimensions is enough for pgvector's iterative scans to matter (the 2-dimension contract
# database is trivial); used by the search-plan tests.
HNSW_DATABASE_NAME = "agintis_test_hnsw"
HNSW_EMBEDDING_DIM = 64
MIGRATIONS_DATABASE_NAME = "agintis_test_migrations"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONNECT_TIMEOUT_SECONDS = 3

# (name, predicate) of every bm25 (pg_textsearch) index on chunks.
BM25_INDEXES_SQL = (
    "SELECT c.relname, pg_get_expr(i.indpred, i.indrelid) FROM pg_index i "
    "JOIN pg_class c ON c.oid = i.indexrelid JOIN pg_am am ON am.oid = c.relam "
    "WHERE i.indrelid = 'chunks'::regclass AND am.amname = 'bm25'"
)

_DISPOSABLE_NAME = re.compile(r"[A-Za-z0-9]+(_[A-Za-z0-9]+)*(_test|_test_[A-Za-z0-9_]+)")


class PgTestSettings(BaseSettings):
    """`TEST_DATABASE_URL`: server (and credentials) used for disposable test databases."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    test_database_url: SecretStr | None = None


def assert_disposable_name(name: str) -> None:
    """Refuse to drop anything that is not obviously a test database."""
    if _DISPOSABLE_NAME.fullmatch(name) is None:
        raise RuntimeError(
            f"Refusing to drop database {name!r}: test databases must be named "
            "'*_test' or '*_test_*'"
        )


def dev_database_url() -> URL:
    """SQLAlchemy URL of the dev database (from DATABASE_URL / .env)."""
    return make_url(get_settings().database_url.get_secret_value())


def admin_database_url() -> URL:
    """Server used to create/drop test databases: TEST_DATABASE_URL, else the dev server."""
    configured = PgTestSettings().test_database_url
    if configured is None or not configured.get_secret_value():
        return dev_database_url()
    return make_url(configured.get_secret_value())


def libpq_dsn(url: URL, database: str | None = None) -> str:
    target = url if database is None else url.set(database=database)
    return target.set(drivername="postgresql").render_as_string(hide_password=False)


def recreate_test_database(
    admin_url: URL, name: str = TEST_DATABASE_NAME, embedding_dim: int = TEST_EMBEDDING_DIM
) -> URL:
    """Drop and recreate database `name`, then migrate it to head. Returns its URL."""
    drop_database(admin_url, name)
    with _admin_connection(admin_url, name) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    test_url = admin_url.set(database=name)
    migrate(test_url, embedding_dim=embedding_dim)
    return test_url


def _admin_connection(admin_url: URL, name: str) -> psycopg.Connection[tuple[object, ...]]:
    # A session cannot drop the database it is connected to: fall back to the maintenance db.
    database = "postgres" if admin_url.database == name else admin_url.database
    return psycopg.connect(
        libpq_dsn(admin_url, database), autocommit=True, connect_timeout=CONNECT_TIMEOUT_SECONDS
    )


def drop_database(admin_url: URL, name: str) -> None:
    """DROP DATABASE IF EXISTS, only for disposable names."""
    assert_disposable_name(name)
    with _admin_connection(admin_url, name) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def alembic_config(url: URL, embedding_dim: int = TEST_EMBEDDING_DIM) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = url.render_as_string(hide_password=False)
    config.attributes["embedding_dim"] = embedding_dim
    return config


def run_alembic(fn: Callable[..., None], config: Config, *args: object) -> None:
    """Run an alembic command. env.py calls asyncio.run(): do it on a worker thread so it also
    works when the caller (e.g. a fixture resolved lazily) already runs inside an event loop."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(fn, config, *args).result()


def migrate(url: URL, revision: str = "head", embedding_dim: int = TEST_EMBEDDING_DIM) -> None:
    run_alembic(command.upgrade, alembic_config(url, embedding_dim), revision)


@asynccontextmanager
async def postgres_engines(url: URL) -> AsyncIterator[tuple[AsyncEngine, AsyncEngine]]:
    """(pooled engine, unpooled lock engine) for `url`, disposed on exit."""
    secret = SecretStr(url.render_as_string(hide_password=False))
    pooled = create_engine(
        secret, pool_size=3, max_overflow=2, pool_timeout_seconds=10, pool_recycle_seconds=1800
    )
    unpooled = create_unpooled_engine(secret)
    try:
        yield pooled, unpooled
    finally:
        await unpooled.dispose()
        await pooled.dispose()


async def truncate_all(factory: async_sessionmaker[AsyncSession]) -> None:
    async with factory.begin() as session:
        await session.execute(text("TRUNCATE documents CASCADE"))
