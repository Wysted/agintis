"""Alembic: up/down round trip, no model drift, unprivileged roles, the optional BM25 index and the
startup schema checks.

Uses its own disposable database (`agintis_test_migrations`), never the shared test databases.
"""

import hashlib
import logging
import secrets
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import psycopg
import pytest
from alembic import command
from pydantic import SecretStr, ValidationError
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.exc import ProgrammingError

from agintis.bootstrap.container import SCHEMA_HEAD_REVISION
from agintis.modules.knowledge.infrastructure.postgres import (
    TextSearchConfigError,
    verify_bm25_index,
)
from agintis.platform.db import DatabaseCheckError, check_database, create_engine
from tests.conftest import skip_or_fail
from tests.support.postgres import (
    admin_database_url,
    alembic_config,
    drop_database,
    libpq_dsn,
    postgres_engines,
    run_alembic,
)

pytestmark = pytest.mark.integration

DATABASE = "agintis_test_migrations"
DIM = 8
# Same value as migrations/env.py::MIGRATION_LOCK_KEY (env.py runs on import: not importable).
MIGRATION_LOCK_KEY = int.from_bytes(hashlib.sha256(b"agintis.alembic.migrate").digest()[:8]) >> 1
ROLE = "agintis_test_migrator"


@pytest.fixture
def admin() -> URL:
    try:
        url = admin_database_url()
    except ValidationError as exc:  # DATABASE_URL not configured
        skip_or_fail(f"Postgres not configured: {exc.__class__.__name__}")
    try:
        with psycopg.connect(libpq_dsn(url), connect_timeout=3):
            pass
    except psycopg.OperationalError as exc:
        skip_or_fail(f"Postgres not available: {exc.__class__.__name__}")
    return url


@pytest.fixture
def empty_database(admin: URL) -> Iterator[URL]:
    drop_database(admin, DATABASE)
    with psycopg.connect(libpq_dsn(admin), autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{DATABASE}"')
    try:
        yield admin.set(database=DATABASE)
    finally:
        drop_database(admin, DATABASE)


def _query(url: URL, sql: str) -> list[tuple[object, ...]]:
    with psycopg.connect(libpq_dsn(url)) as conn:
        return conn.execute(sql).fetchall()


def _tables(url: URL) -> set[str]:
    rows = _query(url, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    return {str(row[0]) for row in rows}


def test_upgrade_downgrade_round_trip_leaves_no_schema_behind(empty_database: URL) -> None:
    config = alembic_config(empty_database, embedding_dim=DIM)
    run_alembic(command.upgrade, config, "head")
    assert {"documents", "chunks", "alembic_version"} <= _tables(empty_database)
    (embedding,) = _query(
        empty_database,
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'",
    )[0]
    assert embedding == f"vector({DIM})"

    run_alembic(command.downgrade, config, "base")
    assert _tables(empty_database) == {"alembic_version"}
    assert _query(empty_database, "SELECT 1 FROM pg_ts_config WHERE cfgname = 'es_unaccent'") == []

    run_alembic(command.upgrade, config, "head")  # and up again
    assert {"documents", "chunks"} <= _tables(empty_database)


def _has_bm25_index(url: URL) -> bool:
    return bool(_query(url, "SELECT 1 FROM pg_indexes WHERE indexname = 'ix_chunks_text_bm25'"))


def _pg_textsearch_available(url: URL) -> bool:
    return bool(_query(url, "SELECT 1 FROM pg_available_extensions WHERE name = 'pg_textsearch'"))


def test_the_bm25_index_exists_only_where_pg_textsearch_is_available(
    empty_database: URL,
) -> None:
    """Revision 0005 is a no-op on servers without the extension (managed Postgres)."""
    config = alembic_config(empty_database, embedding_dim=DIM)
    run_alembic(command.upgrade, config, "head")
    assert _has_bm25_index(empty_database) == _pg_textsearch_available(empty_database)
    run_alembic(command.downgrade, config, "0004")
    assert not _has_bm25_index(empty_database)


async def test_revision_0005_creates_what_the_bm25_boot_check_needs(empty_database: URL) -> None:
    if not _pg_textsearch_available(empty_database):
        skip_or_fail("pg_textsearch not available: use the docker/postgres image")
    config = alembic_config(empty_database, embedding_dim=DIM)
    run_alembic(command.upgrade, config, "0004")
    async with postgres_engines(empty_database) as (engine, _):
        with pytest.raises(TextSearchConfigError, match="pg_textsearch"):
            await verify_bm25_index(engine)
        run_alembic(command.upgrade, config, "head")
        await verify_bm25_index(engine)
        run_alembic(command.downgrade, config, "0004")  # the extension stays, the index goes
        with pytest.raises(TextSearchConfigError, match="ix_chunks_text_bm25"):
            await verify_bm25_index(engine)


def test_upgrade_waits_for_a_concurrent_migrator(empty_database: URL) -> None:
    """The advisory lock serializes two `upgrade head` runs: the second waits, then is a no-op."""
    config = alembic_config(empty_database, embedding_dim=DIM)
    with psycopg.connect(libpq_dsn(empty_database)) as holder:
        holder.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_KEY,))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_alembic, command.upgrade, config, "head")
            with pytest.raises(FuturesTimeout):
                future.result(timeout=2)
            holder.rollback()  # releases the lock
            future.result(timeout=60)
    assert {"documents", "chunks"} <= _tables(empty_database)


def test_models_and_migrations_do_not_drift(empty_database: URL) -> None:
    config = alembic_config(empty_database, embedding_dim=DIM)
    run_alembic(command.upgrade, config, "head")
    run_alembic(command.check, config)  # raises when autogenerate would emit operations


class TestUnprivilegedMigrator:
    """A role that is neither superuser nor extension owner must still be able to migrate."""

    @pytest.fixture
    def role_url(self, admin: URL, empty_database: URL) -> Iterator[URL]:
        password = secrets.token_hex(8)
        with psycopg.connect(libpq_dsn(admin), autocommit=True) as conn:
            conn.execute(f'DROP ROLE IF EXISTS "{ROLE}"')
            conn.execute(f"CREATE ROLE \"{ROLE}\" LOGIN PASSWORD '{password}'")
            conn.execute(f'ALTER DATABASE "{DATABASE}" OWNER TO "{ROLE}"')
        try:
            yield empty_database.set(username=ROLE, password=password)
        finally:
            drop_database(admin, DATABASE)
            with psycopg.connect(libpq_dsn(admin), autocommit=True) as conn:
                conn.execute(f'DROP ROLE IF EXISTS "{ROLE}"')

    def test_existing_extensions_are_left_alone(
        self, admin: URL, role_url: URL, caplog: pytest.LogCaptureFixture
    ) -> None:
        with psycopg.connect(libpq_dsn(admin, DATABASE), autocommit=True) as conn:
            conn.execute("CREATE EXTENSION vector")
            conn.execute("CREATE EXTENSION unaccent")
        with caplog.at_level(logging.WARNING, logger="alembic"):
            run_alembic(command.upgrade, alembic_config(role_url, embedding_dim=DIM), "head")
        assert {"documents", "chunks"} <= _tables(role_url)
        # pg_textsearch needs a superuser: the role skips it instead of failing, and says so
        # (a server NOTICE would be dropped by psycopg).
        assert not _has_bm25_index(role_url)
        if _pg_textsearch_available(admin):
            assert f"role {ROLE} cannot CREATE EXTENSION pg_textsearch" in caplog.text

    def test_missing_extension_fails_clearly_instead_of_with_a_privilege_error(
        self, admin: URL, role_url: URL
    ) -> None:
        (trusted,) = _query(
            admin.set(database=DATABASE),
            "SELECT bool_and(trusted) FROM pg_available_extension_versions "
            "WHERE name IN ('vector', 'unaccent')",
        )[0]
        config = alembic_config(role_url, embedding_dim=DIM)
        if trusted:  # trusted extensions can be created by the database owner
            run_alembic(command.upgrade, config, "head")
            return
        with pytest.raises(ProgrammingError) as info:
            run_alembic(command.upgrade, config, "head")
        # The guard turned the privilege error into a NOTICE; what fails is the first use of the
        # missing type/config, which says what is missing.
        assert not isinstance(info.value.orig, psycopg.errors.InsufficientPrivilege)


class TestStartupSchemaCheck:
    async def test_check_against_a_real_migrated_database(self, empty_database: URL) -> None:
        run_alembic(command.upgrade, alembic_config(empty_database, embedding_dim=DIM), "head")
        engine = create_engine(
            SecretStr(empty_database.render_as_string(hide_password=False)),
            pool_size=1,
            max_overflow=0,
            pool_timeout_seconds=5,
            pool_recycle_seconds=60,
        )
        try:
            ok = await check_database(
                engine,
                min_vector_version=(0, 8, 4),
                expected_embedding_dim=DIM,
                expected_revision=SCHEMA_HEAD_REVISION,
            )
            assert ok >= "0.8.4"
            with pytest.raises(DatabaseCheckError, match=r"vector\(8\).*vector\(1024\)"):
                await check_database(
                    engine, min_vector_version=(0, 8, 4), expected_embedding_dim=1024
                )
            with pytest.raises(DatabaseCheckError, match="expects 9999"):
                await check_database(engine, min_vector_version=(0, 8, 4), expected_revision="9999")
            async with engine.begin() as conn:
                await conn.execute(text("DROP TABLE alembic_version"))
            with pytest.raises(DatabaseCheckError, match="not migrated"):
                await check_database(
                    engine, min_vector_version=(0, 8, 4), expected_revision=SCHEMA_HEAD_REVISION
                )
        finally:
            await engine.dispose()

    async def test_an_empty_database_asks_for_the_migration(self, empty_database: URL) -> None:
        engine = create_engine(
            SecretStr(empty_database.render_as_string(hide_password=False)),
            pool_size=1,
            max_overflow=0,
            pool_timeout_seconds=5,
            pool_recycle_seconds=60,
        )
        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            with pytest.raises(DatabaseCheckError, match="alembic upgrade head"):
                await check_database(
                    engine,
                    min_vector_version=(0, 8, 4),
                    expected_embedding_dim=DIM,
                    expected_revision=SCHEMA_HEAD_REVISION,
                )
        finally:
            await engine.dispose()
