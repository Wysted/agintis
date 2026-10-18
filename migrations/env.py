"""Alembic environment (async, psycopg3). Autogenerates from the shared MetaData."""

import asyncio
import hashlib
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

# Importing the tables registers them on the shared MetaData (add every module's tables here).
import agintis.modules.conversations.infrastructure.postgres.tables
import agintis.modules.knowledge.infrastructure.postgres.tables  # noqa: F401
from agintis.bootstrap.settings import EmbeddingsSettings, get_settings
from agintis.modules.conversations.infrastructure.postgres.tables import THREADS_ORDER_INDEX
from agintis.modules.knowledge.infrastructure.postgres.tables import (
    BM25_INDEX_PREFIX,
    METADATA_INDEX_PREFIX,
)
from agintis.platform.db import EXTERNAL_TABLES, metadata

# Indexes with a COLLATE / expression part: reflection loses it, so the drift check would always
# report them.
UNCOMPARABLE_INDEXES = frozenset({THREADS_ORDER_INDEX})
# Indexes that exist only where something else created them: the per-field metadata indexes
# (`cli.indexes`) and the per-tenant BM25 indexes (revision 0006 and ingest, with pg_textsearch).
UNMANAGED_INDEX_PREFIXES = (METADATA_INDEX_PREFIX, BM25_INDEX_PREFIX)

# Serializes concurrent `alembic upgrade head` runs (two hosts): held until the migration
# transaction ends. Distinct from the checkpointer SETUP_LOCK_KEY (different hash input).
MIGRATION_LOCK_KEY = int.from_bytes(hashlib.sha256(b"agintis.alembic.migrate").digest()[:8]) >> 1

config = context.config
target_metadata = metadata

# Logging is configured only when run from the `alembic` CLI: a programmatic caller (the app,
# tests) must keep its own logging setup.
if config.config_file_name is not None and config.cmd_opts is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Width of chunks.embedding, read by revision 0001. A programmatic caller may preset it.
config.attributes.setdefault("embedding_dim", EmbeddingsSettings().expected_dim)


def _database_url() -> str:
    preset = config.attributes.get("database_url")
    if preset:
        return str(preset)
    return get_settings().database_url.get_secret_value()


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Skip tables owned by other libraries (LangGraph checkpointer) and their indexes, the
    expression indexes Alembic cannot reflect faithfully (guarded by their own tests) and the
    indexes created outside Alembic's model (`UNMANAGED_INDEX_PREFIXES`)."""
    if (
        type_ == "index"
        and name is not None
        and (name in UNCOMPARABLE_INDEXES or name.startswith(UNMANAGED_INDEX_PREFIXES))
    ):
        return False
    if type_ == "table":
        return name not in EXTERNAL_TABLES
    table = getattr(obj, "table", None)
    return not (table is not None and table.name in EXTERNAL_TABLES)


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=False,  # pgvector `vector(N)` has no comparable declared width in the model
    )
    with context.begin_transaction():
        connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": MIGRATION_LOCK_KEY})
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_database_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_sync)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
