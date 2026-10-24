"""The usage ledger on Postgres: a repeated write is a no-op, the window read is half-open."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agintis.platform.db import create_session_factory
from agintis.platform.usage import Purpose, UsageRow
from agintis.platform.usage.repository import PgUsageRepository
from tests.support.postgres import postgres_engines

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 24, 12, tzinfo=UTC)
ROW = UsageRow("run-1", "gpt-x", Purpose.CHAT, NOW, 100, 20, 40, 2, tenant_id="acme")


@pytest.fixture
async def factory(pg_test_url: URL) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    async with postgres_engines(pg_test_url) as (engine, _lock_engine):
        session_factory = create_session_factory(engine)
        async with session_factory.begin() as session:
            await session.execute(text("TRUNCATE llm_usage"))
        yield session_factory


async def test_a_repeated_write_keeps_the_first_row(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    repo = PgUsageRepository(factory)

    assert await repo.add_all([ROW, replace(ROW, model="gpt-y")]) == 2
    assert await repo.add_all([replace(ROW, input_tokens=999), ROW]) == 0

    assert await repo.between() == [ROW, replace(ROW, model="gpt-y")]


async def test_the_window_includes_since_and_excludes_until(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    repo = PgUsageRepository(factory)
    rows = [replace(ROW, run_id=f"r{n}", created_at=NOW + timedelta(days=n)) for n in range(3)]
    await repo.add_all(rows)

    assert await repo.between(since=rows[1].created_at, until=rows[2].created_at) == [rows[1]]
