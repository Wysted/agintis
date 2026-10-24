"""Postgres repository of the usage ledger: idempotent inserts and a time-window read."""

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agintis.platform.db import UnitOfWork
from agintis.platform.usage.rows import Purpose, UsageRow
from agintis.platform.usage.tables import llm_usage


class PgUsageRepository:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def add_all(self, rows: Sequence[UsageRow]) -> int:
        """Insert the rows; one whose (run_id, model) exists is left as it is (a retried write
        never double-counts). Returns how many were new."""
        if not rows:
            return 0
        statement = insert(llm_usage).on_conflict_do_nothing().returning(llm_usage.c.run_id)
        async with UnitOfWork(self._factory) as uow:
            result = await uow.session.execute(
                statement, [asdict(row) | {"purpose": row.purpose.value} for row in rows]
            )
            return len(result.all())  # RETURNING lists only the inserted rows

    async def between(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> list[UsageRow]:
        """Rows with `since <= created_at < until` (a missing bound is open), oldest first."""
        query = select(llm_usage).order_by(llm_usage.c.created_at, llm_usage.c.run_id)
        if since is not None:
            query = query.where(llm_usage.c.created_at >= since)
        if until is not None:
            query = query.where(llm_usage.c.created_at < until)
        async with UnitOfWork(self._factory) as uow:
            result = await uow.session.execute(query)
            return [UsageRow(**{**row._mapping, "purpose": Purpose(row.purpose)}) for row in result]
