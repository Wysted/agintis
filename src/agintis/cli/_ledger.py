"""Access to the LLM usage ledger from the CLIs (a small own engine, separate from the run's)."""

import asyncio
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

from agintis.agent_kit.model_usage import ModelUsage
from agintis.bootstrap.settings import Settings, get_settings
from agintis.cli._common import git_revision
from agintis.platform.db import create_engine, create_session_factory
from agintis.platform.usage import Purpose, UsageRow
from agintis.platform.usage.repository import PgUsageRepository


@asynccontextmanager
async def usage_repository(settings: Settings) -> AsyncIterator[PgUsageRepository]:
    """The ledger repository on a one-connection engine.

    The measurement CLIs open their main database read-only (a bug cannot write), so the ledger
    gets its own writable engine that only this repository uses.
    """
    db = settings.database
    engine = create_engine(
        settings.database_url,
        pool_size=1,
        max_overflow=0,
        pool_timeout_seconds=db.pool_timeout_seconds,
        pool_recycle_seconds=db.pool_recycle_seconds,
        connect_timeout_seconds=db.connect_timeout_seconds,
        statement_timeout_ms=db.statement_timeout_ms,
    )
    try:
        yield PgUsageRepository(create_session_factory(engine))
    finally:
        await engine.dispose()


def record_usage(
    usage: Mapping[str, ModelUsage] | None,
    purpose: Purpose,
    source: str,
    *,
    tenant_id: str | None = None,
    agent_id: str | None = None,
) -> None:
    """Ledger rows for one CLI invocation: one per model, under a fresh run id.

    Called once at the end of the invocation, aborted ones included (their calls were billed).
    Best effort: a ledger failure is a warning on stderr, never a lost run.
    """
    if not usage:
        return
    run_id = uuid4().hex
    now = datetime.now(UTC)
    revision = git_revision()
    rows = [
        UsageRow(
            run_id=run_id,
            model=model,
            purpose=purpose,
            created_at=now,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cached_input_tokens=u.cached_input_tokens,
            calls=u.calls,
            tenant_id=tenant_id,
            agent_id=agent_id,
            source=source,
            git_revision=revision,
        )
        for model, u in usage.items()
    ]

    async def write() -> None:
        async with usage_repository(get_settings()) as repository:
            await repository.add_all(rows)

    try:
        asyncio.run(write())
    except Exception as exc:
        print(
            f"warning: could not record the LLM usage in the ledger ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
