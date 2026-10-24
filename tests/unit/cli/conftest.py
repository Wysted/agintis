from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest

from agintis.bootstrap.settings import Settings
from agintis.cli import _ledger
from agintis.platform.usage import UsageRow


class FakeLedger:
    """Stands in for the usage repository: no CLI test may reach a real database."""

    def __init__(self) -> None:
        self.rows: list[UsageRow] = []

    async def add_all(self, rows: Sequence[UsageRow]) -> int:
        self.rows.extend(rows)
        return len(rows)


@pytest.fixture(autouse=True)
def ledger(monkeypatch: pytest.MonkeyPatch) -> FakeLedger:
    fake = FakeLedger()

    @asynccontextmanager
    async def repository(_settings: Settings) -> AsyncIterator[FakeLedger]:
        yield fake

    monkeypatch.setattr(_ledger, "usage_repository", repository)
    monkeypatch.setattr(_ledger, "get_settings", lambda: None)
    return fake
