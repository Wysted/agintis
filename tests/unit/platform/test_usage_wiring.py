"""The real ledger wiring: Purpose vs the migration's CHECK, and the CLI's engine and write path."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agintis.agent_kit.model_usage import ModelUsage
from agintis.cli import _ledger
from agintis.platform.usage import Purpose
from tests.support.settings import make_settings

MIGRATION = Path(__file__).parents[3] / "migrations/versions/20261024_0007_llm_usage.py"


def test_the_migration_check_allows_exactly_the_purposes_of_the_code() -> None:
    source = MIGRATION.read_text(encoding="utf-8")
    match = re.search(r"purpose IN \(([^)]*)\)", source)
    assert match is not None
    allowed = set(re.findall(r"'([^']+)'", match.group(1)))

    assert allowed == {purpose.value for purpose in Purpose}


class _Engine:
    disposed = False

    async def dispose(self) -> None:
        self.disposed = True


async def test_the_ledger_repository_uses_a_one_connection_writable_engine_and_disposes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine()
    seen: dict[str, Any] = {}

    def fake_create_engine(url: Any, **kwargs: Any) -> _Engine:
        seen.update(kwargs)
        return engine

    monkeypatch.setattr(_ledger, "create_engine", fake_create_engine)
    monkeypatch.setattr(_ledger, "create_session_factory", lambda e: e)

    async with _ledger.usage_repository(make_settings()):
        assert not engine.disposed

    assert (seen["pool_size"], seen["max_overflow"]) == (1, 0)
    assert "read_only" not in seen  # the ledger must be writable
    assert engine.disposed


def test_record_usage_writes_one_row_per_model_through_the_real_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[Any] = []

    class Repo:
        def __init__(self, factory: Any) -> None:
            pass

        async def add_all(self, rows: Any) -> int:
            written.extend(rows)
            return len(rows)

    monkeypatch.setattr(_ledger, "create_engine", lambda *_a, **_k: _Engine())
    monkeypatch.setattr(_ledger, "create_session_factory", lambda e: e)
    monkeypatch.setattr(_ledger, "PgUsageRepository", Repo)
    monkeypatch.setattr(_ledger, "get_settings", make_settings)

    before = datetime.now(UTC)
    _ledger.record_usage(
        {"a": ModelUsage(1, 2, 3, 0, 1), "b": ModelUsage(4, 5, 9, 1, 2)}, Purpose.EVAL, "eval q"
    )

    assert sorted(r.model for r in written) == ["a", "b"]
    assert {r.purpose for r in written} == {Purpose.EVAL}
    assert len({r.run_id for r in written}) == 1
    assert all(r.created_at >= before and r.source == "eval q" for r in written)
