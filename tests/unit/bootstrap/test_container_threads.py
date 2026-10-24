"""Container wiring of threads and the checkpointer lifecycle."""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from agintis.bootstrap.adapters import AgentsThreadTranscript
from agintis.bootstrap.container import Container, StartupError
from agintis.modules.agents_catalog.api import RunAgent, ThreadsUnavailableError
from agintis.modules.conversations.application import NoThreadState
from agintis.modules.conversations.infrastructure.postgres import PgThreadRepository
from agintis.platform.checkpointer import Checkpointer, CheckpointerSetupError
from agintis.shared_kernel.ids import TenantId, ThreadId
from tests.support.settings import make_settings
from tests.unit.bootstrap.test_container import _tokenizer_file


@pytest.fixture(autouse=True)
def _offline_boot_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The boot checks that need a real database / LLM are not what these tests exercise."""

    async def no_check(engine: object, config: str) -> None:
        return None

    monkeypatch.setattr("agintis.bootstrap.container.verify_text_search_config", no_check)
    monkeypatch.setattr("agintis.bootstrap.container.ChatModels.verify", lambda self: None)


def test_threads_are_wired_over_postgres_and_the_agent_can_hold_a_conversation() -> None:
    container = Container.build(make_settings(checkpointer={"enabled": True}))

    assert isinstance(container.checkpointer, Checkpointer)
    assert container.agents_catalog.registry.get("direct-chat").supports_threads is True
    assert isinstance(container.threads.create._threads, PgThreadRepository)
    assert isinstance(container.threads.get._transcript, AgentsThreadTranscript)


async def test_disabled_checkpointer_means_no_pool_and_no_thread_runs() -> None:
    container = Container.build(make_settings())  # the test helper disables it

    assert container.checkpointer is None
    assert container.agents_catalog.registry.get("direct-chat").supports_threads is False
    assert isinstance(container.threads.delete._state, NoThreadState)
    with pytest.raises(ThreadsUnavailableError):
        async with container.agents_catalog.run_agent.start_thread(
            TenantId("t"), ThreadId("x"), message="hi"
        ):
            pass
    await container.aclose()


async def test_pool_opens_at_start_and_closes_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def opened(self: Checkpointer, **kwargs: Any) -> None:
        calls.append("open")

    async def closed(self: Checkpointer) -> None:
        calls.append("close")

    monkeypatch.setattr(Checkpointer, "open", opened)
    monkeypatch.setattr(Checkpointer, "aclose", closed)

    async with Container.build(make_settings(checkpointer={"enabled": True})):
        assert calls == ["open"]  # opened even though the boot checks are off

    assert calls == ["open", "close"]


async def test_disabled_checkpointer_is_never_opened_or_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(self: Checkpointer, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not be called")

    monkeypatch.setattr(Checkpointer, "open", boom)
    monkeypatch.setattr(Checkpointer, "verify", boom)

    async with Container.build(make_settings(startup_checks=False)):
        pass


def _boot_settings(tmp_path: Path, **overrides: Any) -> Any:
    return make_settings(
        startup_checks=True,
        embeddings={"tokenizer": str(_tokenizer_file(tmp_path)), "url": "http://127.0.0.1:9"},
        **overrides,
    )


async def test_boot_check_failure_is_a_startup_error_and_closes_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def database_ok(engine: object, **kwargs: Any) -> str:
        return "0.8.7"

    async def opened(self: Checkpointer, **kwargs: Any) -> None:
        calls.append("open")

    async def missing(self: Checkpointer, timeout_seconds: float = 0) -> None:
        raise CheckpointerSetupError("The checkpoint tables do not exist: run setup")

    async def closed(self: Checkpointer) -> None:
        calls.append("close")

    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_ok)
    monkeypatch.setattr(Checkpointer, "open", opened)
    monkeypatch.setattr(Checkpointer, "verify", missing)
    monkeypatch.setattr(Checkpointer, "aclose", closed)

    with pytest.raises(StartupError, match="checkpoint tables do not exist"):
        await Container.build(_boot_settings(tmp_path, checkpointer={"enabled": True})).__aenter__()

    assert calls == ["open", "close"]


async def test_boot_check_runs_with_the_configured_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []

    async def database_ok(engine: object, **kwargs: Any) -> str:
        return "0.8.7"

    async def noop(self: Checkpointer, **kwargs: Any) -> None:
        return None

    async def verified(self: Checkpointer, timeout_seconds: float = 0) -> None:
        seen.append(timeout_seconds)

    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_ok)
    monkeypatch.setattr(Checkpointer, "open", noop)
    monkeypatch.setattr(Checkpointer, "aclose", noop)
    monkeypatch.setattr(Checkpointer, "verify", verified)

    async with Container.build(
        _boot_settings(
            tmp_path, checkpointer={"enabled": True}, database={"startup_check_timeout_seconds": 7}
        )
    ):
        pass

    assert seen == [7.0]


async def test_in_flight_releases_are_drained_before_the_database_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def drained(self: RunAgent) -> None:
        order.append("run_agent")

    async def disposed(self: AsyncEngine) -> None:
        order.append("engine")

    monkeypatch.setattr(RunAgent, "aclose", drained)
    monkeypatch.setattr(AsyncEngine, "dispose", disposed)

    async with Container.build(make_settings()):
        pass

    assert order[0] == "run_agent"
    assert "engine" in order[1:]
