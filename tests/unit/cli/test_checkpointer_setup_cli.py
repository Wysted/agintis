"""checkpointer_setup CLI: exit codes, and the other CLIs never need the checkpointer."""

import argparse
from pathlib import Path
from typing import Any

import pytest

from agintis.bootstrap.container import Container
from agintis.cli import checkpointer_setup, ingest
from agintis.cli import eval as eval_cli
from agintis.platform.checkpointer import Checkpointer
from tests.support.settings import make_settings


def test_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    async def ready() -> bool:
        return True

    async def disabled() -> bool:
        return False

    async def down() -> bool:
        raise OSError("db down")

    monkeypatch.setattr(checkpointer_setup, "run", ready)
    assert checkpointer_setup.main([]) == checkpointer_setup.EXIT_OK
    monkeypatch.setattr(checkpointer_setup, "run", disabled)
    assert checkpointer_setup.main([]) == checkpointer_setup.EXIT_DISABLED
    monkeypatch.setattr(checkpointer_setup, "run", down)
    assert checkpointer_setup.main([]) == checkpointer_setup.EXIT_ABORTED
    assert "aborted: OSError" in capsys.readouterr().err


async def test_setup_opens_waits_creates_verifies_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def opened(self: Checkpointer, **kwargs: Any) -> None:
        calls.append(f"open wait={kwargs.get('wait')}")

    async def setup(self: Checkpointer) -> None:
        calls.append("setup")

    async def verify(self: Checkpointer) -> None:
        calls.append("verify")

    async def closed(self: Checkpointer) -> None:
        calls.append("close")

    monkeypatch.setattr(
        checkpointer_setup, "get_settings", lambda: make_settings(checkpointer={"enabled": True})
    )
    monkeypatch.setattr(Checkpointer, "open", opened)
    monkeypatch.setattr(Checkpointer, "setup", setup)
    monkeypatch.setattr(Checkpointer, "verify", verify)
    monkeypatch.setattr(Checkpointer, "aclose", closed)

    assert await checkpointer_setup.run() is True
    assert calls == ["open wait=True", "setup", "verify", "close"]


async def test_setup_closes_the_checkpointer_when_setup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def opened(self: Checkpointer, **kwargs: Any) -> None:
        calls.append("open")

    async def setup(self: Checkpointer) -> None:
        raise OSError("no privileges")

    async def closed(self: Checkpointer) -> None:
        calls.append("close")

    monkeypatch.setattr(
        checkpointer_setup, "get_settings", lambda: make_settings(checkpointer={"enabled": True})
    )
    monkeypatch.setattr(Checkpointer, "open", opened)
    monkeypatch.setattr(Checkpointer, "setup", setup)
    monkeypatch.setattr(Checkpointer, "aclose", closed)

    with pytest.raises(OSError, match="no privileges"):
        await checkpointer_setup.run()
    assert calls == ["open", "close"]


async def test_setup_reports_a_disabled_checkpointer_without_building_a_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    def spy(*args: Any, **kwargs: Any) -> None:
        built.append("pool")

    monkeypatch.setattr(checkpointer_setup, "get_settings", make_settings)
    monkeypatch.setattr(checkpointer_setup, "build_checkpointer", spy)

    assert await checkpointer_setup.run() is False
    assert built == []


class _Entered(Exception):
    pass


async def test_ingest_and_eval_run_with_the_checkpointer_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: list[Any] = []

    def spy(settings: Any, **kwargs: Any) -> Container:
        built.append(settings)
        raise _Entered

    enabled = make_settings(checkpointer={"enabled": True})
    monkeypatch.setattr(Container, "build", spy)
    monkeypatch.setattr(ingest, "get_settings", lambda: enabled)
    monkeypatch.setattr(eval_cli, "get_settings", lambda: enabled)
    monkeypatch.setattr(ingest, "local_items", lambda *args: [])
    monkeypatch.setattr(eval_cli, "load_questions", lambda path: [])

    with pytest.raises(_Entered):
        await ingest.run(
            argparse.Namespace(
                profiles=tmp_path,
                offline=False,
                samples=tmp_path,
                index=tmp_path / "i" / "j",
                limit=None,
                tenant="t",
                cleanup_rerouted=False,
            ),
            lambda outcome: None,
        )
    with pytest.raises(_Entered):
        await eval_cli.run(
            argparse.Namespace(
                profiles=tmp_path,
                questions=tmp_path,
                mode="dense",
                rerank=False,
                tenant="t",
                k=[1],
                report=None,
                extract_filters=False,
                agents=None,
                analyze=False,
                route=False,
            )
        )

    assert [s.checkpointer.enabled for s in built] == [False, False]
    assert enabled.checkpointer.enabled is True  # the caller's settings are not mutated


async def test_setup_ignores_a_broken_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Missing profiles/agents dirs or a bad LLM profiles file must not block the tables."""
    calls: list[str] = []

    async def noop(self: Checkpointer, *args: Any, **kwargs: Any) -> None:
        calls.append("call")

    def no_container(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the Container must not be built")

    broken = make_settings(
        checkpointer={"enabled": True},
        agents={"dir": tmp_path / "nope"},
        knowledge={"profiles_dir": tmp_path / "nope"},
        llm={"profiles_file": tmp_path / "nope.yaml"},
    )
    monkeypatch.setattr(checkpointer_setup, "get_settings", lambda: broken)
    monkeypatch.setattr(Container, "build", no_container)
    for name in ("open", "setup", "verify", "aclose"):
        monkeypatch.setattr(Checkpointer, name, noop)

    assert await checkpointer_setup.run() is True
    assert len(calls) == 4
