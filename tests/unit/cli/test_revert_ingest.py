"""Revert-ingest CLI: URI resolution, arguments and the dry-run/apply paths."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agintis.bootstrap.container import Container
from agintis.cli import revert_ingest as cli
from agintis.cli.local_case import source_uris
from agintis.cli.revert_ingest import EXIT_ABORTED, EXIT_OK, EXIT_USAGE, build_parser, main
from agintis.modules.knowledge.api import Document
from agintis.shared_kernel.ids import TenantId
from tests.support.settings import make_settings
from tests.unit.knowledge.test_routing_and_batch import _rerouting_batch, _source


def _case(tmp_path: Path) -> tuple[Path, Path]:
    """Index url `index-url` differs from the manifest url `manifest-url` of the same id."""
    samples = tmp_path / "samples"
    (samples / "fam").mkdir(parents=True)
    (samples / "fam" / "a.pdf").write_bytes(b"x")
    index = tmp_path / "index" / "documents.jsonl"
    index.parent.mkdir()
    index.write_text(
        json.dumps({"id": "a", "url": "index-url"}) + "\n" + json.dumps({"id": "b", "url": "u"}),
        encoding="utf-8",
    )
    entry = {
        "id": "a",
        "family": "fam",
        "path": "/x/a.pdf",
        "url": "manifest-url",
        "content_type": "application/pdf",
    }
    (samples / "_manifest.json").write_text(json.dumps([entry]), encoding="utf-8")
    return index, samples


def _doc(uri: str) -> Document:
    status = SimpleNamespace(value="ready")
    return cast(Document, SimpleNamespace(profile="p", source_uri=uri, status=status))


def test_uris_come_from_the_manifest_like_ingest_and_unfetched_rows_are_skipped(
    tmp_path: Path,
) -> None:
    index, samples = _case(tmp_path)
    assert source_uris(index, samples) == (["manifest-url"], 1)


def test_tenant_is_required_and_apply_is_opt_in() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--profiles", "p", "--index", "i"])
    args = build_parser().parse_args(["--profiles", "p", "--index", "i", "--tenant", "t"])
    assert args.apply is False


def test_an_index_without_local_files_is_a_usage_error(tmp_path: Path) -> None:
    index, _ = _case(tmp_path)
    (tmp_path / "samples" / "_manifest.json").unlink()
    argv = ["--profiles", str(tmp_path), "--index", str(index), "--tenant", "t"]
    assert main(argv) == EXIT_USAGE


def _run_with(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    async def fake_run(args: Any, uris: Any, on_removed: Any) -> list[Document]:
        calls.append((args.apply, list(uris)))
        return cast(list[Document], outcome(args, on_removed))

    monkeypatch.setattr(cli, "run", fake_run)
    return calls


def test_a_dry_run_uses_manifest_uris_and_warns_when_nothing_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index, _ = _case(tmp_path)
    calls = _run_with(monkeypatch, lambda args, cb: [])
    argv = ["--profiles", "p", "--index", str(index), "--tenant", "typo"]
    assert main(argv) == EXIT_OK
    assert calls == [(False, ["manifest-url"])]
    assert "no document of tenant 'typo' matches" in capsys.readouterr().out


def test_an_abort_midway_reports_what_was_already_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index, _ = _case(tmp_path)

    def outcome(args: Any, on_removed: Any) -> list[Document]:
        on_removed(_doc("manifest-url"))
        raise RuntimeError("concurrent ingest")

    _run_with(monkeypatch, outcome)
    argv = ["--profiles", "p", "--index", str(index), "--tenant", "t", "--apply"]
    assert main(argv) == EXIT_ABORTED
    out = capsys.readouterr()
    assert "deleted ready" in out.out
    assert "deleted 1 documents before the abort" in out.out


class _FakeContainer:
    def __init__(self, batch: Any) -> None:
        self.ingest_batch = batch

    async def __aenter__(self) -> "_FakeContainer":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _seeded_container(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The REAL `run` over a real IngestBatch and in-memory fakes (Container.build faked)."""
    batch, documents, _ = _rerouting_batch()
    tenant = TenantId("t")
    asyncio.run(
        batch.execute(tenant, [_source("manifest-url", to="old"), _source("second", to="old")])
    )
    monkeypatch.setattr(
        Container,
        "build",
        lambda *a, **k: _FakeContainer(batch),
    )
    monkeypatch.setattr(cli, "get_settings", make_settings)
    return batch, documents


def _argv(index: Path) -> list[str]:
    return ["--profiles", "p", "--index", str(index), "--tenant", "t", "--apply"]


def _two_entry_case(tmp_path: Path) -> Path:
    index, samples = _case(tmp_path)
    (samples / "fam" / "b.pdf").write_bytes(b"x")
    manifest = json.loads((samples / "_manifest.json").read_text(encoding="utf-8"))
    manifest.append({**manifest[0], "id": "b", "path": "/x/b.pdf", "url": "second"})
    (samples / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return index


def test_the_real_run_reports_every_deleted_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index = _two_entry_case(tmp_path)
    _, documents = _seeded_container(monkeypatch)
    code = main(_argv(index))
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert out.count("deleted ready") == 2
    assert "deleted 2 documents" in out
    assert not documents.items


def test_the_real_run_reports_the_first_deletion_when_it_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index = _two_entry_case(tmp_path)
    batch, documents = _seeded_container(monkeypatch)
    real = batch._ingest.remove_rerouted
    calls = 0

    async def flaky(tenant_id: TenantId, document: Document) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("concurrent ingest")
        return cast(bool, await real(tenant_id, document))

    monkeypatch.setattr(batch._ingest, "remove_rerouted", flaky)
    code = main(_argv(index))
    out = capsys.readouterr().out
    assert code == EXIT_ABORTED
    assert out.count("deleted ready") == 1
    assert "deleted 1 documents before the abort" in out
    assert len(documents.items) == 1
