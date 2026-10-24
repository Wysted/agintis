"""Revert-ingest CLI: URI resolution, arguments and the dry-run/apply paths."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agintis.cli import revert_ingest as cli
from agintis.cli.local_case import source_uris
from agintis.cli.revert_ingest import EXIT_ABORTED, EXIT_OK, EXIT_USAGE, build_parser, main
from agintis.modules.knowledge.api import Document


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
