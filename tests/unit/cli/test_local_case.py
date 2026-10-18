"""Local case files for the batch ingest CLI: index rows paired with downloaded samples."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

from agintis.bootstrap.container import StartupError
from agintis.cli import ingest as cli
from agintis.cli.ingest import build_parser, format_outcome, local_items, main
from agintis.cli.local_case import iter_index, iter_index_entries, load_manifest, source_for
from agintis.modules.knowledge.api import DEFAULT_PROFILE, BatchOutcome, SourceFile
from tests.support.settings import FakeSettings

ROWS = [
    {"id": "doc-a", "doc_type": "circular", "number": "1", "section": ["S1", "S2"], "title": "A"},
    {"id": "doc-b", "doc_type": "ncg", "number": "2", "title": "B"},
    {"id": "doc-c", "doc_type": "circular", "number": "3", "title": "C"},
]


def _case(tmp_path: Path) -> tuple[Path, Path]:
    index = tmp_path / "index" / "documents.jsonl"
    index.parent.mkdir()
    index.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n\n", encoding="utf-8")
    samples = tmp_path / "samples"
    (samples / "circulares").mkdir(parents=True)
    (samples / "circulares" / "doc-a.pdf").write_bytes(b"%PDF A")
    (samples / "circulares" / "doc-c.pdf").write_bytes(b"%PDF C")
    manifest = [
        {
            "id": "doc-a",
            "family": "circulares",
            # absolute path of the machine that downloaded it: only the file name is used
            "path": "/elsewhere/circulares/doc-a.pdf",
            "url": "https://example.test/a.pdf",
            "content_type": "application/pdf",
        },
        {
            "id": "doc-c",
            "family": "circulares",
            "path": "/elsewhere/circulares/doc-c.pdf",
            "url": "https://example.test/c.pdf",
            "content_type": "application/pdf",
        },
        {
            "id": "doc-b",
            "family": "ncg",
            "path": "/elsewhere/ncg/doc-b.pdf",  # never downloaded here
            "url": "https://example.test/b.pdf",
            "content_type": "application/pdf",
        },
    ]
    (samples / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return index, samples


def test_index_rows_are_read_skipping_blank_lines(tmp_path: Path) -> None:
    index, _ = _case(tmp_path)
    assert [row["id"] for row in iter_index(index)] == ["doc-a", "doc-b", "doc-c"]


def test_the_manifest_lists_only_files_that_exist_here(tmp_path: Path) -> None:
    _, samples = _case(tmp_path)
    files = load_manifest(samples)
    assert set(files) == {"doc-a", "doc-c"}
    assert files["doc-a"].path == samples / "circulares" / "doc-a.pdf"
    assert files["doc-a"].url == "https://example.test/a.pdf"


def test_no_manifest_means_no_files(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) == {}


def test_a_row_becomes_a_source_with_the_whole_row_as_document_metadata(tmp_path: Path) -> None:
    index, samples = _case(tmp_path)
    files = load_manifest(samples)
    rows = {row["id"]: row for row in iter_index(index)}
    source = source_for(rows["doc-a"], files)
    assert source is not None
    assert (source.uri, source.mime_type, source.content) == (
        "https://example.test/a.pdf",
        "application/pdf",
        b"%PDF A",
    )
    assert source.document_metadata["section"] == ("S1", "S2")
    assert source.document_metadata["number"] == "1"
    assert source_for(rows["doc-b"], files) is None  # listed but not on disk


def test_local_items_keep_index_order_and_honour_the_limit(tmp_path: Path) -> None:
    index, samples = _case(tmp_path)
    assert [s.uri for s in local_items(index, samples, None) if isinstance(s, SourceFile)] == [
        "https://example.test/a.pdf",
        "https://example.test/c.pdf",
    ]
    assert [s.uri for s in local_items(index, samples, 1) if isinstance(s, SourceFile)] == [
        "https://example.test/a.pdf"
    ]


def test_items_are_produced_lazily(tmp_path: Path) -> None:
    index, samples = _case(tmp_path)
    items = local_items(index, samples, None)
    first = next(items)
    assert isinstance(first, SourceFile)
    (samples / "circulares" / "doc-c.pdf").unlink()  # not read yet: gone by the time it is asked
    (late,) = list(items)
    assert isinstance(late, BatchOutcome)
    assert (late.status, late.uri) == ("failed", "https://example.test/c.pdf")
    assert "FileNotFoundError" in str(late.reason)


class TestBadData:
    def _index(self, tmp_path: Path, *lines: str) -> tuple[Path, Path]:
        index, samples = _case(tmp_path)
        index.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return index, samples

    def test_entries_report_a_malformed_line_instead_of_raising(self, tmp_path: Path) -> None:
        index, _ = self._index(tmp_path, '{"id": "doc-a"}', "{broken", "[1, 2]", "", '{"id": "x"}')
        entries = list(iter_index_entries(index))
        assert [(e.line, e.row is not None) for e in entries] == [
            (1, True),
            (2, False),
            (3, False),
            (5, True),
        ]
        assert "invalid JSON" in str(entries[1].error)
        assert "JSON object" in str(entries[2].error)
        assert [r["id"] for r in iter_index(index)] == ["doc-a", "x"]

    def test_a_bad_line_becomes_a_failed_outcome_and_the_run_goes_on(self, tmp_path: Path) -> None:
        index, samples = self._index(tmp_path, json.dumps(ROWS[0]), "{broken", json.dumps(ROWS[2]))
        items = list(local_items(index, samples, None))
        assert [type(i).__name__ for i in items] == ["SourceFile", "BatchOutcome", "SourceFile"]
        failed = items[1]
        assert isinstance(failed, BatchOutcome)
        assert (failed.status, failed.uri) == ("failed", "documents.jsonl:2")
        assert "invalid JSON" in str(failed.reason)

    def test_a_row_with_invalid_metadata_is_a_failed_outcome(self, tmp_path: Path) -> None:
        bad = {"id": "doc-a", "title": "bad\u0000title", "url": "https://example.test/a.pdf"}
        index, samples = self._index(tmp_path, json.dumps(bad), json.dumps(ROWS[2]))
        first, second = list(local_items(index, samples, None))
        assert isinstance(first, BatchOutcome)
        assert first.status == "failed"
        assert first.uri == "https://example.test/a.pdf"
        assert "InvalidMetadataError" in str(first.reason)
        assert isinstance(second, SourceFile)

    def test_a_sample_that_is_not_a_regular_file_is_simply_not_local(self, tmp_path: Path) -> None:
        index, samples = _case(tmp_path)
        broken = samples / "circulares" / "doc-a.pdf"
        broken.unlink()
        broken.mkdir()  # a directory where the file should be
        items = list(local_items(index, samples, None))
        assert [i.status for i in items if isinstance(i, BatchOutcome)] == []  # not is_file: gone
        assert [i.uri for i in items if isinstance(i, SourceFile)] == ["https://example.test/c.pdf"]

    def test_failed_rows_do_not_count_towards_the_limit(self, tmp_path: Path) -> None:
        index, samples = self._index(tmp_path, "{broken", json.dumps(ROWS[0]), json.dumps(ROWS[2]))
        items = list(local_items(index, samples, 1))
        assert [type(i).__name__ for i in items] == ["BatchOutcome", "SourceFile"]


class TestManifestContainment:
    def _manifest(self, tmp_path: Path, **entry: str) -> dict[str, object]:
        samples = tmp_path / "samples"
        (samples / "ok").mkdir(parents=True)
        (samples / "ok" / "good.pdf").write_bytes(b"x")
        (tmp_path / "secret.txt").write_text("secret")
        base = {"id": "e", "url": "u", "family": "ok", "path": "good.pdf", "content_type": "t"}
        (samples / "_manifest.json").write_text(json.dumps([{**base, **entry}]), encoding="utf-8")
        return dict(load_manifest(samples))

    def test_an_entry_inside_the_samples_directory_is_kept(self, tmp_path: Path) -> None:
        assert set(self._manifest(tmp_path)) == {"e"}

    def test_a_family_escaping_the_samples_directory_is_ignored(self, tmp_path: Path) -> None:
        assert self._manifest(tmp_path, family="..", path="secret.txt") == {}

    def test_an_absolute_family_is_ignored(self, tmp_path: Path) -> None:
        assert self._manifest(tmp_path, family=str(tmp_path), path="secret.txt") == {}

    def test_a_symlink_pointing_out_is_ignored(self, tmp_path: Path) -> None:
        samples = tmp_path / "samples"
        (samples / "ok").mkdir(parents=True)
        (tmp_path / "secret.txt").write_text("secret")
        (samples / "ok" / "link.pdf").symlink_to(tmp_path / "secret.txt")
        entry = {"id": "e", "url": "u", "family": "ok", "path": "link.pdf", "content_type": "t"}
        (samples / "_manifest.json").write_text(json.dumps([entry]), encoding="utf-8")
        assert load_manifest(samples) == {}

    def test_a_malformed_entry_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        samples = tmp_path / "samples"
        (samples / "ok").mkdir(parents=True)
        (samples / "ok" / "good.pdf").write_bytes(b"x")
        good = {"id": "e", "url": "u", "family": "ok", "path": "good.pdf", "content_type": "t"}
        (samples / "_manifest.json").write_text(json.dumps([{"id": "no-url"}, 7, good]), "utf-8")
        assert set(load_manifest(samples)) == {"e"}


class TestMain:
    def _run_with(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outcomes: list[BatchOutcome],
        error: Exception | None = None,
    ) -> None:
        async def fake_run(args: object, emit: Callable[[BatchOutcome], None]) -> None:
            for outcome in outcomes:
                emit(outcome)
            if error is not None:
                raise error

        monkeypatch.setattr(cli, "run", fake_run)

    ARGS: ClassVar[list[str]] = ["--profiles", "p", "--index", "i"]

    def test_everything_handled_exits_zero_and_prints_as_it_goes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run_with(
            monkeypatch,
            [BatchOutcome("u1", "indexed", "p", 2), BatchOutcome("u2", "unrouted", reason="x")],
        )
        assert main(self.ARGS) == 0
        out = capsys.readouterr().out
        assert out.index("u1") < out.index("u2") < out.index("summary:")
        assert "indexed=1, unrouted=1" in out

    def test_refreshed_titles_are_counted_next_to_the_statuses(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run_with(
            monkeypatch,
            [
                BatchOutcome("u1", "skipped", "p", 2, title_refreshed=True),
                BatchOutcome("u2", "skipped", "p", 2),
            ],
        )
        assert main(self.ARGS) == 0
        out = capsys.readouterr().out
        assert "skipped=2, titles_refreshed=1" in out
        assert "title refreshed" in out.splitlines()[0]

    def test_a_failed_document_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run_with(
            monkeypatch,
            [BatchOutcome("u1", "failed", reason="bad"), BatchOutcome("u2", "indexed", "p", 1)],
        )
        assert main(self.ARGS) == 1
        assert "failed=1, indexed=1" in capsys.readouterr().out

    def test_an_abort_exits_three_with_a_partial_summary_and_no_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._run_with(
            monkeypatch,
            [BatchOutcome("u1", "indexed", "p", 1)],
            error=StartupError("database unreachable"),
        )
        assert main(self.ARGS) == 3
        captured = capsys.readouterr()
        assert "aborted: StartupError: database unreachable" in captured.err
        assert "u1" in captured.out
        assert "summary (partial): indexed=1" in captured.out
        assert "Traceback" not in captured.err

    def test_noisy_http_loggers_are_quietened(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._run_with(monkeypatch, [])
        main(self.ARGS)
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING


def test_arguments() -> None:
    args = build_parser().parse_args(["--profiles", "p", "--index", "i.jsonl", "--limit", "3"])
    assert (args.profiles, args.index, args.limit, args.tenant) == (
        Path("p"),
        Path("i.jsonl"),
        3,
        "default",
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--index", "i.jsonl"])


def test_a_non_positive_limit_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--profiles", "p", "--index", "i", "--limit", "0"]) == 2
    assert "--limit" in capsys.readouterr().err


def test_a_rerouted_source_is_flagged_in_the_output() -> None:
    flagged = format_outcome(BatchOutcome("u", "indexed", "new", 3, rerouted_from=("old",)))
    assert "also under old" in flagged
    assert "--cleanup-rerouted" in flagged
    cleaned = format_outcome(
        BatchOutcome("u", "indexed", "new", 3, rerouted_from=("old",), removed_rerouted=True)
    )
    assert "removed" in cleaned
    assert "--cleanup-rerouted" not in cleaned


def test_the_cleanup_flag_is_off_by_default() -> None:
    base = ["--profiles", "p", "--index", "i"]
    assert build_parser().parse_args(base).cleanup_rerouted is False
    assert build_parser().parse_args([*base, "--cleanup-rerouted"]).cleanup_rerouted is True


def test_outcomes_are_printed_one_per_line() -> None:
    ok = format_outcome(BatchOutcome("u1", "indexed", "legal", 12))
    assert "indexed" in ok
    assert "legal" in ok
    assert "12 chunks" in ok
    skipped = format_outcome(BatchOutcome("u2", "unrouted", reason="no profile"))
    assert "no profile" in skipped
    assert skipped.split()[1] == "-"  # no profile


def test_prune_is_opt_in_and_incompatible_with_limit(capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--profiles", "p", "--index", "i"]
    assert build_parser().parse_args(base).prune is False
    assert build_parser().parse_args([*base, "--prune"]).prune is True
    assert main([*base, "--prune", "--limit", "3"]) == 2
    assert "--prune" in capsys.readouterr().err


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeProfiles:
    def all(self) -> list[_Named]:
        return [_Named(DEFAULT_PROFILE.name), _Named("guide"), _Named("ref")]


def _fake_container(
    prune_result: list[object] | None,
    calls: list[tuple[object, ...]],
    events: list[tuple[str, object]] | None = None,
) -> type:
    seen = [] if events is None else events

    class FakeChunks:
        async def repair_bm25_index(self, tenant: object) -> None:
            seen.append(("repair_bm25_index", tenant))

    class FakeBatch:
        async def stream(self, tenant: object, items: object, **kw: object):  # type: ignore[no-untyped-def]
            for item in items:  # type: ignore[attr-defined]
                seen.append(("item", item))
                yield item if isinstance(item, BatchOutcome) else BatchOutcome(item.uri, "indexed")

        async def prune(self, *args: object) -> list[object] | None:
            calls.append(args)
            return prune_result

    class FakeContainer:
        ingest_batch = FakeBatch()
        profiles = _FakeProfiles()
        chunks = FakeChunks()

        @classmethod
        def build(cls, settings: object, *, for_ingest: bool) -> "FakeContainer":
            assert for_ingest  # the ingest path checks the parsers' extras at boot
            return cls()

        async def __aenter__(self) -> "FakeContainer":
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    return FakeContainer


def _prune_argv(index: Path, samples: Path) -> list[str]:
    return ["--profiles", "p", "--index", str(index), "--samples", str(samples), "--prune"]


def test_the_tenants_bm25_index_is_repaired_once_before_the_first_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged documents never reach add_many: the run itself builds a missing index."""
    index, samples = _case(tmp_path)
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(cli, "get_settings", FakeSettings)
    monkeypatch.setattr(cli, "Container", _fake_container(None, [], events))
    assert main(["--profiles", "p", "--index", str(index), "--samples", str(samples)]) == 0
    assert events[0] == ("repair_bm25_index", "default")
    assert [kind for kind, _ in events].count("repair_bm25_index") == 1
    assert len(events) > 1


def test_prune_is_skipped_when_an_index_row_has_no_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index, samples = _case(tmp_path)  # doc-b was never downloaded
    pruned: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "get_settings", FakeSettings)
    monkeypatch.setattr(cli, "Container", _fake_container([], pruned))
    assert main(_prune_argv(index, samples)) == 0
    assert pruned == []
    assert "prune skipped" in capsys.readouterr().out


def test_prune_gets_the_loaded_profiles_without_the_default_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index, samples = _case(tmp_path)
    (samples / "ncg").mkdir()
    (samples / "ncg" / "doc-b.pdf").write_bytes(b"%PDF B")  # every row now has its file
    pruned: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "get_settings", FakeSettings)
    monkeypatch.setattr(cli, "Container", _fake_container([object(), object()], pruned))
    assert main(_prune_argv(index, samples)) == 0
    ((tenant, names, outcomes),) = pruned
    assert (str(tenant), names) == ("default", ["guide", "ref"])
    assert [o.uri for o in outcomes] == [  # type: ignore[attr-defined]
        "https://example.test/a.pdf",
        "https://example.test/b.pdf",
        "https://example.test/c.pdf",
    ]
    assert "pruned=2" in capsys.readouterr().out


def test_an_unreadable_index_line_skips_prune_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index, samples = _case(tmp_path)
    (samples / "ncg").mkdir()
    (samples / "ncg" / "doc-b.pdf").write_bytes(b"%PDF B")
    index.write_text(index.read_text("utf-8") + "{not json\n", encoding="utf-8")

    pruned: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli, "get_settings", FakeSettings)
    monkeypatch.setattr(cli, "Container", _fake_container(None, pruned))
    assert main(_prune_argv(index, samples)) == 1
    out = capsys.readouterr().out
    assert "unreadable index line" in out
    assert "prune skipped" in out
