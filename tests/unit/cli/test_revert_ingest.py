"""Revert-ingest CLI: argument handling and index reading (the use case is tested with fakes)."""

import json
from pathlib import Path

import pytest

from agintis.cli.revert_ingest import EXIT_USAGE, build_parser, index_uris, main


def test_index_uris_read_the_url_of_each_row(tmp_path: Path) -> None:
    index = tmp_path / "documents.jsonl"
    index.write_text(
        "\n".join([json.dumps({"id": "1", "url": "u1"}), "not json", json.dumps({"id": "2"})]),
        encoding="utf-8",
    )
    assert index_uris(index) == ["u1"]


def test_tenant_is_required_and_apply_is_opt_in() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--profiles", "p", "--index", "i"])
    args = build_parser().parse_args(["--profiles", "p", "--index", "i", "--tenant", "t"])
    assert args.apply is False


def test_an_index_without_urls_is_a_usage_error(tmp_path: Path) -> None:
    index = tmp_path / "documents.jsonl"
    index.write_text(json.dumps({"id": "1"}) + "\n", encoding="utf-8")
    argv = ["--profiles", str(tmp_path), "--index", str(index), "--tenant", "t"]
    assert main(argv) == EXIT_USAGE
