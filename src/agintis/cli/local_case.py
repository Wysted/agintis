"""A case's local files: index rows (document metadata) paired with documents already on disk.

A case (`case/<name>/`) keeps a crawler index (`index/documents.jsonl`, one JSON row per
document) and, optionally, downloaded files described by `samples/_manifest.json`
(`id`, `url`, `family`, `path`, `content_type`). This module only READS local files (no
network): a row becomes a `SourceFile` whose `document_metadata` is the whole index row, which
is what profile routing and `document:<field>` metadata sources read.

Everything is lazy and forgiving about DATA problems: a malformed index line is reported as an
`IndexEntry` with an `error` (the caller decides what to do with it) and a manifest entry that is
malformed or points outside the samples directory is skipped with a warning.
"""

import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agintis.modules.knowledge.api import SourceFile

logger = logging.getLogger(__name__)

MANIFEST_NAME = "_manifest.json"


@dataclass(frozen=True)
class LocalFile:
    id: str
    url: str
    path: Path
    content_type: str


@dataclass(frozen=True)
class IndexEntry:
    """One non-blank line of the index: the decoded row, or why it could not be read."""

    line: int
    row: dict[str, Any] | None = None
    error: str | None = None


def iter_index_entries(path: Path) -> Iterator[IndexEntry]:
    """Lines of a JSON Lines index (blank lines skipped), one `IndexEntry` each, lazily."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:  # JSONDecodeError and friends
                yield IndexEntry(number, error=f"invalid JSON: {exc}")
                continue
            if isinstance(row, dict):
                yield IndexEntry(number, row=row)
            else:
                yield IndexEntry(number, error=f"expected a JSON object, got {type(row).__name__}")


def iter_index(path: Path) -> Iterator[dict[str, Any]]:
    """Valid rows of a JSON Lines index; unreadable lines are logged and skipped."""
    for entry in iter_index_entries(path):
        if entry.row is not None:
            yield entry.row
        else:
            logger.warning("%s line %d skipped: %s", path.name, entry.line, entry.error)


def load_manifest(samples_dir: Path) -> dict[str, LocalFile]:
    """Downloaded files by index id.

    An entry is left out (with a warning) when it is malformed, when its file is missing or when
    it would resolve OUTSIDE `samples_dir` (a `family` such as `../..` must not make the ingest
    read arbitrary files).
    """
    manifest = samples_dir / MANIFEST_NAME
    if not manifest.is_file():
        return {}
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    root = samples_dir.resolve()
    files: dict[str, LocalFile] = {}
    for entry in entries:
        try:
            entry_id, url = str(entry["id"]), str(entry["url"])
            content_type = str(entry["content_type"])
            # The recorded absolute path belongs to the machine that downloaded the file:
            # resolve the file from its family folder under THIS samples directory instead.
            local = samples_dir / str(entry["family"]) / Path(str(entry["path"])).name
        except (KeyError, TypeError):
            logger.warning("Ignoring a malformed manifest entry: %r", entry)
            continue
        if not local.resolve().is_relative_to(root):
            logger.warning("Sample %s resolves outside %s: ignored (%s)", entry_id, root, local)
            continue
        if not local.is_file():
            logger.warning("Sample %s listed in the manifest is missing: %s", entry_id, local)
            continue
        files[entry_id] = LocalFile(entry_id, url, local, content_type)
    return files


def source_uris(index: Path, samples: Path) -> tuple[list[str], int]:
    """The `source_uri` ingest stored for each index row, plus the rows it never ingested.

    Same resolution as `source_for`: the URI is the MANIFEST url of the row's id, and a row
    without a usable local file was never ingested (counted, not listed).
    """
    files = load_manifest(samples)
    uris: list[str] = []
    skipped = 0
    for row in iter_index(index):
        local = files.get(str(row.get("id")))
        if local is None:
            skipped += 1
        else:
            uris.append(local.url)
    return uris, skipped


def source_for(row: Mapping[str, Any], files: Mapping[str, LocalFile]) -> SourceFile | None:
    """The row's local document as a `SourceFile`, or None when it was not downloaded.

    Raises `InvalidMetadataError` (the row does not fit the document metadata rules) or
    `OSError` (the file cannot be read): problems of THIS document, not of the run.
    """
    local = files.get(str(row.get("id")))
    if local is None:
        return None
    return SourceFile(
        uri=local.url,
        mime_type=local.content_type,
        content=local.path.read_bytes(),
        document_metadata=row,
    )
