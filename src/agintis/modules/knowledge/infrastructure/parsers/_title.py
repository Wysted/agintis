"""Shared helpers for the lightweight parsers."""

import re
from urllib.parse import parse_qsl, unquote, urlparse

_STEM = re.compile(r"\.[^./]+$")
# Only document-like extensions: a bare `x.cl` (a host in a redirect parameter) is not a file.
_FILE_NAME = re.compile(
    r"[^/\\]+\.(?:pdf|docx?|xlsx?|pptx?|odt|rtf|txt|md|csv|html?)", re.IGNORECASE
)


def _stem(segment: str) -> str:
    return _STEM.sub("", segment)


def _query_file_stem(query: str) -> str | None:
    """Stem of the first query value that looks like a file path (`?namefile=REF-X/N0372.pdf`)."""
    for _, value in parse_qsl(query, keep_blank_values=False):
        name = value.rstrip("/").replace("\\", "/").rsplit("/", 1)[-1]
        if _FILE_NAME.fullmatch(name):
            return _stem(name) or None
    return None


def title_from_uri(uri: str) -> str:
    """A readable name for a source: the last path segment without extension when it names a
    document; otherwise the file-like query parameter when there is one (generic scripts such as
    `GetFile.php?namefile=.../NP0000372.pdf`), else that segment; falls back to the whole URI."""
    parsed = urlparse(uri)
    path = unquote(parsed.path) or uri
    name = path.rstrip("/").rsplit("/", 1)[-1]
    # A path that already names a document wins over any query parameter.
    if not _FILE_NAME.fullmatch(name):
        from_query = _query_file_stem(parsed.query)
        if from_query is not None:
            return from_query
    return _stem(name) or uri
