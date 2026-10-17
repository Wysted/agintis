"""Lightweight parser (key "simple"): PDF with text layer, DOCX, Markdown, plain text, HTML.

Pure-Python/CPU-bound work runs in a worker thread so the event loop stays responsive.
Every result goes through one sanitizing step (NUL bytes and lone surrogates removed) so
nothing unstorable reaches the chunker or the database.
"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import ClassVar

from agintis.modules.knowledge.application import (
    DocumentTooLargeError,
    ParsedDocument,
    Parser,
    SourceFile,
    UnsupportedDocumentTypeError,
)
from agintis.modules.knowledge.infrastructure.parsers._title import title_from_uri
from agintis.modules.knowledge.infrastructure.parsers.docx import parse_docx
from agintis.modules.knowledge.infrastructure.parsers.html import parse_html
from agintis.modules.knowledge.infrastructure.parsers.limits import ParseContext, ParseLimits
from agintis.modules.knowledge.infrastructure.parsers.markdown import parse_markdown, parse_text
from agintis.modules.knowledge.infrastructure.parsers.pdf import parse_pdf
from agintis.modules.knowledge.infrastructure.text import (
    mime_charset,
    strip_unstorable,
)

type ParseFn = Callable[[bytes, str, ParseContext], ParsedDocument]

SIMPLE_PARSER_KEY = "simple"

_BY_MIME: Mapping[str, ParseFn] = MappingProxyType(
    {
        "application/pdf": parse_pdf,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": parse_docx,
        "text/markdown": parse_markdown,
        "text/x-markdown": parse_markdown,
        "text/plain": parse_text,
        "text/html": parse_html,
        "application/xhtml+xml": parse_html,
    }
)
# Used when the caller sends a generic MIME type (application/octet-stream, empty).
_BY_EXTENSION: Mapping[str, ParseFn] = MappingProxyType(
    {
        ".pdf": parse_pdf,
        ".docx": parse_docx,
        ".md": parse_markdown,
        ".txt": parse_text,
        ".html": parse_html,
        ".htm": parse_html,
    }
)


def _resolve(source: SourceFile) -> ParseFn:
    mime = source.mime_type.split(";", 1)[0].strip().lower()
    if mime in _BY_MIME:
        return _BY_MIME[mime]
    path = source.uri.split("?", 1)[0].lower()
    for extension, fn in _BY_EXTENSION.items():
        if path.endswith(extension):
            return fn
    raise UnsupportedDocumentTypeError(
        f"The 'simple' parser does not support {source.mime_type!r} ({source.uri!r}); "
        "supported: PDF with text layer, DOCX, Markdown, plain text, HTML"
    )


def _sanitize(parsed: ParsedDocument, uri: str) -> ParsedDocument:
    """The single place where unstorable characters are removed from parsed text."""
    sections = tuple(
        replace(
            section,
            text=text,
            headings=tuple(h for h in map(strip_unstorable, section.headings) if h.strip()),
        )
        for section in parsed.sections
        if (text := strip_unstorable(section.text).strip())
    )
    title = strip_unstorable(parsed.title).strip() or title_from_uri(uri)
    source_html = strip_unstorable(parsed.source_html) if parsed.source_html is not None else None
    return replace(parsed, title=title, sections=sections, source_html=source_html)


class SimpleParser(Parser):
    # 2: HTML extraction on lxml (implied end tags, block boundaries, meta charset sniffing).
    # 3: Markdown headings drop a trailing attribute block (`{ #id }`).
    version: ClassVar[int | str] = 3  # str: subclasses may name an engine (see Parser.version)

    def __init__(self, limits: ParseLimits | None = None) -> None:
        self._limits = limits if limits is not None else ParseLimits()

    async def parse(self, source: SourceFile) -> ParsedDocument:
        fn = _resolve(source)
        if len(source.content) > self._limits.max_bytes:
            raise DocumentTooLargeError(
                f"{source.uri!r} is {len(source.content)} bytes (limit {self._limits.max_bytes})"
            )
        context = ParseContext(charset=mime_charset(source.mime_type), limits=self._limits)
        parsed = await asyncio.to_thread(fn, source.content, source.uri, context)
        return _sanitize(parsed, source.uri)
