"""Parser "ocr_index_card": `simple_index_card` with OCR of scanned PDFs before the card.

Chain: the `simple` parser (text layer) -> on `ScannedDocumentError`, OCR of every page
(pypdfium2 renders it at `OCR_DPI`, RapidOCR reads it; one section per page with one line per
text box, like the text path, so page cleaners such as `strip_repeated_page_lines` keep working)
-> when OCR finds no text, the index card. `collected["content_mode"]` is `full_text` | `ocr` |
`index_card`.

Needs the `ocr` uv extra. Its packages are imported lazily; bootstrap calls `require_ocr_extra`
when a profile uses this parser, so a missing extra fails the boot, not an ingest. OCR is
CPU-bound (seconds per page): it runs in a worker thread, with one engine per parser created
on first use.
"""

import asyncio
import threading
from collections.abc import Callable, Sequence
from importlib.util import find_spec
from typing import Any, Final

from agintis.modules.knowledge.application import (
    ParsedDocument,
    ParsedSection,
    SourceFile,
    UnparseableDocumentError,
)
from agintis.modules.knowledge.infrastructure.parsers.index_card import (
    CONTENT_MODE_KEY,
    IndexCardParser,
)
from agintis.modules.knowledge.infrastructure.parsers.limits import ParseLimits
from agintis.modules.knowledge.infrastructure.parsers.simple import SimpleParser, _sanitize

OCR_INDEX_CARD_PARSER_KEY: Final = "ocr_index_card"
OCR_EXTRA: Final = "ocr"
# Engine and model behind the OCR text (rapidocr is pinned exactly in the `ocr` extra and its
# wheel ships these default models): part of the parser version, so changing them re-indexes.
OCR_ENGINE: Final = "rapidocr-3.10.0/PP-OCRv6-small"
OCR_DPI: Final = 300
_PDF_POINTS_PER_INCH: Final = 72
_OCR_MODULES: Final = ("rapidocr", "onnxruntime", "pypdfium2")

type PageOcr = Callable[[bytes], Sequence[str]]  # PDF bytes -> recognized text of each page
type OcrEngine = Callable[[Any], Any]  # RGB image array -> result with `txts`


class OcrNotInstalledError(RuntimeError):
    """A profile uses the OCR parser but the `ocr` extra is not installed."""


def require_ocr_extra() -> None:
    """Fail fast (call from bootstrap) when the OCR packages are missing; imports nothing."""
    missing = [name for name in _OCR_MODULES if find_spec(name) is None]
    if missing:
        raise OcrNotInstalledError(
            f"Parser '{OCR_INDEX_CARD_PARSER_KEY}' needs the '{OCR_EXTRA}' extra: install it "
            f'(uv sync --extra {OCR_EXTRA}; Docker: --build-arg UV_EXTRAS="... {OCR_EXTRA}"). '
            f"Missing: {', '.join(missing)}"
        )


def _rapidocr() -> OcrEngine:
    from rapidocr import RapidOCR

    engine: OcrEngine = RapidOCR()
    return engine


class RapidOcrPages:
    """`PageOcr` on pypdfium2 + RapidOCR. Blocking: call it from a worker thread."""

    def __init__(self, engine_factory: Callable[[], OcrEngine] = _rapidocr) -> None:
        self._engine_factory = engine_factory
        self._engine: OcrEngine | None = None
        # One document at a time: pdfium is not thread-safe and the engine already uses every
        # core; the same lock guards the engine's lazy creation.
        self._lock = threading.Lock()

    def __call__(self, content: bytes) -> list[str]:
        import numpy as np
        import pypdfium2 as pdfium

        with self._lock:
            if self._engine is None:
                self._engine = self._engine_factory()
            engine = self._engine
            try:
                pdf = pdfium.PdfDocument(content)
            except Exception as exc:  # pdfium raises PdfiumError on a broken file
                raise UnparseableDocumentError(f"Cannot render PDF for OCR: {exc}") from exc
            try:
                pages = []
                for page in pdf:
                    image = page.render(scale=OCR_DPI / _PDF_POINTS_PER_INCH).to_pil()
                    pages.append("\n".join(engine(np.array(image)).txts or ()))
                return pages
            finally:
                pdf.close()


class OcrIndexCardParser(IndexCardParser):
    version = f"{SimpleParser.version}+{OCR_ENGINE}"

    def __init__(self, limits: ParseLimits | None = None, ocr: PageOcr | None = None) -> None:
        super().__init__(limits)
        self._ocr: PageOcr = ocr if ocr is not None else RapidOcrPages()

    async def _scanned(self, source: SourceFile) -> ParsedDocument | None:
        # Only PDFs raise ScannedDocumentError, after the size and page limits were checked.
        # shortcut: a cancelled ingest does not stop the worker thread (it OCRs up to
        # PARSER_MAX_PAGES pages, ~4 s each); add a per-page cancel check if long scans hurt.
        pages = await asyncio.to_thread(self._ocr, source.content)
        title = source.document_metadata.get("title")
        parsed = _sanitize(
            ParsedDocument(
                title=title if isinstance(title, str) else "",
                sections=tuple(
                    ParsedSection(text=text, page=number)
                    for number, text in enumerate(pages, start=1)
                ),
                collected={CONTENT_MODE_KEY: ("ocr",)},
            ),
            source.uri,
        )
        return parsed if parsed.sections else await super()._scanned(source)
