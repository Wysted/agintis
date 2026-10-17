"""Parser "ocr_index_card": `simple_index_card` with OCR of scanned PDFs before the card.

Chain: the `simple` parser (text layer) -> on `ScannedDocumentError`, OCR of every page
(pypdfium2 renders it at `OCR_DPI`, RapidOCR reads it; one section per page with one line per
text box, like the text path, so page cleaners such as `strip_repeated_page_lines` keep working)
-> when OCR finds no text, the index card. With OCR text, the index card (when the metadata has
a title) still leads as a page-less first section: its curated title and description read
better than OCR of the same header. `collected["content_mode"]` is `full_text` | `ocr` |
`index_card`.

Needs the `ocr` uv extra. Its packages are imported lazily; bootstrap calls `require_ocr_extra`
on the ingest path when a profile uses this parser, so a missing extra fails the ingest boot,
not the first scanned PDF (the app never parses: it boots without the extra). OCR is
CPU-bound (seconds per page): it runs in a worker thread, with one engine per parser created
on first use. Any rendering or OCR failure (except MemoryError) is an `UnparseableDocumentError`:
it fails that document, not the batch.
"""

import asyncio
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from importlib.util import find_spec
from typing import Any, Final

from agintis.modules.knowledge.application import (
    DocumentTooLargeError,
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
# Pixel cap per rendered page: 4x an A4 page at 300 dpi (2480 x 3508 px, ~35 Mpx, ~100 MB RGB).
# Up to A2 renders at full OCR_DPI; a bigger MediaBox (a tiny PDF can declare one) is scaled
# down instead of allocating a multi-GB bitmap.
MAX_OCR_PIXELS: Final = 4 * 2480 * 3508
_PDF_POINTS_PER_INCH: Final = 72
_OCR_MODULES: Final = ("rapidocr", "onnxruntime", "pypdfium2")
# pdfium is not thread-safe process-wide: one document at a time across every parser instance
# (the engine already uses every core, so OCR runs under it too).
_PDFIUM_LOCK: Final = threading.Lock()

# (PDF bytes, max pages) -> recognized text of each page; over max pages: DocumentTooLargeError
type PageOcr = Callable[[bytes, int], Sequence[str]]
type OcrEngine = Callable[[Any], Any]  # PIL RGB image -> result with `txts`


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

    def __call__(self, content: bytes, max_pages: int) -> list[str]:
        import pypdfium2 as pdfium

        with _PDFIUM_LOCK:  # also guards the engine's lazy creation
            if self._engine is None:
                self._engine = self._engine_factory()
            engine = self._engine
            try:
                pdf = pdfium.PdfDocument(content)
                try:
                    # pypdf validated the page count, but pdfium may count pages differently.
                    if len(pdf) > max_pages:
                        raise DocumentTooLargeError(
                            f"PDF has {len(pdf)} pages for OCR (limit {max_pages})"
                        )
                    return [_read_page(engine, page) for page in pdf]
                finally:
                    pdf.close()
            except (MemoryError, DocumentTooLargeError):
                raise
            except Exception as exc:  # PdfiumError (broken file/page), onnxruntime errors...
                raise UnparseableDocumentError(f"Cannot OCR PDF: {exc}") from exc


def _read_page(engine: OcrEngine, page: Any) -> str:
    width, height = page.get_size()  # points
    scale = min(OCR_DPI / _PDF_POINTS_PER_INCH, math.sqrt(MAX_OCR_PIXELS / (width * height)))
    # RapidOCR converts a PIL image from RGB to its BGR input (a bare ndarray is taken as BGR).
    # shortcut: one line per text box, so a spaced header comes out one word per line; merge
    # boxes on the same visual line (`result.boxes` y-overlap) when chunking/search needs it.
    return "\n".join(engine(page.render(scale=scale).to_pil()).txts or ())


class OcrIndexCardParser(IndexCardParser):
    # shortcut: a mixed PDF (mostly text, a few scanned pages) never reaches OCR: `parse_pdf`
    # drops its textless pages (logged); OCR those pages when a corpus has such documents.
    # `+card`: the index card leads the OCR text (changing that layout re-indexes too).
    version = f"{SimpleParser.version}+{OCR_ENGINE}+card"

    def __init__(self, limits: ParseLimits | None = None, ocr: PageOcr | None = None) -> None:
        super().__init__(limits)
        self._ocr: PageOcr = ocr if ocr is not None else RapidOcrPages()

    async def _scanned(self, source: SourceFile) -> ParsedDocument | None:
        # Only PDFs raise ScannedDocumentError, after the size and page limits were checked.
        # shortcut: a cancelled ingest does not stop the worker thread (it OCRs up to
        # PARSER_MAX_PAGES pages, ~4 s each); add a per-page cancel check if long scans hurt.
        pages = await asyncio.to_thread(self._ocr, source.content, self._limits.max_pages)
        card = self._card(source)
        parsed = _sanitize(
            ParsedDocument(
                title=card.title if card else "",
                sections=tuple(
                    ParsedSection(text=text, page=number)
                    for number, text in enumerate(pages, start=1)
                ),
                collected={CONTENT_MODE_KEY: ("ocr",)},
            ),
            source.uri,
        )
        if not parsed.sections:
            return card
        return replace(parsed, sections=(*card.sections, *parsed.sections)) if card else parsed
