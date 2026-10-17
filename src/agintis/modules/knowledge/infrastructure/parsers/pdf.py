"""PDF with a text layer via pypdf (BSD-3). One section per page; no OCR.

Encrypted PDFs whose user password is empty (owner-restricted: no printing/copying) are
decrypted transparently (AES needs `pypdf[crypto]`); any other encrypted PDF is rejected.

Scanned-page policy: a page with fewer than `MIN_CHARS_PER_PAGE` extractable characters is
"textless". If MORE than `MAX_TEXTLESS_PAGE_RATIO` of the pages are textless the document is
treated as scanned (`ScannedDocumentError`); otherwise the textless pages are dropped and
logged (their page numbers stay out of the index, numbering of the others is preserved).
"""

import logging
from io import BytesIO
from typing import Final

from pypdf import PdfReader

from agintis.modules.knowledge.application import (
    DocumentTooLargeError,
    ParsedDocument,
    ParsedSection,
    ScannedDocumentError,
    UnparseableDocumentError,
)
from agintis.modules.knowledge.infrastructure.parsers._title import title_from_uri
from agintis.modules.knowledge.infrastructure.parsers.limits import ParseContext
from agintis.shared_kernel.errors import DomainError

MIN_CHARS_PER_PAGE: Final = 8
MAX_TEXTLESS_PAGE_RATIO: Final = 0.5
LOGGED_PAGES_LIMIT: Final = 20

logger = logging.getLogger(__name__)


def _read(content: bytes, uri: str, max_pages: int) -> tuple[list[tuple[int, str]], str]:
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted and not reader.decrypt(""):
        raise UnparseableDocumentError(f"{uri!r} is password protected")
    page_count = len(reader.pages)
    if page_count == 0:
        raise UnparseableDocumentError(f"{uri!r} has no pages")
    if page_count > max_pages:
        raise DocumentTooLargeError(f"{uri!r} has {page_count} pages (limit {max_pages})")
    pages = [
        (number, (page.extract_text() or "").strip())
        for number, page in enumerate(reader.pages, start=1)
    ]
    metadata_title = reader.metadata.title if reader.metadata else None
    return pages, (metadata_title or "").strip()


def parse_pdf(content: bytes, uri: str, context: ParseContext) -> ParsedDocument:
    try:
        pages, metadata_title = _read(content, uri, context.limits.max_pages)
    except DomainError:
        raise
    except Exception as exc:  # pypdf raises PyPdfError, ValueError, KeyError, RecursionError...
        raise UnparseableDocumentError(f"Cannot read PDF {uri!r}: {exc}") from exc

    textless = [number for number, text in pages if len(text) < MIN_CHARS_PER_PAGE]
    if len(textless) / len(pages) > MAX_TEXTLESS_PAGE_RATIO or len(textless) == len(pages):
        raise ScannedDocumentError(
            f"{uri!r} has no usable text layer in {len(textless)} of {len(pages)} pages "
            "(scanned PDF?). The 'simple' parser does not do OCR: ingest it with a profile "
            "that uses the 'ocr_index_card' parser (`ocr` extra)."
        )
    if textless:
        logger.warning(
            "Dropping %d textless page(s) of %r (pages %s%s)",
            len(textless),
            uri,
            textless[:LOGGED_PAGES_LIMIT],
            "..." if len(textless) > LOGGED_PAGES_LIMIT else "",
        )
    dropped = set(textless)
    sections = tuple(
        ParsedSection(text=text, page=number) for number, text in pages if number not in dropped
    )
    return ParsedDocument(title=metadata_title or title_from_uri(uri), sections=sections)
