"""Parser "simple_index_card": `simple`, falling back to an index card for scanned PDFs.

A scanned document has no text for the lightweight parser (`ocr_index_card` adds OCR). Rather
than failing the ingest, this parser indexes the document's CARD: the `title` and `description` the
caller supplied in `SourceFile.document_metadata` (a crawler index row), so the document can
still be found and cited. It reports which path it took as `collected["content_mode"]`
(`full_text` | `index_card`), readable as a `parsed:content_mode` metadata source.

A scanned document without a `title` in its document metadata still raises
`ScannedDocumentError`. Inherits `SimpleParser.version`: a change of the real parser re-indexes.
"""

from typing import Final

from agintis.modules.knowledge.application import (
    ParsedDocument,
    ParsedSection,
    ScannedDocumentError,
    SourceFile,
)
from agintis.modules.knowledge.infrastructure.parsers.simple import SimpleParser
from agintis.modules.knowledge.infrastructure.text import strip_unstorable

INDEX_CARD_PARSER_KEY: Final = "simple_index_card"
CONTENT_MODE_KEY: Final = "content_mode"


class IndexCardParser(SimpleParser):
    async def parse(self, source: SourceFile) -> ParsedDocument:
        try:
            parsed = await super().parse(source)
        except ScannedDocumentError:
            fallback = await self._scanned(source)
            if fallback is None:
                raise
            return fallback
        return ParsedDocument(
            title=parsed.title,
            sections=parsed.sections,
            source_html=parsed.source_html,
            collected={**parsed.collected, CONTENT_MODE_KEY: ("full_text",)},
        )

    async def _scanned(self, source: SourceFile) -> ParsedDocument | None:
        """What a scanned document becomes (`None` = re-raise `ScannedDocumentError`)."""
        return self._card(source)

    @staticmethod
    def _card(source: SourceFile) -> ParsedDocument | None:
        title = source.document_metadata.get("title")
        if not isinstance(title, str) or not title.strip():
            return None
        description = source.document_metadata.get("description")
        parts = [title.strip()]
        if isinstance(description, str) and description.strip():
            parts.append(description.strip())
        text = strip_unstorable("\n\n".join(parts))
        return ParsedDocument(
            title=strip_unstorable(title.strip()),
            sections=(ParsedSection(text=text),),
            collected={CONTENT_MODE_KEY: ("index_card",)},
        )
