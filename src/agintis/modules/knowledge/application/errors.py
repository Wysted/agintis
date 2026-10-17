"""Application-level errors."""

from agintis.shared_kernel.errors import DomainError


class UnknownStrategyError(DomainError):
    """A RagProfile references a parser/chunker key with no registered implementation."""


class QueryTooLongError(DomainError):
    """The retrieval query exceeds the configured length cap (`max_query_chars`).

    Rejected instead of truncated: a silently cut question changes its meaning. Every query
    term becomes a lexeme of the full-text branch and a token of the embedding input, so the cap
    bounds the work one request can ask for.
    """


class EmptyDocumentError(DomainError):
    """Parsing/chunking produced no chunks.

    Indexing zero chunks would mark the document READY and make the content-hash check skip
    it forever, hiding a parse failure. Failing loudly (before touching storage) is safer.
    """


class ConcurrentIngestError(DomainError):
    """The document being ingested was removed or changed by another writer mid-flight.

    Raised when the conditional PENDING -> READY transition does not apply. The attempt is
    compensated; the caller may retry.
    """


class UnsupportedDocumentTypeError(DomainError):
    """The parser cannot handle the MIME type of the source."""


class ScannedDocumentError(DomainError):
    """The document has no extractable text layer (e.g. a scanned PDF).

    The lightweight parser does not do OCR; use a profile whose parser does (`ocr_index_card`).
    """


class UnparseableDocumentError(DomainError):
    """The source is corrupt, encrypted or otherwise unreadable by the parser."""


class DocumentTooLargeError(DomainError):
    """The source exceeds a configured resource limit (bytes, pages, decompressed size)."""


class InvalidProfileError(DomainError, ValueError):
    """A profile definition (YAML file or parameters of one of its strategies) is invalid.

    Also a `ValueError`: it is a bad configuration value, whichever layer detects it.
    """
