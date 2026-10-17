"""Strategy registries for the lightweight (no Docling) knowledge infrastructure.

Plain dicts keyed by strategy key: bootstrap builds them once and passes them to the use cases
(`IngestDocument(parsers=..., chunkers=...)`, `RetrieveForTopic(rerankers=...)`). New strategies
are added here (or in bootstrap) without touching the use cases.

These are pure factories: PRIMITIVE arguments only, no `Settings` import and no defaults that
duplicate configuration. Bootstrap reads the settings and passes every value explicitly.
"""

from agintis.modules.knowledge.application import (
    NO_RERANKER_KEY,
    Chunker,
    Cleaner,
    DenseRetrieval,
    Extractor,
    HybridRetrieval,
    NoopReranker,
    Parser,
    Reranker,
    RetrievalStrategy,
    ValueTransform,
)
from agintis.modules.knowledge.domain import DENSE_RETRIEVAL_MODE, HYBRID_RETRIEVAL_MODE
from agintis.modules.knowledge.infrastructure.chunkers import (
    PATTERN_HIERARCHY_CHUNKER_KEY,
    RECORDS_CHUNKER_KEY,
    STRUCTURED_CHUNKER_KEY,
    WHOLE_DOCUMENT_CHUNKER_KEY,
    PatternHierarchyChunker,
    PatternHierarchyParams,
    RecordsChunker,
    RecordsParams,
    StructuredChunker,
    StructuredParams,
    TokenCounter,
    WholeDocumentChunker,
    WholeDocumentParams,
)
from agintis.modules.knowledge.infrastructure.cleaners import CLEANER_CLASSES
from agintis.modules.knowledge.infrastructure.extractors import (
    EXTRACTOR_CLASSES,
    TRANSFORM_CLASSES,
)
from agintis.modules.knowledge.infrastructure.parsers import (
    INDEX_CARD_PARSER_KEY,
    OCR_INDEX_CARD_PARSER_KEY,
    SIMPLE_PARSER_KEY,
    IndexCardParser,
    OcrIndexCardParser,
    ParseLimits,
    SimpleParser,
)
from agintis.modules.knowledge.infrastructure.tei import FallbackReranker, TeiReranker
from agintis.modules.knowledge.profiles import CleanerOrdering, ParamsModel

TEI_RERANKER_KEY = "tei"


def build_retrieval_modes(*, branch_depth: int, rrf_k: int) -> dict[str, RetrievalStrategy]:
    """`retrieval_mode` keys of a profile: vector search only, or fused with full-text."""
    return {
        DENSE_RETRIEVAL_MODE: DenseRetrieval(),
        HYBRID_RETRIEVAL_MODE: HybridRetrieval(branch_depth=branch_depth, rrf_k=rrf_k),
    }


def build_parsers(
    *,
    max_bytes: int,
    max_pages: int,
    max_uncompressed_bytes: int,
    max_compression_ratio: float,
) -> dict[str, Parser]:
    limits = ParseLimits(
        max_bytes=max_bytes,
        max_pages=max_pages,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_compression_ratio=max_compression_ratio,
    )
    return {
        SIMPLE_PARSER_KEY: SimpleParser(limits),
        INDEX_CARD_PARSER_KEY: IndexCardParser(limits),
        # Always registered (nothing imported until a scanned PDF): bootstrap checks the extra.
        OCR_INDEX_CARD_PARSER_KEY: OcrIndexCardParser(limits),
    }


def build_chunkers(counter: TokenCounter) -> dict[str, Chunker]:
    structured = StructuredChunker(counter)
    return {
        STRUCTURED_CHUNKER_KEY: structured,
        WHOLE_DOCUMENT_CHUNKER_KEY: WholeDocumentChunker(counter, structured),
        PATTERN_HIERARCHY_CHUNKER_KEY: PatternHierarchyChunker(counter),
        RECORDS_CHUNKER_KEY: RecordsChunker(counter),
    }


def chunker_param_schemas() -> dict[str, type[ParamsModel]]:
    """Params schema of each chunker key (same keys as `build_chunkers`)."""
    return {
        STRUCTURED_CHUNKER_KEY: StructuredParams,
        WHOLE_DOCUMENT_CHUNKER_KEY: WholeDocumentParams,
        PATTERN_HIERARCHY_CHUNKER_KEY: PatternHierarchyParams,
        RECORDS_CHUNKER_KEY: RecordsParams,
    }


def build_extractors() -> dict[str, Extractor]:
    """Whitelisted metadata extractors by key; profiles can only name these."""
    return {cls.key: cls() for cls in EXTRACTOR_CLASSES}


def extractor_param_schemas() -> dict[str, type[ParamsModel]]:
    """Params schema of each extractor key (same keys as `build_extractors`)."""
    return {cls.key: cls.params_model for cls in EXTRACTOR_CLASSES}


def build_transforms() -> dict[str, ValueTransform]:
    """Whitelisted metadata value transforms by key."""
    return {cls.key: cls() for cls in TRANSFORM_CLASSES}


def transform_param_schemas() -> dict[str, type[ParamsModel]]:
    """Params schema of each transform key (same keys as `build_transforms`)."""
    return {cls.key: cls.params_model for cls in TRANSFORM_CLASSES}


def build_cleaners() -> dict[str, Cleaner]:
    """Whitelisted cleaners by key; profiles can only name these (never import paths)."""
    return {cls.key: cls() for cls in CLEANER_CLASSES}


def cleaner_param_schemas() -> dict[str, type[ParamsModel]]:
    """Params schema of each cleaner key (same keys as `build_cleaners`)."""
    return {cls.key: cls.params_model for cls in CLEANER_CLASSES}


def cleaner_ordering() -> CleanerOrdering:
    """Order constraints declared by the cleaners (same keys as `build_cleaners`)."""
    return CleanerOrdering(
        rebuilds_text=frozenset(cls.key for cls in CLEANER_CLASSES if cls.rebuilds_text),
        read_only=frozenset(cls.key for cls in CLEANER_CLASSES if not cls.edits_text),
    )


def build_rerankers(
    *,
    tei_url: str | None,
    timeout_seconds: float,
    deadline_seconds: float,
    api_key: str | None = None,
) -> dict[str, Reranker]:
    """ "none" is always available; "tei" only when a reranker URL is configured.

    "tei" is wrapped in `FallbackReranker`: an unreachable or slow reranker degrades to the
    vector order instead of failing retrieval. A profile that names an UNREGISTERED key still
    fails fast in `validate_profiles`.
    """
    rerankers: dict[str, Reranker] = {NO_RERANKER_KEY: NoopReranker()}
    if tei_url:
        rerankers[TEI_RERANKER_KEY] = FallbackReranker(
            TeiReranker(
                tei_url,
                timeout_seconds=timeout_seconds,
                deadline_seconds=deadline_seconds,
                api_key=api_key,
            ),
            deadline_seconds=deadline_seconds,
        )
    return rerankers
