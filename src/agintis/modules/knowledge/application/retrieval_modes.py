"""Retrieval strategies: how a profile turns a query into the candidate chunks (`retrieval_mode`).

`dense` is vector search only; `hybrid` fuses it with full-text search (RRF) and `bm25_hybrid`
with a BM25 search (weighted RRF, needs the pg_textsearch extension). All only produce
CANDIDATES: READY re-checking, reranking and trimming stay in `RetrieveForTopic`. A strategy
never builds filters or tenant scope itself; it receives the final `filter` and passes it
unchanged to every branch.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from agintis.modules.knowledge.domain import (
    BM25_HYBRID_RETRIEVAL_MODE,
    DEFAULT_BRANCH_DEPTH,
    DEFAULT_RRF_K,
    DENSE_RETRIEVAL_MODE,
    HYBRID_RETRIEVAL_MODE,
    ChunkRepository,
    FusedChunk,
    MetadataField,
    ScoredChunk,
    reciprocal_rank_fusion,
)
from agintis.shared_kernel.criteria import FilterExpr
from agintis.shared_kernel.ids import TenantId

__all__ = [
    "BM25_HYBRID_RETRIEVAL_MODE",
    "BM25_LEXICAL_WEIGHT",
    "DENSE_RETRIEVAL_MODE",
    "HYBRID_RETRIEVAL_MODE",
    "Bm25HybridRetrieval",
    "Candidate",
    "CandidateRequest",
    "DenseRetrieval",
    "HybridRetrieval",
    "RetrievalStrategy",
]

# RRF weight of the BM25 branch (dense = 1.0). Measured on pre-registered dev sets: 0.5 gained
# MRR@10 on fresh questions while staying within -0.01 of dense on the older sets; 1.0 did not.
BM25_LEXICAL_WEIGHT = 0.5


@dataclass(frozen=True)
class CandidateRequest:
    tenant_id: TenantId
    query: str
    vector: Sequence[float]
    filter: FilterExpr | None
    k: int
    metadata_fields: Sequence[MetadataField] = ()


@dataclass(frozen=True)
class Candidate:
    """A candidate chunk. `scored.score` is the ordering key of the strategy (cosine
    similarity for dense, the fused RRF score for hybrid); the rest is the evidence."""

    scored: ScoredChunk
    # Cosine similarity; None for a chunk only the lexical branch found.
    similarity: float | None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    fused_score: float | None = None


class RetrievalStrategy(ABC):
    @abstractmethod
    async def candidates(
        self, chunks: ChunkRepository, request: CandidateRequest
    ) -> list[Candidate]:
        """Best-first candidates, at most `request.k`."""


class DenseRetrieval(RetrievalStrategy):
    async def candidates(
        self, chunks: ChunkRepository, request: CandidateRequest
    ) -> list[Candidate]:
        scored = await chunks.search(
            request.tenant_id,
            request.vector,
            request.filter,
            request.k,
            metadata_fields=request.metadata_fields,
        )
        return [Candidate(s, similarity=s.score) for s in scored]


class HybridRetrieval(RetrievalStrategy):
    """Dense + lexical branches fused with Reciprocal Rank Fusion.

    `branch_depth` candidates are taken from each branch (at least `k`); `rrf_k` damps the
    weight of the top ranks.
    """

    def __init__(
        self, *, branch_depth: int = DEFAULT_BRANCH_DEPTH, rrf_k: int = DEFAULT_RRF_K
    ) -> None:
        if branch_depth < 1:
            raise ValueError("branch_depth must be at least 1")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self._branch_depth = branch_depth
        self._rrf_k = rrf_k

    async def candidates(
        self, chunks: ChunkRepository, request: CandidateRequest
    ) -> list[Candidate]:
        fused = await chunks.search_hybrid(
            request.tenant_id,
            request.vector,
            request.query,
            request.filter,
            request.k,
            metadata_fields=request.metadata_fields,
            branch_depth=self._branch_depth,
            rrf_k=self._rrf_k,
        )
        return _fused_candidates(fused)


class Bm25HybridRetrieval(HybridRetrieval):
    """Dense + BM25 branches fused with weighted RRF (`BM25_LEXICAL_WEIGHT`).

    Same depth and `rrf_k` as `HybridRetrieval`; only the lexical branch and its weight differ.
    """

    async def candidates(
        self, chunks: ChunkRepository, request: CandidateRequest
    ) -> list[Candidate]:
        if request.k <= 0:
            raise ValueError("k must be greater than 0")
        depth = max(self._branch_depth, request.k)
        dense = await chunks.search(
            request.tenant_id,
            request.vector,
            request.filter,
            depth,
            metadata_fields=request.metadata_fields,
        )
        lexical = await chunks.search_bm25(
            request.tenant_id,
            request.query,
            request.filter,
            depth,
            metadata_fields=request.metadata_fields,
        )
        fused = reciprocal_rank_fusion(
            dense, lexical, k=self._rrf_k, limit=request.k, lexical_weight=BM25_LEXICAL_WEIGHT
        )
        return _fused_candidates(fused)


def _fused_candidates(fused: Sequence[FusedChunk]) -> list[Candidate]:
    return [
        Candidate(
            ScoredChunk(f.chunk, f.fused_score),
            similarity=f.dense_similarity,
            dense_rank=f.dense_rank,
            lexical_rank=f.lexical_rank,
            fused_score=f.fused_score,
        )
        for f in fused
    ]
