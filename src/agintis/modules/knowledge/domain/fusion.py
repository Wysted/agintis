"""Reciprocal Rank Fusion of the dense and lexical result lists of a hybrid search.

Pure and deterministic: the fused order is (score descending, chunk id ascending in code-point
order), so equal scores never depend on dict or database ordering. Only RANKS are fused (cosine
similarity and `ts_rank_cd` live on unrelated scales), which is the point of RRF.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from agintis.modules.knowledge.domain.models import Chunk, ScoredChunk
from agintis.shared_kernel.ids import ChunkId

DEFAULT_RRF_K = 60  # the constant of the original RRF paper; damps the weight of the top ranks
DEFAULT_BRANCH_DEPTH = 50  # candidates taken from EACH branch before fusing


@dataclass(frozen=True)
class FusedChunk:
    """A chunk with the evidence of the branches that found it (ranks are 1-based)."""

    chunk: Chunk
    fused_score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    # Cosine similarity, only when the dense branch returned the chunk. The lexical score is
    # deliberately not kept: `ts_rank_cd` values are not comparable across queries or with a
    # similarity, and only the lexical RANK takes part in the fusion.
    dense_similarity: float | None = None


def reciprocal_rank_fusion(
    dense: Sequence[ScoredChunk],
    lexical: Sequence[ScoredChunk],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
    lexical_weight: float = 1.0,
) -> list[FusedChunk]:
    """Fuse two best-first lists: `score(c) = sum over branches of weight / (k + rank)`.

    The dense branch weighs 1; `lexical_weight` 1 (the default) is plain RRF, a smaller one
    lowers how far the lexical branch alone can lift a chunk.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not lexical_weight > 0:  # also rejects NaN
        raise ValueError("lexical_weight must be greater than 0")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than 0")
    chunks: dict[ChunkId, Chunk] = {}
    dense_rank: dict[ChunkId, int] = {}
    lexical_rank: dict[ChunkId, int] = {}
    for rank, hit in enumerate(dense, start=1):
        chunks.setdefault(hit.chunk.id, hit.chunk)
        dense_rank.setdefault(hit.chunk.id, rank)
    for rank, hit in enumerate(lexical, start=1):
        chunks.setdefault(hit.chunk.id, hit.chunk)
        lexical_rank.setdefault(hit.chunk.id, rank)
    dense_score = {h.chunk.id: h.score for h in reversed(dense)}

    fused = [
        FusedChunk(
            chunk=chunk,
            fused_score=sum(
                weight / (k + ranks[chunk_id])
                for ranks, weight in ((dense_rank, 1.0), (lexical_rank, lexical_weight))
                if chunk_id in ranks
            ),
            dense_rank=dense_rank.get(chunk_id),
            lexical_rank=lexical_rank.get(chunk_id),
            dense_similarity=dense_score.get(chunk_id),
        )
        for chunk_id, chunk in chunks.items()
    ]
    fused.sort(key=lambda f: (-f.fused_score, f.chunk.id))
    return fused if limit is None else fused[:limit]
