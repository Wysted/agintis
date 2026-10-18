"""Hybrid retrieval: RRF, the retrieval-mode strategies and `RetrieveForTopic` wiring."""

import math
from collections.abc import Sequence

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.modules.knowledge.api import (
    BM25_HYBRID_RETRIEVAL_MODE,
    DENSE_RETRIEVAL_MODE,
    HYBRID_RETRIEVAL_MODE,
    HybridRetrieval,
    IngestDocument,
    QueryTooLongError,
    RagProfile,
    RagProfileRegistry,
    Reranker,
    RetrieveForTopic,
    ScoredChunk,
    SourceFile,
    UnknownStrategyError,
    parse_profile,
    validate_profiles,
)
from agintis.modules.knowledge.application import (
    BM25_LEXICAL_WEIGHT,
    CandidateRequest,
    DenseRetrieval,
    NoopReranker,
    RetrievedHit,
)
from agintis.modules.knowledge.domain import Chunk, MetadataField, reciprocal_rank_fusion
from agintis.modules.knowledge.infrastructure.registry import build_retrieval_modes
from agintis.shared_kernel.criteria import Filter, FilterExpr
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    FakeChunker,
    FakeParser,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryIngestLock,
    any_term_score,
    lexical_score,
    lexical_terms,
)

TENANT = TenantId("t")
TEXT = "\n\n".join(f"paragraph {i}" for i in range(10))


def _scored(*names: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(Chunk(ChunkId(n), DocumentId("d"), TENANT, "p", i, n), 1.0 - i / 10)
        for i, n in enumerate(names)
    ]


class TestReciprocalRankFusion:
    def test_scores_sum_the_reciprocal_ranks_of_each_branch(self) -> None:
        fused = reciprocal_rank_fusion(_scored("a", "b"), _scored("b", "c"), k=60)
        scores = {str(f.chunk.id): f.fused_score for f in fused}
        assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["c"] == pytest.approx(1 / 62)
        assert [f.chunk.id for f in fused] == ["b", "a", "c"]

    def test_ranks_and_branch_scores_are_reported_per_branch(self) -> None:
        dense, lexical = _scored("a", "b"), _scored("b", "c")
        by_id = {str(f.chunk.id): f for f in reciprocal_rank_fusion(dense, lexical)}
        assert (by_id["b"].dense_rank, by_id["b"].lexical_rank) == (2, 1)
        assert (by_id["a"].dense_rank, by_id["a"].lexical_rank) == (1, None)
        assert by_id["a"].dense_similarity == dense[0].score
        assert by_id["c"].dense_similarity is None
        assert (by_id["c"].lexical_rank, by_id["c"].dense_rank) == (2, None)

    def test_a_chunk_repeated_inside_a_branch_counts_once_at_its_first_rank(self) -> None:
        """A branch listing the same chunk twice must not double its weight."""
        dense = _scored("a", "b") + _scored("a")
        fused = {str(f.chunk.id): f for f in reciprocal_rank_fusion(dense, [], k=60)}
        assert fused["a"].dense_rank == 1
        assert fused["a"].fused_score == pytest.approx(1 / 61)
        assert fused["a"].dense_similarity == dense[0].score  # the first occurrence's score
        assert len(fused) == 2

    def test_equal_scores_break_by_id_and_limit_cuts(self) -> None:
        fused = reciprocal_rank_fusion(_scored("z", "y"), _scored("y", "z"))
        assert [f.chunk.id for f in fused] == ["y", "z"]  # same score: id order
        assert [f.chunk.id for f in reciprocal_rank_fusion(_scored("a", "b"), [], limit=1)] == ["a"]

    def test_one_branch_empty_keeps_the_other_order(self) -> None:
        assert [f.chunk.id for f in reciprocal_rank_fusion([], _scored("b", "a"))] == ["b", "a"]
        assert reciprocal_rank_fusion([], []) == []

    def test_invalid_parameters_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="k must"):
            reciprocal_rank_fusion([], [], k=0)
        with pytest.raises(ValueError, match="limit"):
            reciprocal_rank_fusion([], [], limit=0)

    def test_branch_weights_scale_each_reciprocal_rank(self) -> None:
        fused = reciprocal_rank_fusion(
            _scored("a", "b"), _scored("b", "c"), k=60, lexical_weight=0.5
        )
        scores = {str(f.chunk.id): f.fused_score for f in fused}
        assert scores["b"] == pytest.approx(1 / 62 + 0.5 / 61)
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["c"] == pytest.approx(0.5 / 62)

    def test_a_lighter_lexical_branch_cannot_outrank_the_dense_top_alone(self) -> None:
        dense, lexical = _scored("a", "b"), _scored("b", "c")
        lighter = reciprocal_rank_fusion(_scored("z"), _scored("c"), lexical_weight=0.5)
        assert [f.chunk.id for f in lighter] == ["z", "c"]  # equal weights would put c first
        assert reciprocal_rank_fusion(dense, lexical) == reciprocal_rank_fusion(
            dense, lexical, lexical_weight=1.0
        )

    @pytest.mark.parametrize("weight", [0.0, -1.0, math.nan])
    def test_non_positive_weights_are_rejected(self, weight: float) -> None:
        with pytest.raises(ValueError, match="lexical_weight"):
            reciprocal_rank_fusion([], [], lexical_weight=weight)


class TestFakeLexicalScorer:
    def test_terms_are_accent_free_lowercase_without_stopwords(self) -> None:
        assert lexical_terms("La AFILIACIÓN de los Trabajadores") == ["afiliacion", "trabajadores"]

    def test_every_query_term_must_occur(self) -> None:
        assert lexical_score(["afiliacion"], "la afiliación") > 0
        assert lexical_score(["afiliacion", "pago"], "la afiliación") == 0.0
        assert lexical_score([], "texto") == 0.0

    def test_the_bm25_stand_in_needs_any_query_term(self) -> None:
        assert any_term_score(["afiliacion", "pago"], "la afiliación") > 0
        assert any_term_score(["pago"], "la afiliación") == 0.0
        assert any_term_score([], "texto") == 0.0


class RecordingReranker(Reranker):
    """Reverses what it receives and remembers the order it was given."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        self.seen = [c.chunk.text for c in chunks]
        return [ScoredChunk(c.chunk, 100.0 + i) for i, c in enumerate(reversed(chunks))][:top_n]


async def _env(top_k: int = 3, **profile: object) -> tuple[RetrieveForTopic, RagProfile]:
    documents, chunks = InMemoryDocumentRepository(), InMemoryChunkRepository()
    embeddings = DeterministicFakeEmbedding(size=16)
    ingest_profile = RagProfile("p", top_k, 100, "fake", "fake")
    await IngestDocument(
        documents,
        chunks,
        {"fake": FakeParser()},
        {"fake": FakeChunker()},
        embeddings,
        FixedClock(),
        InMemoryIngestLock(),
    ).execute(TENANT, ingest_profile, SourceFile("file:///d.txt", "text/plain", TEXT.encode()))
    retrieve = RetrieveForTopic(
        documents,
        chunks,
        embeddings,
        rerankers={"none": NoopReranker(), "rev": RecordingReranker()},
        retrieval_modes=build_retrieval_modes(branch_depth=20, rrf_k=60),
    )
    return retrieve, RagProfile("p", top_k, 100, "fake", "fake", **profile)  # type: ignore[arg-type]


class TestRetrievalModes:
    def test_the_registry_exposes_every_key(self) -> None:
        modes = build_retrieval_modes(branch_depth=10, rrf_k=30)
        assert set(modes) == {
            DENSE_RETRIEVAL_MODE,
            HYBRID_RETRIEVAL_MODE,
            BM25_HYBRID_RETRIEVAL_MODE,
        }
        assert isinstance(modes[DENSE_RETRIEVAL_MODE], DenseRetrieval)
        assert isinstance(modes[BM25_HYBRID_RETRIEVAL_MODE], HybridRetrieval)

    def test_hybrid_parameters_are_validated(self) -> None:
        with pytest.raises(ValueError, match="branch_depth"):
            HybridRetrieval(branch_depth=0)
        with pytest.raises(ValueError, match="rrf_k"):
            HybridRetrieval(rrf_k=0)

    async def test_dense_candidates_carry_similarity_only(self) -> None:
        retrieve, _ = await _env()
        chunks = retrieve._chunks
        found = await DenseRetrieval().candidates(
            chunks, CandidateRequest(TENANT, "q", [1.0] * 16, None, 3)
        )
        assert len(found) == 3
        assert all(c.fused_score is None and c.lexical_rank is None for c in found)
        assert all(c.similarity == c.scored.score for c in found)

    async def test_hybrid_candidates_order_by_the_fused_score(self) -> None:
        retrieve, _ = await _env()
        vector = await retrieve._embeddings.aembed_query("paragraph 3")
        found = await HybridRetrieval(branch_depth=20).candidates(
            retrieve._chunks, CandidateRequest(TENANT, "paragraph 3", vector, None, 3)
        )
        assert found[0].scored.chunk.text == "paragraph 3"
        assert found[0].lexical_rank == 1
        assert found[0].dense_rank == 1
        assert found[0].scored.score == found[0].fused_score


class BranchSpy(InMemoryChunkRepository):
    """Records which branch searches ran, with their filter, depth and metadata schema."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, FilterExpr | None, int, tuple[MetadataField, ...]]] = []

    async def search(  # type: ignore[no-untyped-def]
        self, tenant_id, query_vector, filter, k, *, metadata_fields=()
    ):
        self.calls.append(("dense", filter, k, tuple(metadata_fields)))
        return await super().search(
            tenant_id, query_vector, filter, k, metadata_fields=metadata_fields
        )

    async def search_lexical(  # type: ignore[no-untyped-def]
        self, tenant_id, query_text, filter, k, *, metadata_fields=()
    ):
        self.calls.append(("lexical", filter, k, tuple(metadata_fields)))
        return await super().search_lexical(
            tenant_id, query_text, filter, k, metadata_fields=metadata_fields
        )

    async def search_bm25(  # type: ignore[no-untyped-def]
        self, tenant_id, query_text, filter, k, *, metadata_fields=()
    ):
        self.calls.append(("bm25", filter, k, tuple(metadata_fields)))
        return await super().search_bm25(
            tenant_id, query_text, filter, k, metadata_fields=metadata_fields
        )


def _spy_chunk(name: str, text: str) -> Chunk:
    return Chunk(ChunkId(name), DocumentId("d"), TENANT, "p", int(name[-1]), text, {"area": "x"})


class TestBm25HybridRetrieval:
    async def test_fuses_dense_with_the_bm25_branch_at_the_measured_weight(self) -> None:
        spy = BranchSpy()
        # c1 is the dense top; c2 matches only one of the two query terms (OR, unlike plainto).
        await spy.add_many(
            [_spy_chunk("c1", "otra cosa"), _spy_chunk("c2", "afiliacion sola")],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        bm25_hybrid = build_retrieval_modes(branch_depth=5, rrf_k=60)[BM25_HYBRID_RETRIEVAL_MODE]
        found = await bm25_hybrid.candidates(
            spy, CandidateRequest(TENANT, "afiliacion pago", [1.0, 0.0], None, 2)
        )
        by_id = {str(c.scored.chunk.id): c for c in found}
        assert [str(c.scored.chunk.id) for c in found] == ["c2", "c1"]
        assert by_id["c2"].lexical_rank == 1
        assert by_id["c2"].fused_score == pytest.approx(1 / 62 + BM25_LEXICAL_WEIGHT / 61)
        assert by_id["c1"].fused_score == pytest.approx(1 / 61)
        assert by_id["c1"].lexical_rank is None
        assert by_id["c1"].similarity == pytest.approx(1.0)

    async def test_both_branches_get_the_same_filter_schema_and_depth(self) -> None:
        spy = BranchSpy()
        schema = (MetadataField("area", "str", filterable=True),)
        expr = Filter.eq("area", "x")
        request = CandidateRequest(TENANT, "q", [1.0, 0.0], expr, 7, schema)
        await HybridRetrieval(branch_depth=3, bm25=True).candidates(spy, request)
        assert spy.calls == [("dense", expr, 7, schema), ("bm25", expr, 7, schema)]
        spy.calls.clear()
        await HybridRetrieval(branch_depth=3).candidates(spy, request)  # plain hybrid
        assert spy.calls == [("dense", expr, 7, schema), ("lexical", expr, 7, schema)]

    async def test_a_bm25_profile_retrieves_through_retrieve_for_topic(self) -> None:
        retrieve, profile = await _env(retrieval_mode=BM25_HYBRID_RETRIEVAL_MODE)
        top = (await retrieve.execute(TENANT, profile, "paragraph 3")).hits[0]
        assert top.chunk.chunk.text == "paragraph 3"
        assert top.score == top.fused_score
        assert (top.dense_rank, top.lexical_rank) == (1, 1)


class TestRetrieveForTopicModes:
    async def test_dense_is_the_default_and_reports_no_fusion_evidence(self) -> None:
        retrieve, profile = await _env()
        hit = (await retrieve.execute(TENANT, profile, "paragraph 3")).hits[0]
        assert profile.retrieval_mode == DENSE_RETRIEVAL_MODE
        assert hit.similarity is not None
        assert (hit.fused_score, hit.lexical_rank, hit.dense_rank) == (None, None, None)
        assert hit.score == hit.similarity

    async def test_hybrid_hits_carry_the_fusion_evidence_and_order_by_it(self) -> None:
        retrieve, profile = await _env(retrieval_mode=HYBRID_RETRIEVAL_MODE)
        result = await retrieve.execute(TENANT, profile, "paragraph 3")
        top = result.hits[0]
        assert top.chunk.chunk.text == "paragraph 3"
        assert top.fused_score is not None
        assert top.lexical_rank == 1
        assert top.dense_rank == 1
        assert top.score == top.fused_score == top.chunk.score
        assert [h.score for h in result.hits] == sorted(
            (h.score for h in result.hits), reverse=True
        )

    async def test_similarity_is_missing_when_only_the_lexical_branch_found_the_chunk(
        self,
    ) -> None:
        retrieve, profile = await _env(top_k=10, retrieval_mode=HYBRID_RETRIEVAL_MODE)
        hits = (await retrieve.execute(TENANT, profile, "paragraph 3")).hits
        assert {h.chunk.chunk.text for h in hits} >= {"paragraph 3"}
        for hit in hits:
            assert isinstance(hit, RetrievedHit)
            assert hit.fused_score is not None
            assert hit.score == hit.fused_score
            assert (hit.similarity is None) == (hit.dense_rank is None)

    async def test_the_reranker_runs_after_the_fusion(self) -> None:
        retrieve, profile = await _env(retrieval_mode=HYBRID_RETRIEVAL_MODE, reranker_key="rev")
        result = await retrieve.execute(TENANT, profile, "paragraph 3")
        reranker = retrieve._rerankers["rev"]
        assert isinstance(reranker, RecordingReranker)
        assert reranker.seen[0] == "paragraph 3"  # the reranker got the fused order
        top = result.hits[0]
        assert top.rerank_score is not None
        assert top.score == top.rerank_score
        assert top.fused_score is not None  # the evidence survives reranking

    async def test_unknown_retrieval_mode_fails_loudly(self) -> None:
        retrieve, profile = await _env(retrieval_mode="nope")
        with pytest.raises(UnknownStrategyError, match="nope"):
            await retrieve.execute(TENANT, profile, "q")

    async def test_a_retrieve_use_case_without_modes_serves_dense_only(self) -> None:
        retrieve, _ = await _env()
        dense_only = RetrieveForTopic(retrieve._documents, retrieve._chunks, retrieve._embeddings)
        hybrid = RagProfile("p", 3, 100, "fake", "fake", retrieval_mode=HYBRID_RETRIEVAL_MODE)
        with pytest.raises(UnknownStrategyError, match="hybrid"):
            await dense_only.execute(TENANT, hybrid, "q")


class RecordingEmbeddings(DeterministicFakeEmbedding):
    """Remembers the texts it was asked to embed as queries."""

    queries: list[str] = []  # noqa: RUF012 - pydantic model field of the langchain base class

    async def aembed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return await super().aembed_query(text)


class TestQueryHandling:
    async def _retrieve(self, max_query_chars: int = 50) -> tuple[RetrieveForTopic, RagProfile]:
        base, profile = await _env(retrieval_mode=HYBRID_RETRIEVAL_MODE)
        embeddings = RecordingEmbeddings(size=16, queries=[])
        retrieve = RetrieveForTopic(
            base._documents,
            base._chunks,
            embeddings,
            retrieval_modes=build_retrieval_modes(branch_depth=20, rrf_k=60),
            max_query_chars=max_query_chars,
        )
        return retrieve, profile

    async def test_nul_and_lone_surrogates_are_removed_once_for_every_consumer(self) -> None:
        retrieve, profile = await self._retrieve()
        result = await retrieve.execute(TENANT, profile, "para\x00graph\ud800 3")
        assert isinstance(retrieve._embeddings, RecordingEmbeddings)
        assert retrieve._embeddings.queries == ["paragraph 3"]
        assert result.hits[0].chunk.chunk.text == "paragraph 3"

    async def test_search_operators_are_plain_text(self) -> None:
        """`OR` is a term to match (AND), not syntax: it cannot widen the lexical branch."""
        retrieve, profile = await self._retrieve()
        widened = await retrieve.execute(TENANT, profile, "paragraph 3 OR paragraph 4")
        assert widened.hits  # the dense branch still answers
        assert all(hit.lexical_rank is None for hit in widened.hits)
        plain = await retrieve.execute(TENANT, profile, "paragraph 3")
        assert plain.hits[0].lexical_rank == 1

    async def test_a_query_over_the_cap_is_rejected_before_any_work(self) -> None:
        retrieve, profile = await self._retrieve(max_query_chars=10)
        with pytest.raises(QueryTooLongError, match="11 characters"):
            await retrieve.execute(TENANT, profile, "x" * 11)
        assert isinstance(retrieve._embeddings, RecordingEmbeddings)
        assert retrieve._embeddings.queries == []
        assert (await retrieve.execute(TENANT, profile, "x" * 10)).hits is not None  # at the cap

    async def test_the_cap_counts_the_sanitized_query(self) -> None:
        retrieve, profile = await self._retrieve(max_query_chars=10)
        await retrieve.execute(TENANT, profile, "\x00" * 50 + "paragraph")  # 9 real characters

    async def test_a_query_with_nothing_left_returns_no_hits_and_embeds_nothing(self) -> None:
        retrieve, profile = await self._retrieve()
        assert (await retrieve.execute(TENANT, profile, "\x00\x00 ")).hits == ()
        assert isinstance(retrieve._embeddings, RecordingEmbeddings)
        assert retrieve._embeddings.queries == []

    def test_the_cap_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_query_chars"):
            RetrieveForTopic(
                InMemoryDocumentRepository(),
                InMemoryChunkRepository(),
                DeterministicFakeEmbedding(size=4),
                max_query_chars=0,
            )


class TestProfileRetrievalMode:
    def test_it_is_not_part_of_the_fingerprint(self) -> None:
        dense = RagProfile("p", 3, 100, "simple", "structured")
        hybrid = RagProfile("p", 3, 100, "simple", "structured", retrieval_mode="hybrid")
        assert dense.fingerprint == hybrid.fingerprint

    def test_empty_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="retrieval_mode"):
            RagProfile("p", 3, 100, "simple", "structured", retrieval_mode="")

    def test_yaml_default_and_override(self) -> None:
        raw = {
            "key": "p",
            "top_k": 3,
            "chunk_max_tokens": 100,
            "parser_key": "simple",
            "chunker_key": "structured",
        }
        assert parse_profile(raw, origin="test").retrieval_mode == "dense"
        hybrid = parse_profile({**raw, "retrieval_mode": "hybrid"}, origin="test")
        assert hybrid.retrieval_mode == "hybrid"

    def test_unknown_modes_fail_validation_at_boot(self) -> None:
        profile = RagProfile("p", 3, 100, "simple", "structured", retrieval_mode="graph")
        registry = RagProfileRegistry([profile])
        with pytest.raises(UnknownStrategyError, match="retrieval mode 'graph'"):
            validate_profiles(registry, ["simple"], ["structured"])
        validate_profiles(
            registry, ["simple"], ["structured"], retrieval_mode_keys=["dense", "graph"]
        )
