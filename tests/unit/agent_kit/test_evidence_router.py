"""EvidenceTopicRouter (corpus evidence, no model), its selection rule, the per-run vector memo
and how RouteNode / RetrieveNode use them."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from langgraph.runtime import Runtime
from langgraph.types import Command

from agintis.agent_kit.nodes import (
    DEFAULT_EVIDENCE_AGGREGATE,
    DEFAULT_EVIDENCE_MARGIN,
    EvidenceTopic,
    EvidenceTopicRouter,
    RetrieveNode,
    RouteNode,
    RouteScope,
    TopicEvidence,
    aggregate_similarities,
    fallback_topics,
    select_by_evidence,
)
from agintis.agent_kit.runtime import MAX_QUERY_VECTORS, AgentContext, QueryVectors
from agintis.modules.knowledge.api import ProbeResult, ProfileEvidence, RagProfile
from agintis.shared_kernel.ids import RunId, TenantId
from tests.fakes.retrieval import FakeRetrieval, make_hit
from tests.unit.agent_kit.support import TENANT, runtime_for

NAMES = ["rules", "notices", "faq"]
PROFILES = {n: RagProfile(f"p_{n}", 3, 100, "simple", "structured") for n in NAMES}
TOPICS = [EvidenceTopic(n, PROFILES[n]) for n in NAMES]
VECTOR = (0.1, 0.2, 0.3, 0.4)


@dataclass
class ProbeCall:
    tenant_id: TenantId
    profiles: list[str]
    query: str
    k: int
    query_vector: Sequence[float] | None


class FakeProbe:
    """Answers with the scripted similarities per PROFILE name and records every call."""

    def __init__(self, similarities: dict[str, Sequence[float]], vector: Sequence[float] = VECTOR):
        self._similarities = similarities
        self._vector = tuple(vector)
        self.calls: list[ProbeCall] = []

    async def execute(
        self,
        tenant_id: TenantId,
        profiles: Sequence[RagProfile],
        query: str,
        k: int = 3,
        query_vector: Sequence[float] | None = None,
    ) -> ProbeResult:
        self.calls.append(ProbeCall(tenant_id, [p.name for p in profiles], query, k, query_vector))
        return ProbeResult(
            self._vector,
            tuple(
                ProfileEvidence(p.name, tuple(self._similarities.get(p.name, ()))) for p in profiles
            ),
        )


def _scores(**by_topic: Sequence[float]) -> dict[str, Sequence[float]]:
    return {PROFILES[n].name: s for n, s in by_topic.items()}


def _router(probe: FakeProbe, **kwargs: Any) -> EvidenceTopicRouter:
    return EvidenceTopicRouter(probe, TOPICS, **kwargs)


def _scope(tenant: TenantId = TENANT) -> RouteScope:
    return RouteScope(tenant, QueryVectors())


def _names(routed: Sequence[Any]) -> list[str]:
    return [r["topic"] for r in routed]


class TestRanking:
    async def test_the_best_topic_is_the_primary_and_its_score_is_the_confidence(self) -> None:
        probe = FakeProbe(_scores(rules=[0.5], notices=[0.8, 0.6], faq=[0.7]))

        routed = await _router(probe, max_topics=3, margin=0.0).route("q", _scope())

        assert routed == [
            {"topic": "notices", "confidence": 0.8, "similarities": [0.8, 0.6]},
        ]

    async def test_the_probe_gets_the_tenant_every_profile_and_the_probe_k(self) -> None:
        probe = FakeProbe(_scores(rules=[0.5]))

        await _router(probe, probe_k=5).route("the query", _scope(TenantId("acme")))

        (call,) = probe.calls
        assert call.tenant_id == "acme"
        assert call.profiles == ["p_rules", "p_notices", "p_faq"]
        assert (call.query, call.k, call.query_vector) == ("the query", 5, None)

    async def test_ties_keep_the_spec_order(self) -> None:
        probe = FakeProbe(_scores(rules=[0.6], notices=[0.6], faq=[0.6]))

        routed = await _router(probe, max_topics=3, margin=0.0).route("q", _scope())

        assert _names(routed) == ["rules", "notices", "faq"]


class TestMarginSelection:
    async def test_extras_only_within_the_margin_of_the_best(self) -> None:
        probe = FakeProbe(_scores(rules=[0.70], notices=[0.69], faq=[0.60]))

        routed = await _router(probe, max_topics=3, margin=0.02).route("q", _scope())

        assert _names(routed) == ["rules", "notices"]

    async def test_a_gap_equal_to_the_margin_is_still_selected(self) -> None:
        probe = FakeProbe(_scores(rules=[0.75], notices=[0.73], faq=[0.5]))

        routed = await _router(probe, max_topics=3, margin=0.02).route("q", _scope())

        assert _names(routed) == ["rules", "notices"]

    async def test_max_topics_caps_the_selection(self) -> None:
        probe = FakeProbe(_scores(rules=[0.7], notices=[0.7], faq=[0.7]))

        routed = await _router(probe, max_topics=2, margin=0.5).route("q", _scope())

        assert _names(routed) == ["rules", "notices"]

    async def test_a_zero_margin_keeps_only_the_primary(self) -> None:
        probe = FakeProbe(_scores(rules=[0.7], notices=[0.699]))

        assert _names(await _router(probe, margin=0.0).route("q", _scope())) == ["rules"]

    async def test_defaults(self) -> None:
        assert DEFAULT_EVIDENCE_AGGREGATE == "max"
        assert pytest.approx(0.02) == DEFAULT_EVIDENCE_MARGIN


class TestZeroHits:
    async def test_a_topic_without_hits_is_never_selected_even_with_a_huge_margin(self) -> None:
        probe = FakeProbe(_scores(rules=[0.3], notices=[], faq=[]))

        routed = await _router(probe, max_topics=3, margin=1.0).route("q", _scope())

        assert _names(routed) == ["rules"]

    async def test_when_nothing_has_hits_the_fallback_topics_are_used(self) -> None:
        probe = FakeProbe({})

        routed = await _router(probe, max_topics=2).route("q", _scope())

        assert routed == fallback_topics(NAMES, 2)
        assert [r["confidence"] for r in routed] == [0.0, 0.0]


class TestAggregation:
    def test_aggregates(self) -> None:
        assert aggregate_similarities([0.9, 0.5, 0.1], "max") == 0.9
        assert aggregate_similarities([0.9, 0.5, 0.1], "mean_top_k") == pytest.approx(0.5)
        assert aggregate_similarities([], "max") is None
        assert aggregate_similarities([], "mean_top_k") is None

    async def test_the_mode_can_change_the_winner(self) -> None:
        # rules has one great chunk; notices has several good ones.
        probe = FakeProbe(_scores(rules=[0.80, 0.30, 0.20], notices=[0.70, 0.68, 0.66]))

        by_max = await _router(probe, aggregate="max", margin=0.0).route("q", _scope())
        by_mean = await _router(probe, aggregate="mean_top_k", margin=0.0).route("q", _scope())

        assert _names(by_max) == ["rules"]
        assert _names(by_mean) == ["notices"]
        assert by_mean[0]["confidence"] == pytest.approx(0.68)

    async def test_the_confidence_is_clamped_to_the_unit_interval(self) -> None:
        probe = FakeProbe(_scores(rules=[-0.2], notices=[-0.5]))

        routed = await _router(probe, margin=0.0).route("q", _scope())

        assert routed[0]["confidence"] == 0.0
        assert routed[0]["similarities"] == [-0.2]


class TestSelectByEvidence:
    def test_pure_selection(self) -> None:
        evidence = [
            TopicEvidence("a", (0.5,), 0.5),
            TopicEvidence("b", (), None),
            TopicEvidence("c", (0.49,), 0.49),
        ]

        assert _names(select_by_evidence(evidence, 3, 0.02)) == ["a", "c"]
        assert _names(select_by_evidence(evidence, 3, 0.0)) == ["a"]
        assert select_by_evidence([TopicEvidence("b", (), None)], 3, 1.0) == []


class TestScopeAndMemo:
    async def test_it_requires_a_scope(self) -> None:
        with pytest.raises(ValueError, match="RouteScope"):
            await _router(FakeProbe({})).route("q")

    async def test_the_vector_is_left_in_the_run_memo_under_the_probed_text(self) -> None:
        scope = _scope()

        await _router(FakeProbe(_scores(rules=[0.5]))).route("the query", scope)

        assert scope.vectors.get("the query") == VECTOR

    async def test_a_memoized_vector_is_handed_to_the_probe(self) -> None:
        scope = _scope()
        scope.vectors.put("the query", (9.0, 9.0))
        probe = FakeProbe(_scores(rules=[0.5]))

        await _router(probe).route("the query", scope)

        assert probe.calls[0].query_vector == (9.0, 9.0)

    async def test_the_query_is_cut_like_retrieval_cuts_it(self) -> None:
        scope = _scope()
        probe = FakeProbe(_scores(rules=[0.5]))

        await _router(probe, max_query_chars=5).route("0123456789", scope)

        assert probe.calls[0].query == "01234"
        assert scope.vectors.get("01234") == VECTOR
        assert scope.vectors.get("0123456789") is None

    async def test_evidence_lists_every_topic_with_its_score(self) -> None:
        probe = FakeProbe(_scores(rules=[0.5, 0.4], faq=[0.7]))

        evidence = await _router(probe, aggregate="mean_top_k").evidence("q", _scope())

        assert evidence == [
            TopicEvidence("rules", (0.5, 0.4), pytest.approx(0.45)),  # type: ignore[arg-type]
            TopicEvidence("notices", (), None),
            TopicEvidence("faq", (0.7,), 0.7),
        ]


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_topics": 0}, "max_topics"),
            ({"margin": -0.1}, "margin"),
            ({"margin": float("nan")}, "margin"),
            ({"aggregate": "median"}, "aggregate"),
            ({"probe_k": 0}, "probe_k"),
            ({"max_query_chars": 0}, "max_query_chars"),
        ],
    )
    def test_bad_knobs(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            _router(FakeProbe({}), **kwargs)

    def test_topics_must_be_a_unique_non_empty_set(self) -> None:
        probe = FakeProbe({})
        with pytest.raises(ValueError, match="at least one"):
            EvidenceTopicRouter(probe, [])
        with pytest.raises(ValueError, match="names"):
            EvidenceTopicRouter(probe, [TOPICS[0], EvidenceTopic("rules", PROFILES["faq"])])
        with pytest.raises(ValueError, match="profiles"):
            EvidenceTopicRouter(probe, [TOPICS[0], EvidenceTopic("other", PROFILES["rules"])])


class TestQueryVectors:
    def test_it_is_bounded_and_drops_the_oldest(self) -> None:
        memo = QueryVectors(max_entries=2)
        for text in ("a", "b", "c"):
            memo.put(text, (1.0,))

        assert (memo.get("a"), memo.get("b"), memo.get("c")) == (None, (1.0,), (1.0,))
        assert len(memo) == 2
        assert MAX_QUERY_VECTORS >= 4  # an attempt per rewrite plus the first one

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="max_entries"):
            QueryVectors(max_entries=0)

    def test_every_context_gets_its_own_memo_and_equality_ignores_it(self) -> None:
        first = AgentContext(TENANT, RunId("r"))
        second = AgentContext(TENANT, RunId("r"))

        first.query_vectors.put("q", (1.0,))

        assert second.query_vectors.get("q") is None
        assert first == second


class TestNodes:
    async def test_route_node_gives_the_router_the_runs_tenant_and_memo(self) -> None:
        probe = FakeProbe(_scores(rules=[0.9], notices=[0.89]))
        node = RouteNode(_router(probe, max_topics=2, margin=0.05), NAMES, 2)
        runtime = runtime_for(TenantId("acme"))

        command = await node({"messages": [], "query": "the query"}, runtime)

        assert isinstance(command, Command)
        assert probe.calls[0].tenant_id == "acme"
        assert runtime.context.query_vectors.get("the query") == VECTOR
        update: dict[str, Any] = command.update  # type: ignore[assignment]
        assert _names(update["routed_topics"]) == ["rules", "notices"]
        assert [s.arg["topic"] for s in command.goto] == ["rules", "notices"]  # type: ignore[union-attr]

    async def test_retrieve_node_reuses_the_vector_the_router_left_in_the_run(self) -> None:
        retrieval = FakeRetrieval((make_hit("body"),))
        node = RetrieveNode(retrieval, {"rules": PROFILES["rules"]})
        runtime: Runtime[AgentContext] = runtime_for()
        runtime.context.query_vectors.put("the query", VECTOR)

        await node({"topic": "rules", "query": "the query"}, runtime)

        assert [c.query_vector for c in retrieval.calls] == [VECTOR]

    async def test_retrieve_node_without_a_memoized_vector_passes_none(self) -> None:
        retrieval = FakeRetrieval((make_hit("body"),))
        node = RetrieveNode(retrieval, {"rules": PROFILES["rules"]})

        await node({"topic": "rules", "query": "another query"}, runtime_for())

        assert [c.query_vector for c in retrieval.calls] == [None]

    async def test_every_relaxation_step_reuses_the_same_vector(self) -> None:
        retrieval = FakeRetrieval()  # finds nothing: the ladder walks every step
        node = RetrieveNode(retrieval, {"rules": PROFILES["rules"]})
        runtime: Runtime[AgentContext] = runtime_for()
        runtime.context.query_vectors.put("q", VECTOR)
        clauses = [{"field": "x", "op": "eq", "value": 1}]

        await node({"topic": "rules", "query": "q", "filters": clauses}, runtime)  # type: ignore[typeddict-item]

        assert len(retrieval.calls) >= 2
        assert {c.query_vector for c in retrieval.calls} == {VECTOR}


class TestBias:
    async def test_the_bias_is_added_to_the_score_before_ranking(self) -> None:
        probe = FakeProbe(_scores(rules=[0.70], notices=[0.68]))
        topics = [
            EvidenceTopic("rules", PROFILES["rules"]),
            EvidenceTopic("notices", PROFILES["notices"], 0.05),
        ]
        router = EvidenceTopicRouter(probe, topics, margin=0.0)

        routed = await router.route("q", _scope())

        assert _names(routed) == ["notices"]
        assert routed[0]["confidence"] == pytest.approx(0.73)
        assert routed[0]["similarities"] == [0.68]  # the raw evidence stays raw

    async def test_evidence_keeps_the_raw_score_and_the_bias_apart(self) -> None:
        probe = FakeProbe(_scores(rules=[0.70]))
        topics = [EvidenceTopic("rules", PROFILES["rules"], -0.02)]
        router = EvidenceTopicRouter(probe, topics)

        (found,) = await router.evidence("q", _scope())

        assert found.score == pytest.approx(0.68)
        assert found.bias == -0.02
        assert found.raw_score == pytest.approx(0.70)

    async def test_a_topic_without_hits_gets_no_score_even_with_a_bias(self) -> None:
        topics = [EvidenceTopic("rules", PROFILES["rules"], 0.1)]

        (found,) = await EvidenceTopicRouter(FakeProbe({}), topics).evidence("q", _scope())

        assert (found.score, found.raw_score) == (None, None)

    def test_a_non_finite_bias_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="bias"):
            EvidenceTopicRouter(
                FakeProbe({}), [EvidenceTopic("rules", PROFILES["rules"], float("nan"))]
            )

    def test_non_finite_scores_are_never_selected(self) -> None:
        evidence = [TopicEvidence("a", (), float("nan")), TopicEvidence("b", (0.4,), 0.4)]

        assert _names(select_by_evidence(evidence, 2, 1.0)) == ["b"]
