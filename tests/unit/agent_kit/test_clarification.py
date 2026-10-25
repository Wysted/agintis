"""ClarificationPolicy (3.5c): the evidence bands, RouteNode's scope check and RefuseNode."""

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command, Overwrite, Send

from agintis.agent_kit.nodes import (
    DEFAULT_OUT_OF_SCOPE_MESSAGE,
    DEFAULT_SOCIAL_MESSAGE,
    EvidenceClarification,
    EvidenceTopic,
    EvidenceTopicRouter,
    NoClarification,
    RefuseNode,
    RouteNode,
    RouteScope,
    ScopeDecision,
    TopicEvidence,
    max_raw_score,
    scope_level,
)
from agintis.agent_kit.nodes.state import RoutedTopic
from agintis.agent_kit.runtime import QueryVectors
from agintis.modules.agents_catalog.api import Scope, Token
from agintis.modules.knowledge.api import RagProfile
from tests.fakes.retrieval import FakeTopicProbe
from tests.unit.agent_kit.support import TENANT, runtime_for

NAMES = ["rules", "notices", "faq"]
PROFILES = {n: RagProfile(f"p_{n}", 3, 100, "simple", "structured") for n in NAMES}
LOW = 0.5  # exactly representable, so the boundary comparisons are exact
OUT = 0.25


def _scores(**by_topic: Sequence[float]) -> dict[str, Sequence[float]]:
    return {PROFILES[n].name: s for n, s in by_topic.items()}


def _router(probe: FakeTopicProbe, bias: float = 0.0) -> EvidenceTopicRouter:
    topics = [EvidenceTopic(n, PROFILES[n], bias if n == "rules" else 0.0) for n in NAMES]
    return EvidenceTopicRouter(probe, topics)


def _scope() -> RouteScope:
    return RouteScope(TENANT, QueryVectors())


class TestBands:
    @pytest.mark.parametrize(
        ("score", "level"),
        [
            (0.9, "in"),
            (0.5000001, "in"),  # just above low_confidence_below
            (0.5, "low"),  # exactly low_confidence_below is NOT above it
            (0.4, "low"),
            (0.2500001, "low"),  # just above out_of_scope_below
            (0.25, "out"),  # exactly out_of_scope_below is NOT above it
            (0.0, "out"),
            (-0.3, "out"),
            (None, "out"),  # no evidence at all
        ],
    )
    def test_strict_comparison_at_the_boundaries(self, score: float | None, level: str) -> None:
        assert scope_level(score, LOW, OUT) == level

    def test_the_signal_is_the_max_raw_score_over_all_topics(self) -> None:
        evidence = [
            TopicEvidence("a", (0.3,), 0.3, bias=0.5),  # the bias must not leak into the signal
            TopicEvidence("b", (0.7, 0.6), 0.7),
            TopicEvidence("c", (), None),
        ]
        assert max_raw_score(evidence) == 0.7
        assert max_raw_score([TopicEvidence("c", (), None)]) is None
        assert max_raw_score([]) is None
        assert max_raw_score([TopicEvidence("n", (float("nan"),), float("nan"))]) is None


class TestEvidenceClarification:
    async def test_in_carries_the_topics_selected_from_the_same_probe(self) -> None:
        probe = FakeTopicProbe({"q": _scores(rules=[0.8, 0.2], notices=[0.3])})
        policy = EvidenceClarification(_router(probe), LOW, OUT)

        decision = await policy.decide("q", _scope())

        assert decision is not None
        assert (decision.level, decision.score) == ("in", 0.8)
        assert decision.routed is not None
        assert [r["topic"] for r in decision.routed] == ["rules"]
        assert probe.calls == 1  # the router was not asked again

    async def test_the_gray_band_is_low_and_still_routes(self) -> None:
        probe = FakeTopicProbe({"q": _scores(notices=[0.5])})
        decision = await EvidenceClarification(_router(probe), LOW, OUT).decide("q", _scope())

        assert decision is not None
        assert (decision.level, decision.score) == ("low", 0.5)
        assert decision.routed is not None
        assert decision.routed[0]["topic"] == "notices"

    async def test_out_has_no_routed_topics(self) -> None:
        probe = FakeTopicProbe({"q": _scores(faq=[0.25])})
        decision = await EvidenceClarification(_router(probe), LOW, OUT).decide("q", _scope())

        assert decision == ScopeDecision("out", 0.25, None)

    async def test_no_evidence_at_all_is_out_with_no_score(self) -> None:
        probe = FakeTopicProbe({"q": {}})
        decision = await EvidenceClarification(_router(probe), LOW, OUT).decide("q", _scope())

        assert decision == ScopeDecision("out", None, None)

    async def test_the_bias_never_moves_a_question_across_a_band(self) -> None:
        # raw 0.5 + bias 0.2 would be "in"; the policy reads the raw score: "low".
        probe = FakeTopicProbe({"q": _scores(rules=[0.5])})
        decision = await EvidenceClarification(_router(probe, bias=0.2), LOW, OUT).decide(
            "q", _scope()
        )

        assert decision is not None
        assert (decision.level, decision.score) == ("low", 0.5)

    @pytest.mark.parametrize(
        ("low", "out"), [(0.5, 0.5), (0.4, 0.5), (1.1, 0.2), (0.5, -0.1), (float("nan"), 0.1)]
    )
    def test_inconsistent_thresholds_are_rejected(self, low: float, out: float) -> None:
        with pytest.raises(ValueError, match="thresholds"):
            EvidenceClarification(_router(FakeTopicProbe({})), low, out)


class TestLowBandFanOut:
    @staticmethod
    def _wide(probe: FakeTopicProbe) -> EvidenceTopicRouter:
        return EvidenceTopicRouter(
            probe, [EvidenceTopic(n, PROFILES[n]) for n in NAMES], max_topics=1, margin=0.02
        )

    async def test_low_searches_the_best_topics_whatever_the_margin(self) -> None:
        probe = FakeTopicProbe({"q": _scores(rules=[0.5], notices=[0.3], faq=[0.1])})
        policy = EvidenceClarification(self._wide(probe), LOW, OUT, low_scope_max_topics=2)

        decision = await policy.decide("q", _scope())

        assert decision is not None
        assert decision.level == "low"
        assert decision.routed is not None
        assert [r["topic"] for r in decision.routed] == ["rules", "notices"]

    async def test_in_keeps_the_normal_selection(self) -> None:
        probe = FakeTopicProbe({"q": _scores(rules=[0.9], notices=[0.8])})
        policy = EvidenceClarification(self._wide(probe), LOW, OUT, low_scope_max_topics=3)

        decision = await policy.decide("q", _scope())

        assert decision is not None
        assert decision.routed is not None
        assert [r["topic"] for r in decision.routed] == ["rules"]

    @pytest.mark.parametrize("widen", [None, 1])
    async def test_not_above_max_topics_changes_nothing(self, widen: int | None) -> None:
        probe = FakeTopicProbe({"q": _scores(rules=[0.5], notices=[0.3])})
        policy = EvidenceClarification(self._wide(probe), LOW, OUT, widen)

        decision = await policy.decide("q", _scope())

        assert decision is not None
        assert decision.routed is not None
        assert [r["topic"] for r in decision.routed] == ["rules"]

    def test_a_zero_cap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="low_scope_max_topics"):
            EvidenceClarification(self._wide(FakeTopicProbe({})), LOW, OUT, 0)


class _Router:
    """A TopicRouter that records its calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def route(
        self, query: str, scope: RouteScope | None = None, *, widen_to: int | None = None
    ) -> list[RoutedTopic]:
        self.calls += 1
        return [RoutedTopic(topic="alpha", confidence=1.0)]


class _Policy:
    def __init__(self, decision: ScopeDecision | None) -> None:
        self._decision = decision
        self.queries: list[str] = []

    @property
    def may_refuse(self) -> bool:
        return True

    async def decide(self, query: str, scope: RouteScope) -> ScopeDecision | None:
        self.queries.append(query)
        return self._decision


def _node(router: _Router, policy: _Policy | None) -> RouteNode:
    return RouteNode(router, ["alpha", "beta"], 2, policy=policy)


def _sent_topics(command: Command[Any]) -> list[str]:
    assert isinstance(command.goto, list)
    return [send.arg["topic"] for send in command.goto if isinstance(send, Send)]


def _update(command: Command[Any]) -> dict[str, Any]:
    assert isinstance(command.update, dict)
    return command.update


def _state(attempts: int | None = None, kind: str | None = None) -> Any:
    state: dict[str, Any] = {"messages": [HumanMessage("q")], "query": "q"}
    if attempts is not None:
        state["attempts"] = attempts
    if kind is not None:
        state["request_kind"] = kind
    return state


class TestRouteNodeScopeCheck:
    async def test_out_goes_to_refuse_with_no_fan_out_and_emits_scope(self) -> None:
        router = _Router()
        events: list[Any] = []
        node = _node(router, _Policy(ScopeDecision("out", 0.1)))

        command = await node(_state(0), runtime_for(events=events))

        assert isinstance(command, Command)
        assert command.goto == "refuse"
        assert _update(command) == {
            "scope_level": "out",
            "scope_score": 0.1,
            "refusal_reason": "out_of_scope",
            "routed_topics": [],
            "topic_results": Overwrite([]),
        }
        assert events == [Scope("out", 0.1)]
        assert router.calls == 0

    async def test_low_emits_scope_and_uses_the_topics_of_the_decision(self) -> None:
        router = _Router()
        events: list[Any] = []
        routed = [RoutedTopic(topic="beta", confidence=0.4)]
        node = _node(router, _Policy(ScopeDecision("low", 0.4, routed)))

        command = await node(_state(0), runtime_for(events=events))

        assert events == [Scope("low", 0.4)]
        assert _sent_topics(command) == ["beta"]
        assert _update(command)["scope_level"] == "low"
        assert _update(command)["routed_topics"] == routed
        assert router.calls == 0  # one probe: the policy's, not the router's

    async def test_a_decision_without_topics_asks_the_router(self) -> None:
        router = _Router()
        events: list[Any] = []
        node = _node(router, _Policy(ScopeDecision("in", 0.9)))

        command = await node(_state(), runtime_for(events=events))  # no attempts key = first

        assert events == [Scope("in", 0.9)]
        assert router.calls == 1
        assert _sent_topics(command) == ["alpha"]

    async def test_a_rewrite_round_is_not_checked_again(self) -> None:
        router = _Router()
        events: list[Any] = []
        policy = _Policy(ScopeDecision("out", 0.0))
        node = _node(router, policy)

        command = await node(_state(1), runtime_for(events=events))

        assert policy.queries == []
        assert events == []
        assert "scope_level" not in _update(command)  # the turn keeps its level
        assert router.calls == 1

    @pytest.mark.parametrize(
        ("kind", "reason", "scope"),
        [("social", "social", []), ("other", "off_domain", [Scope("out", None)])],
    )
    @pytest.mark.parametrize(
        "decision", [ScopeDecision("low", 0.4), ScopeDecision("out", 0.1), None]
    )
    async def test_a_non_domain_request_below_the_in_band_is_refused_without_a_search(
        self, kind: str, reason: str, scope: list[Scope], decision: ScopeDecision | None
    ) -> None:
        router, policy = _Router(), _Policy(decision)
        events: list[Any] = []
        state = _state(0, kind)

        command = await _node(router, policy)(state, runtime_for(events=events))

        assert command.goto == "refuse"
        assert _update(command)["refusal_reason"] == reason
        assert _update(command).get("scope_level") == ("out" if scope else None)
        assert events == scope
        assert policy.queries == ["q"]  # probed once
        assert router.calls == 0

    @pytest.mark.parametrize("kind", ["social", "other"])
    async def test_a_non_domain_request_with_strong_evidence_continues_as_domain(
        self, kind: str
    ) -> None:
        router, policy = _Router(), _Policy(ScopeDecision("in", 0.9))
        events: list[Any] = []

        command = await _node(router, policy)(_state(0, kind), runtime_for(events=events))

        assert _sent_topics(command) == ["alpha"]
        assert "refusal_reason" not in _update(command)
        assert _update(command)["scope_level"] == "in"
        assert events == [Scope("in", 0.9)]
        assert policy.queries == ["q"]
        assert router.calls == 1

    @pytest.mark.parametrize("kind", ["domain", None])
    @pytest.mark.parametrize("level", ["in", "low"])
    async def test_a_domain_request_follows_the_band(self, kind: str | None, level: Any) -> None:
        routed = [RoutedTopic(topic="beta", confidence=0.4)]
        state = _state(0, kind)

        command = await _node(_Router(), _Policy(ScopeDecision(level, 0.4, routed)))(
            state, runtime_for()
        )

        assert _sent_topics(command) == ["beta"]
        assert "refusal_reason" not in _update(command)

    async def test_a_non_domain_request_is_ignored_without_a_refusing_policy(self) -> None:
        router = _Router()
        state = _state(0, "other")

        command = await _node(router, None)(state, runtime_for())

        assert _sent_topics(command) == ["alpha"]
        assert router.calls == 1

    async def test_a_rewrite_round_keeps_a_domain_turn_even_if_the_kind_says_otherwise(
        self,
    ) -> None:
        state = _state(1, "other")

        command = await _node(_Router(), _Policy(ScopeDecision("out", 0.0)))(state, runtime_for())

        assert _sent_topics(command) == ["alpha"]

    @pytest.mark.parametrize("policy", [None, NoClarification(), _Policy(None)])
    async def test_without_a_verdict_nothing_changes(self, policy: Any) -> None:
        router = _Router()
        events: list[Any] = []

        command = await _node(router, policy)(_state(0), runtime_for(events=events))

        assert events == []
        assert set(_update(command)) == {"routed_topics", "topic_results"}
        assert router.calls == 1


class TestRefuseNode:
    async def test_it_delivers_the_message_and_keeps_the_history_coherent(self) -> None:
        events: list[Any] = []
        state: Any = {"messages": [HumanMessage("weather?")]}

        update = await RefuseNode("Out of scope.")(state, runtime_for(events=events))

        assert events == [Token("Out of scope.")]
        (message,) = update["messages"]
        assert isinstance(message, AIMessage)
        assert message.content == "Out of scope."
        assert update["answer"] == "Out of scope."
        assert update["sources"] == []

    async def test_social_gets_the_courtesy_message_and_the_rest_the_refusal(self) -> None:
        node = RefuseNode("Out of scope.", "Hello, I can help with X.")
        for reason, text in [
            ("social", "Hello, I can help with X."),
            ("off_domain", "Out of scope."),
            ("out_of_scope", "Out of scope."),
            (None, "Out of scope."),
        ]:
            events: list[Any] = []
            state: Any = {"messages": [HumanMessage("hi")], "refusal_reason": reason}

            update = await node(state, runtime_for(events=events))

            assert events == [Token(text)]
            assert update["answer"] == text

    def test_the_default_message_is_not_blank_and_a_blank_one_is_rejected(self) -> None:
        assert DEFAULT_OUT_OF_SCOPE_MESSAGE.strip()
        assert DEFAULT_SOCIAL_MESSAGE.strip()
        with pytest.raises(ValueError, match="blank"):
            RefuseNode("  ")
        with pytest.raises(ValueError, match="blank"):
            RefuseNode("x", " ")
