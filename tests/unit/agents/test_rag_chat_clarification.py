"""The rag-chat graph with a clarification policy (3.5c): scope bands, the refuse path, one
probe per turn, rewrite rounds and thread follow-ups; plus the spec block and its boot checks."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from agintis.agent_kit.nodes import (
    DEFAULT_OUT_OF_SCOPE_MESSAGE,
    DEFAULT_SOCIAL_MESSAGE,
    EvidenceClarification,
    NoClarification,
)
from agintis.bootstrap.adapters import ChainedErrorClassifier
from agintis.modules.agents.rag_chat import (
    AgentSpec,
    InvalidAgentSpecError,
    RagChatDeps,
    build_clarification,
    build_graph,
    build_router,
    parse_agent_spec,
    rag_chat_runner,
)
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    RunAgent,
    RunCompleted,
    RunFailed,
    Scope,
    Sources,
    Step,
    Token,
)
from agintis.modules.knowledge.api import RagProfile, RagProfileRegistry
from agintis.modules.knowledge.infrastructure.errors import RetrievalErrorClassifier
from agintis.modules.knowledge.infrastructure.tei import TeiUnavailableError
from agintis.platform.llm import ProviderErrorClassifier
from agintis.shared_kernel.ids import ThreadId
from tests.fakes.llm import ScriptedChatModel, chat_models_for
from tests.fakes.retrieval import FakeRetrieval, FakeTopicProbe
from tests.unit.agents.support import (
    CLARIFICATION,
    LOW,
    OUT,
    PROFILES,
    REFUSAL,
    Script,
    make_input,
    make_model,
    make_probe,
    make_retrieval,
    scoped_spec,
)


def _runner(
    model: ScriptedChatModel,
    probe: FakeTopicProbe,
    retrieval: FakeRetrieval,
    spec: AgentSpec | None = None,
    saver: InMemorySaver | None = None,
) -> Any:
    deps = RagChatDeps(chat_models_for(model), retrieval, probe=probe)
    return rag_chat_runner(spec or scoped_spec(), PROFILES, deps, saver)


async def _events(runner: Any, message: str, thread: ThreadId | None = None) -> list[Any]:
    return [e async for e in runner.astream(make_input(message, thread))]


def _steps(events: list[Any]) -> list[str]:
    return [e.node for e in events if isinstance(e, Step) and e.status == "started"]


def _scopes(events: list[Any]) -> list[Scope]:
    return [e for e in events if isinstance(e, Scope)]


def _text(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, Token))


class TestRefusal:
    async def test_out_of_scope_refuses_without_retrieval_synthesis_or_sources(self) -> None:
        model = make_model(Script([]), ["must not be used"])
        probe, retrieval = make_probe(weather=0.1), make_retrieval()
        registry = AgentRegistry((_runner(model, probe, retrieval),))
        run_agent = RunAgent(registry, ProviderErrorClassifier(), id_generator=lambda: "run-1")

        events = [e async for e in run_agent.start("scoped-rag", make_input().tenant_id, "weather")]

        content = events[1:-1]
        assert content[0] == Step("analyze", "started")
        assert _scopes(events) == [Scope("out", 0.1)]
        assert not [e for e in events if isinstance(e, Sources)]
        assert _text(events) == REFUSAL
        assert model.seen == []  # the synthesis model was never called
        assert len(model.structured_seen) == 1  # only the analyzer
        assert retrieval.calls == []
        assert probe.queries == ["weather"]  # one probe
        steps = _steps(events)
        assert steps == ["analyze", "route", "refuse"]
        assert isinstance(events[-1], RunCompleted)
        assert events[-1].finish_reason == "stop"

    async def test_scope_comes_before_the_message(self) -> None:
        model = make_model(Script([]), [])

        events = await _events(_runner(model, make_probe(q=0.1), make_retrieval()), "q")

        kinds = [type(e) for e in events]
        assert kinds.index(Scope) < kinds.index(Token)

    async def test_no_evidence_at_all_is_out_with_no_score(self) -> None:
        model = make_model(Script([]), [])

        events = await _events(_runner(model, make_probe(q=None), make_retrieval()), "q")

        assert _scopes(events) == [Scope("out", None)]
        assert _text(events) == REFUSAL

    async def test_no_evidence_at_all_logs_a_warning_about_the_tenant_or_the_ingest(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            await _events(
                _runner(make_model(Script([]), []), make_probe(q=None), make_retrieval()), "q"
            )

        assert "no topic has any evidence" in caplog.text

    async def test_a_weak_score_is_refused_without_that_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            await _events(
                _runner(make_model(Script([]), []), make_probe(q=0.1), make_retrieval()), "q"
            )

        assert "no topic has any evidence" not in caplog.text

    async def test_the_default_message_is_used_when_the_spec_gives_none(self) -> None:
        clarification = {k: v for k, v in CLARIFICATION.items() if k != "out_of_scope_message"}
        model = make_model(Script([]), [])

        events = await _events(
            _runner(
                model, make_probe(q=0.1), make_retrieval(), scoped_spec(clarification=clarification)
            ),
            "q",
        )

        assert _text(events) == DEFAULT_OUT_OF_SCOPE_MESSAGE


class TestRequestKind:
    @pytest.mark.parametrize(
        ("kind", "text", "scopes", "reason"),
        [
            ("social", "Courtesy.", [], "social"),
            ("other", REFUSAL, [Scope("out", None)], "off_domain"),
        ],
    )
    async def test_a_non_domain_request_out_of_scope_is_refused_with_no_retrieval_or_synthesis(
        self, kind: str, text: str, scopes: list[Scope], reason: str
    ) -> None:
        saver = InMemorySaver()
        clarification = {**CLARIFICATION, "social_message": "Courtesy."}
        spec = scoped_spec(clarification=clarification)
        model = make_model(Script([], request_kind=kind), ["must not be used"])
        probe, retrieval = make_probe(thanks=0.1), make_retrieval()
        runner = _runner(model, probe, retrieval, spec, saver)
        graph = build_graph(
            spec, PROFILES, RagChatDeps(chat_models_for(model), retrieval, probe=probe), saver
        )

        events = await _events(runner, "thanks", ThreadId("t-1"))
        values = (await graph.aget_state({"configurable": {"thread_id": "t-1"}})).values

        assert _text(events) == text
        assert _scopes(events) == scopes
        assert _steps(events) == ["analyze", "route", "refuse"]
        assert not [e for e in events if isinstance(e, Sources)]
        assert (model.seen, retrieval.calls, probe.queries) == ([], [], ["thanks"])
        assert values["refusal_reason"] == reason
        assert values["answer"] == text

    async def test_the_next_domain_turn_clears_the_reason(self) -> None:
        saver = InMemorySaver()
        model = make_model(Script([], request_kind="social"), [])
        probe = make_probe(deadline=0.9, hi=0.1)
        thread = ThreadId("t-1")
        await _events(_runner(model, probe, make_retrieval(), saver=saver), "hi", thread)
        model = make_model(Script([]), ["Real answer [1]."])
        runner = _runner(model, probe, make_retrieval(), saver=saver)
        graph = build_graph(
            scoped_spec(),
            PROFILES,
            RagChatDeps(chat_models_for(model), make_retrieval(), probe=probe),
            saver,
        )

        await _events(runner, "deadline", thread)
        values = (await graph.aget_state({"configurable": {"thread_id": "t-1"}})).values

        assert values["refusal_reason"] is None
        assert values["scope_level"] == "in"

    async def test_the_default_courtesy_message_is_used_when_the_spec_gives_none(self) -> None:
        model = make_model(Script([], request_kind="social"), [])

        events = await _events(_runner(model, make_probe(q=0.1), make_retrieval()), "q")

        assert _text(events) == DEFAULT_SOCIAL_MESSAGE

    @pytest.mark.parametrize("kind", ["social", "other"])
    async def test_a_non_domain_label_with_strong_evidence_is_answered(self, kind: str) -> None:
        model = make_model(Script([], request_kind=kind), ["Answer [1]."])
        probe = make_probe(q=0.9)

        events = await _events(_runner(model, probe, make_retrieval()), "q")

        assert _scopes(events) == [Scope("in", 0.9)]
        assert _text(events) == "Answer [1]."
        assert probe.queries == ["q"]

    async def test_a_domain_request_in_the_low_band_is_answered(self) -> None:
        model = make_model(Script([]), ["Answer [1]."])

        events = await _events(_runner(model, make_probe(q=0.4), make_retrieval()), "q")

        assert _scopes(events) == [Scope("low", 0.4)]
        assert _text(events) == "Answer [1]."


class TestProbeFailure:
    async def test_a_failing_probe_fails_the_run_and_is_never_read_as_out_of_scope(self) -> None:
        model = make_model(Script([]), ["must not be used"])
        probe = FakeTopicProbe({}, raises=TeiUnavailableError("embeddings down"))
        retrieval = make_retrieval()
        classifier = ChainedErrorClassifier(ProviderErrorClassifier(), RetrievalErrorClassifier())
        registry = AgentRegistry((_runner(model, probe, retrieval),))
        run_agent = RunAgent(registry, classifier, id_generator=lambda: "run-1")

        events = [e async for e in run_agent.start("scoped-rag", make_input().tenant_id, "weather")]

        failures = [e for e in events if isinstance(e, RunFailed)]
        assert [(f.code, f.retryable) for f in failures] == [("retrieval_unavailable", True)]
        assert events[-1] is failures[0]
        assert probe.queries == ["weather"]
        assert not [e for e in events if isinstance(e, Scope | Sources | RunCompleted)]
        assert REFUSAL not in _text(events)
        assert retrieval.calls == []
        assert model.seen == []


class TestBandsInTheGraph:
    @pytest.mark.parametrize(
        ("score", "level", "refused"),
        [
            (0.5000001, "in", False),
            (0.5, "low", False),
            (0.2500001, "low", False),
            (0.25, "out", True),
        ],
    )
    async def test_strict_boundaries(self, score: float, level: str, refused: bool) -> None:
        model = make_model(Script([]), ["Answer [1]."])
        retrieval = make_retrieval()

        events = await _events(_runner(model, make_probe(q=score), retrieval), "q")

        assert _scopes(events) == [Scope(level, score)]  # type: ignore[arg-type]
        assert (retrieval.calls == []) is refused
        assert (_text(events) == REFUSAL) is refused

    async def test_low_answers_normally_after_announcing_it(self) -> None:
        model = make_model(Script([]), ["Answer [1]."])
        retrieval = make_retrieval()

        events = await _events(_runner(model, make_probe(q=0.4), retrieval), "q")

        kinds = [type(e) for e in events]
        assert _scopes(events) == [Scope("low", 0.4)]
        assert kinds.index(Scope) < kinds.index(Sources) < kinds.index(Token)
        assert _text(events) == "Answer [1]."
        assert len(retrieval.calls) == 1  # alpha only (margin 0)

    async def test_in_announces_in_and_answers(self) -> None:
        model = make_model(Script([]), ["Answer [1]."])

        events = await _events(_runner(model, make_probe(q=0.9), make_retrieval()), "q")

        assert _scopes(events) == [Scope("in", 0.9)]
        assert _text(events) == "Answer [1]."

    async def test_the_signal_is_the_max_over_all_topics_not_the_selected_ones(self) -> None:
        # beta has the best evidence; alpha is not selected (margin 0). Max = 0.8 either way.
        probe = FakeTopicProbe({"q": {"alpha": [0.3], "beta": [0.8]}})

        events = await _events(
            _runner(make_model(Script([]), ["A."]), probe, make_retrieval()), "q"
        )

        assert _scopes(events) == [Scope("in", 0.8)]
        assert probe.calls == 1

    async def test_one_probe_per_turn_serves_the_scope_check_the_routing_and_the_retrieval(
        self,
    ) -> None:
        probe, retrieval = make_probe(q=0.9), make_retrieval()

        await _events(_runner(make_model(Script([]), ["A."]), probe, retrieval), "q")

        assert probe.queries == ["q"]
        assert [c.query_vector for c in retrieval.calls] == [(0.1,)]  # the probe's vector


GAMMA = RagProfile("gamma", 3, 100, "simple", "structured")
THREE_PROFILES = RagProfileRegistry([*(PROFILES.get(n) for n in ("alpha", "beta")), GAMMA])
THREE_TOPICS: list[dict[str, Any]] = [
    {"name": n, "description": n, "profile": n, "filters": {}} for n in ("alpha", "beta", "gamma")
]


def _wide_spec(**changes: Any) -> AgentSpec:
    """3 topics, max_topics 2, the `low` band widened to 3."""
    return scoped_spec(
        topics=THREE_TOPICS,
        max_topics=2,
        clarification={**CLARIFICATION, "low_scope_max_topics": 3},
        **changes,
    )


def _wide_probe(*queries: str) -> FakeTopicProbe:
    # all three topics inside the `low` band, alpha best, margin 0: only widening reaches gamma
    return FakeTopicProbe({q: {"alpha": [0.4], "beta": [0.38], "gamma": [0.36]} for q in queries})


def _searched(retrieval: FakeRetrieval, query: str) -> list[str]:
    return sorted(c.profile.name for c in retrieval.calls if c.query == query)


class TestLowBandWidening:
    async def test_the_low_band_searches_low_scope_max_topics(self) -> None:
        retrieval = make_retrieval()
        model = make_model(Script([]), ["A."])
        deps = RagChatDeps(chat_models_for(model), retrieval, probe=_wide_probe("q"))

        events = await _events(rag_chat_runner(_wide_spec(), THREE_PROFILES, deps), "q")

        assert _scopes(events) == [Scope("low", 0.4)]
        assert _searched(retrieval, "q") == ["alpha", "beta", "gamma"]

    async def test_the_rewrite_round_keeps_the_widened_fan_out(self) -> None:
        retrieval = make_retrieval()
        script = Script([], rewrites=["rewritten"])
        deps = RagChatDeps(
            chat_models_for(make_model(script, ["Final."])),
            retrieval,
            probe=_wide_probe("q", "rewritten"),
        )
        spec = _wide_spec(min_sources=10, max_sources=10, max_rewrites=1)

        await _events(rag_chat_runner(spec, THREE_PROFILES, deps), "q")

        assert _searched(retrieval, "q") == ["alpha", "beta", "gamma"]
        assert _searched(retrieval, "rewritten") == ["alpha", "beta", "gamma"]


class TestPolicyNone:
    async def test_it_emits_no_scope_and_routes_as_before(self) -> None:
        spec = scoped_spec(clarification={"policy": "none"})
        probe, retrieval = (
            make_probe(q=0.01),
            make_retrieval(),
        )  # would be "out" under the evidence policy

        events = await _events(_runner(make_model(Script([]), ["A."]), probe, retrieval, spec), "q")

        assert _scopes(events) == []
        assert _text(events) == "A."
        assert _steps(events)[:3] == ["analyze", "route", "topic"]
        assert probe.queries == ["q"]  # the router's own probe

    async def test_the_scope_level_stays_unchecked_in_the_state(self) -> None:
        saver = InMemorySaver()
        spec = scoped_spec(clarification={"policy": "none"})
        model, probe = make_model(Script([]), ["A."]), make_probe(q=0.9)
        await _events(_runner(model, probe, make_retrieval(), spec, saver), "q", ThreadId("t-1"))
        graph = build_graph(
            spec,
            PROFILES,
            RagChatDeps(chat_models_for(model), make_retrieval(), probe=probe),
            saver,
        )

        state = (await graph.aget_state({"configurable": {"thread_id": "t-1"}})).values

        assert (state["scope_level"], state["scope_score"]) == (None, None)

    async def test_no_clarification_block_is_the_same_as_none(self) -> None:
        spec = parse_agent_spec(
            {
                "id": "plain",
                "description": "d",
                "router": "evidence",
                "topics": [
                    {"name": "alpha", "description": "A", "profile": "alpha", "filters": {}},
                    {"name": "beta", "description": "B", "profile": "beta", "filters": {}},
                ],
                "sufficiency": "count",
            },
            origin="t",
        )

        events = await _events(
            _runner(make_model(Script([]), ["A."]), make_probe(q=0.01), make_retrieval(), spec), "q"
        )

        assert _scopes(events) == []
        deps = RagChatDeps(chat_models_for(), make_retrieval(), probe=make_probe())
        graph = build_graph(spec, PROFILES, deps)
        assert "refuse" not in graph.nodes  # unreachable without a policy: not added


class TestRoutingLog:
    @pytest.mark.parametrize("policy", ["evidence", "none"])
    async def test_the_first_route_logs_the_selected_topics_once(
        self, policy: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        spec = scoped_spec(
            clarification=CLARIFICATION if policy == "evidence" else {"policy": "none"}
        )

        with caplog.at_level("INFO"):
            await _events(
                _runner(make_model(Script([]), ["A."]), make_probe(q=0.9), make_retrieval(), spec),
                "q",
            )

        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Evidence")]
        assert lines == ["Evidence router: alpha=0.900 -> alpha"]


class TestRewriteRound:
    async def test_the_rewrite_does_not_re_evaluate_nor_re_emit_the_scope(self) -> None:
        # One hit, min_sources 2: insufficient, so the question is rewritten and re-routed. The
        # rewritten query scores in the OUT band: it must still be answered (level kept).
        spec = scoped_spec(min_sources=2, max_rewrites=1, max_topics=1)
        probe = make_probe(q=0.9, rewritten=0.05)
        retrieval = make_retrieval()
        script = Script([], rewrites=["rewritten"])

        events = await _events(_runner(make_model(script, ["Final."]), probe, retrieval, spec), "q")

        steps = _steps(events)
        assert steps.count("rewrite") == 1
        assert steps.count("route") == 2
        assert probe.queries == ["q", "rewritten"]  # routed twice, probed once per route
        assert _scopes(events) == [Scope("in", 0.9)]  # emitted once
        assert _text(events) == "Final."
        assert "refuse" not in steps


class TestThreads:
    async def test_a_follow_up_after_a_refusal_sees_a_coherent_history(self) -> None:
        saver = InMemorySaver()
        model = make_model(Script([]), ["Real answer [1]."])
        probe = make_probe(weather=0.1, deadline=0.9)
        runner = _runner(model, probe, make_retrieval(), saver=saver)
        thread = ThreadId("t-1")

        first = await _events(runner, "weather", thread)
        graph = build_graph(
            scoped_spec(),
            PROFILES,
            RagChatDeps(chat_models_for(model), make_retrieval(), probe=probe),
            saver,
        )
        config: RunnableConfig = {"configurable": {"thread_id": "t-1"}}
        refused = (await graph.aget_state(config)).values
        second = await _events(runner, "deadline", thread)
        answered = (await graph.aget_state(config)).values

        assert _scopes(first) == [Scope("out", 0.1)]
        assert refused["scope_level"] == "out"
        assert [type(m) for m in refused["messages"]] == [HumanMessage, AIMessage]
        assert refused["answer"] == REFUSAL
        assert refused["routed_topics"] == []
        # Turn 2: the analyzer and the synthesis saw the refusal as an assistant turn.
        analyzer_second = model.structured_seen[1]
        assert [m.content for m in analyzer_second[1:]] == ["weather", REFUSAL, "deadline"]
        assert [m.content for m in model.seen[0][1:3]] == ["weather", REFUSAL]
        assert _scopes(second) == [Scope("in", 0.9)]
        assert _text(second) == "Real answer [1]."
        # Fresh state: the new turn rewrote the verdict.
        assert (answered["scope_level"], answered["scope_score"]) == ("in", 0.9)
        assert [type(m) for m in answered["messages"]] == [
            HumanMessage,
            AIMessage,
            HumanMessage,
            AIMessage,
        ]

    async def test_a_refusal_after_an_answer_does_not_keep_the_old_sources(self) -> None:
        saver = InMemorySaver()
        model = make_model(Script([]), ["Real answer [1]."])
        probe = make_probe(deadline=0.9, weather=0.1)
        runner = _runner(model, probe, make_retrieval(), saver=saver)
        thread = ThreadId("t-1")

        await _events(runner, "deadline", thread)
        events = await _events(runner, "weather", thread)
        graph = build_graph(
            scoped_spec(),
            PROFILES,
            RagChatDeps(chat_models_for(model), make_retrieval(), probe=probe),
            saver,
        )
        config: RunnableConfig = {"configurable": {"thread_id": "t-1"}}
        state = (await graph.aget_state(config)).values

        assert not [e for e in events if isinstance(e, Sources)]
        assert state["sources"] == []
        assert state["answer"] == REFUSAL


class TestSpecBlock:
    def test_the_default_is_none(self) -> None:
        spec = parse_agent_spec(
            {
                "id": "x",
                "description": "d",
                "topics": [{"name": "a", "description": "d", "profile": "p"}],
            },
            origin="t",
        )
        assert spec.clarification.policy == "none"
        assert spec.clarification.out_of_scope_message == DEFAULT_OUT_OF_SCOPE_MESSAGE

    def test_a_valid_evidence_block_is_parsed(self) -> None:
        clarification = scoped_spec().clarification
        assert (clarification.low_confidence_below, clarification.out_of_scope_below) == (LOW, OUT)
        assert clarification.out_of_scope_message == REFUSAL

    def test_the_new_knobs_default_and_parse(self) -> None:
        default = scoped_spec().clarification
        assert (default.low_scope_max_topics, default.social_message) == (
            None,
            DEFAULT_SOCIAL_MESSAGE,
        )
        block = {**CLARIFICATION, "low_scope_max_topics": 3, "social_message": "Hi."}
        clarification = scoped_spec(clarification=block).clarification
        assert (clarification.low_scope_max_topics, clarification.social_message) == (3, "Hi.")

    @pytest.mark.parametrize(
        "extra", [{"low_scope_max_topics": 0}, {"low_scope_max_topics": 9}, {"social_message": " "}]
    )
    def test_the_new_knobs_are_validated(self, extra: dict[str, Any]) -> None:
        with pytest.raises(InvalidAgentSpecError):
            scoped_spec(clarification={**CLARIFICATION, **extra})

    def test_low_scope_max_topics_cannot_be_below_max_topics(self) -> None:
        block = {**CLARIFICATION, "low_scope_max_topics": 1}
        with pytest.raises(InvalidAgentSpecError, match="low_scope_max_topics"):
            scoped_spec(max_topics=2, clarification=block)

        equal = {**CLARIFICATION, "low_scope_max_topics": 2}
        assert scoped_spec(max_topics=2, clarification=equal).max_topics == 2  # no widening

    def test_the_evidence_policy_needs_the_evidence_router(self) -> None:
        with pytest.raises(InvalidAgentSpecError, match="needs `router: evidence`"):
            scoped_spec(router="llm")

    @pytest.mark.parametrize(
        "block",
        [
            {"policy": "evidence"},
            {"policy": "evidence", "low_confidence_below": 0.5},
            {"policy": "evidence", "out_of_scope_below": 0.2},
        ],
    )
    def test_the_evidence_policy_needs_both_thresholds(self, block: dict[str, Any]) -> None:
        with pytest.raises(InvalidAgentSpecError, match="needs low_confidence_below"):
            scoped_spec(clarification=block)

    @pytest.mark.parametrize(("low", "out"), [(0.5, 0.5), (0.3, 0.5), (0.0, 0.0)])
    def test_the_thresholds_must_be_strictly_ordered(self, low: float, out: float) -> None:
        block = {**CLARIFICATION, "low_confidence_below": low, "out_of_scope_below": out}
        with pytest.raises(InvalidAgentSpecError, match="out_of_scope_below < low_confidence"):
            scoped_spec(clarification=block)

    @pytest.mark.parametrize(
        "change",
        [
            {"low_confidence_below": 1.01},
            {"out_of_scope_below": -0.01},
            {"policy": "ask"},
            {"unknown": 1},
            {"out_of_scope_message": ""},
            {"out_of_scope_message": "   "},
        ],
    )
    def test_bad_values_and_unknown_keys_are_rejected(self, change: dict[str, Any]) -> None:
        with pytest.raises(InvalidAgentSpecError):
            scoped_spec(clarification={**CLARIFICATION, **change})

    def test_the_boundary_values_zero_and_one_are_accepted(self) -> None:
        block = {**CLARIFICATION, "low_confidence_below": 1.0, "out_of_scope_below": 0.0}
        assert scoped_spec(clarification=block).clarification.low_confidence_below == 1.0

    def test_knobs_with_policy_none_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            scoped_spec(clarification={"policy": "none", "low_confidence_below": 0.5})

        assert "low_confidence_below" in caplog.text
        assert "ignored" in caplog.text

    def test_a_consistent_block_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            scoped_spec()
            scoped_spec(clarification={"policy": "none"})

        assert caplog.text == ""

    def test_the_strategy_registry_builds_the_policy_over_the_same_router(self) -> None:
        probe = make_probe()
        spec = scoped_spec()
        router = build_router(spec, chat_models_for(), probe=probe, profiles=PROFILES)

        assert isinstance(build_clarification(spec, router), EvidenceClarification)
        none = scoped_spec(clarification={"policy": "none"})
        assert isinstance(build_clarification(none, router), NoClarification)

    def test_only_the_evidence_policy_may_refuse(self) -> None:
        router = build_router(
            scoped_spec(), chat_models_for(), probe=make_probe(), profiles=PROFILES
        )

        assert build_clarification(scoped_spec(), router).may_refuse is True
        assert build_clarification(
            scoped_spec(clarification={"policy": "none"}), router
        ).may_refuse is (False)
