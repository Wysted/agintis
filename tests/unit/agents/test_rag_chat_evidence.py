"""The rag-chat graph with `router: evidence`: no router model, ONE query embedding per attempt
(probe -> retrieval), re-routing after a rewrite, nothing heavy in the checkpointed state."""

from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langgraph.checkpoint.memory import InMemorySaver

from agintis.modules.agents.rag_chat import (
    AgentSpec,
    InvalidAgentSpecError,
    RagChatDeps,
    build_graph,
    build_router,
    parse_agent_spec,
    rag_chat_runner,
    roles_used,
)
from agintis.modules.agents_catalog.api import Sources, Step
from agintis.modules.knowledge.api import (
    IngestDocument,
    ProbeProfiles,
    RagProfile,
    RagProfileRegistry,
    RetrieveForTopic,
    SourceFile,
)
from agintis.platform.llm import ChatRole
from agintis.shared_kernel.ids import TenantId, ThreadId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    FakeChunker,
    FakeParser,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryIngestLock,
)
from tests.fakes.llm import ScriptedChatModel, chat_models_for
from tests.unit.agents.support import Script, make_input, make_model

TENANT = TenantId("acme")
EMBEDDING_SIZE = 16
ALPHA = RagProfile("alpha", 5, 100, "fake", "fake")
BETA = RagProfile("beta", 5, 100, "fake", "fake")
PROFILES = RagProfileRegistry([ALPHA, BETA])


class CountingEmbeddings(Embeddings):
    """Deterministic vectors; remembers the queries that were really embedded."""

    def __init__(self) -> None:
        self._inner = DeterministicFakeEmbedding(size=EMBEDDING_SIZE)
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._inner.embed_query(text)


class World:
    """Real probe and retrieval use cases over in-memory repositories, one embedding counter."""

    def __init__(self) -> None:
        self.embeddings = CountingEmbeddings()
        self.documents = InMemoryDocumentRepository()
        self.chunks = InMemoryChunkRepository(self.documents)
        self.probe = ProbeProfiles(self.chunks, self.embeddings)
        self.retrieval = RetrieveForTopic(self.documents, self.chunks, self.embeddings)

    async def seed(self) -> "World":
        ingest = IngestDocument(
            self.documents,
            self.chunks,
            {"fake": FakeParser()},
            {"fake": FakeChunker()},
            self.embeddings,
            FixedClock(),
            InMemoryIngestLock(),
        )
        for profile, uri, text in (
            (ALPHA, "file:///a", "alpha one\n\nalpha two\n\nalpha three\n\nalpha four"),
            (BETA, "file:///b", "beta one\n\nbeta two"),
        ):
            await ingest.execute(TENANT, profile, SourceFile(uri, "text/plain", text.encode()))
        self.embeddings.queries.clear()  # only the questions of the test count
        return self

    def deps(self, model: ScriptedChatModel) -> RagChatDeps:
        return RagChatDeps(chat_models_for(model), self.retrieval, probe=self.probe)


def _spec(**changes: Any) -> AgentSpec:
    return parse_agent_spec(
        {
            "id": "evidence-rag",
            "description": "Routes by evidence.",
            "topics": [
                {"name": "alpha", "description": "Alpha", "profile": "alpha", "filters": {}},
                {"name": "beta", "description": "Beta", "profile": "beta", "filters": {}},
            ],
            "router": "evidence",
            "sufficiency": "count",
            "primary_weight": 1,
            "max_rewrites": 0,
            **changes,
        },
        origin="test",
    )


def _steps(events: list[Any]) -> list[str]:
    return [e.node for e in events if isinstance(e, Step) and e.status == "started"]


async def _run(
    world: World,
    spec: AgentSpec,
    message: str,
    script: Script | None = None,
    saver: InMemorySaver | None = None,
    thread: ThreadId | None = None,
) -> tuple[list[Any], Script]:
    script = script or Script([])
    runner = rag_chat_runner(spec, PROFILES, world.deps(make_model(script, ["Answer [1]."])), saver)
    events = [e async for e in runner.astream(make_input(message, thread))]
    return events, script


def _topics_searched(events: list[Any]) -> set[str]:
    (sources,) = [e for e in events if isinstance(e, Sources)]
    return {s.source for s in sources.items}


class TestEvidenceRoutingEndToEnd:
    async def test_it_routes_by_the_corpus_without_a_router_model_call(self) -> None:
        world = await World().seed()

        events, script = await _run(world, _spec(evidence_margin=0.0), "alpha one")

        assert script.route_calls == 0  # no TopicRoute structured call
        assert _topics_searched(events) == {"file:///a"}  # the best topic only
        assert [e for e in _steps(events) if e == "topic"] == ["topic"]

    async def test_the_query_is_embedded_once_by_the_probe_and_reused_by_the_retrieval(
        self,
    ) -> None:
        world = await World().seed()

        # A huge margin searches both topics: the probe plus two retrievals, ONE embedding.
        events, _ = await _run(world, _spec(evidence_margin=1.0), "alpha one")

        assert _topics_searched(events) == {"file:///a", "file:///b"}
        assert world.embeddings.queries == ["alpha one"]

    async def test_the_evidence_router_needs_no_router_role(self) -> None:
        assert ChatRole.ROUTER not in roles_used(_spec())
        assert ChatRole.ROUTER in roles_used(_spec(router="llm"))

    async def test_a_spec_with_this_router_fails_to_build_without_the_probe(self) -> None:
        world = await World().seed()
        deps = RagChatDeps(chat_models_for(make_model(Script([]), [])), world.retrieval)

        with pytest.raises(ValueError, match="probe"):
            build_graph(_spec(), PROFILES, deps)


class TestMergeTieBreak:
    async def _first_source(self, spec: AgentSpec, replies: list[str]) -> tuple[str, Any]:
        world = await World().seed()
        model = make_model(Script([]), replies)
        runner = rag_chat_runner(spec, PROFILES, world.deps(model))
        events = [e async for e in runner.astream(make_input("alpha one"))]
        (sources,) = [e for e in events if isinstance(e, Sources)]
        return sources.items[0].source, model

    async def test_off_by_default_no_extra_model_call(self) -> None:
        # A huge margin searches both topics; the only plain call is the synthesis.
        first, model = await self._first_source(_spec(evidence_margin=1.0), ["Answer [1]."])

        assert first == "file:///a"
        assert len(model.seen) == 1

    async def test_the_router_model_reorders_near_tied_topics(self) -> None:
        spec = _spec(evidence_margin=1.0, merge_tiebreak="llm")

        first, model = await self._first_source(spec, ["B", "Answer [1]."])

        assert first == "file:///b"
        assert len(model.seen) == 2  # tie-break, then synthesis

    def test_the_knob_defaults_to_none_and_is_validated(self) -> None:
        assert _spec().merge_tiebreak == "none"
        with pytest.raises(InvalidAgentSpecError, match="merge_tiebreak"):
            _spec(merge_tiebreak="vibes")

    def test_a_router_binding_is_used_by_the_tie_break(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            spec = _spec(merge_tiebreak="llm", models={"router": "default"})
            _spec(models={"router": "default"})

        assert spec.uses_router_model
        assert [r.getMessage() for r in caplog.records] == [
            "agent 'evidence-rag': models.router is ignored (this agent makes no ROUTER model call)"
        ]

    def test_it_needs_the_router_role_and_two_topics(self) -> None:
        assert ChatRole.ROUTER in roles_used(_spec(merge_tiebreak="llm"))
        one_topic = {"topics": [{"name": "alpha", "description": "A", "profile": "alpha"}]}
        with pytest.raises(InvalidAgentSpecError, match="merge_tiebreak"):
            _spec(router="auto", merge_tiebreak="llm", **one_topic)


class TestRewriteLoopReroutes:
    async def test_each_attempt_probes_its_own_query_once(self) -> None:
        world = await World().seed()
        # min_sources 5 with alpha only (4 chunks): insufficient, so the question is rewritten.
        spec = _spec(evidence_margin=0.0, max_topics=1, min_sources=5, max_rewrites=1)
        script = Script([], rewrites=["alpha two"])

        events, _ = await _run(world, spec, "alpha one", script)

        steps = _steps(events)
        assert steps.count("rewrite") == 1
        assert steps.count("route") == 2  # re-routed with the rewritten query
        # One embedding per attempt: the probe's. The retrievals reused it.
        assert world.embeddings.queries == ["alpha one", "alpha two"]
        assert script.route_calls == 0


class TestCheckpointedState:
    @staticmethod
    def _heavy_lists(value: Any) -> list[Any]:
        """Every list of exactly `EMBEDDING_SIZE` floats anywhere in `value`."""
        found: list[Any] = []
        if isinstance(value, list | tuple):
            if len(value) == EMBEDDING_SIZE and all(isinstance(v, float) for v in value):
                found.append(value)
            for item in value:
                found.extend(TestCheckpointedState._heavy_lists(item))
        elif isinstance(value, dict):
            for item in value.values():
                found.extend(TestCheckpointedState._heavy_lists(item))
        return found

    async def test_a_thread_run_does_not_persist_the_query_vector(self) -> None:
        world = await World().seed()
        saver = InMemorySaver()
        spec = _spec(evidence_margin=1.0)
        thread = ThreadId("t-1")
        model = make_model(Script([]), ["first", "second"])
        runner = rag_chat_runner(spec, PROFILES, world.deps(model), saver)

        for message in ("alpha one", "beta one"):
            _ = [e async for e in runner.astream(make_input(message, thread))]
        graph = build_graph(spec, PROFILES, world.deps(model), saver)
        snapshot = await graph.aget_state({"configurable": {"thread_id": "t-1"}})

        assert not self._heavy_lists(snapshot.values)
        routed = snapshot.values["routed_topics"]
        # What IS kept is small: the raw similarities behind each confidence (<= probe_k).
        assert all(0 < len(r["similarities"]) <= spec.probe_k for r in routed)
        # A new turn is a new run: it embeds again instead of finding a stale memo.
        assert world.embeddings.queries == ["alpha one", "beta one"]
        checkpoints = [c async for c in saver.alist({"configurable": {"thread_id": "t-1"}})]
        assert {c.config["configurable"]["checkpoint_ns"] for c in checkpoints} == {""}  # parent

    async def test_routed_topics_carry_the_evidence_scores(self) -> None:
        world = await World().seed()
        saver = InMemorySaver()
        spec = _spec(evidence_margin=1.0)
        model = make_model(Script([]), ["answer"])
        runner = rag_chat_runner(spec, PROFILES, world.deps(model), saver)

        _ = [e async for e in runner.astream(make_input("alpha one", ThreadId("t-1")))]
        graph = build_graph(spec, PROFILES, world.deps(model), saver)
        snapshot = await graph.aget_state({"configurable": {"thread_id": "t-1"}})

        routed = snapshot.values["routed_topics"]
        assert [r["topic"] for r in routed] == ["alpha", "beta"]
        assert routed[0]["confidence"] == pytest.approx(1.0)  # the chunk with that very text
        assert routed[0]["confidence"] > routed[1]["confidence"]


class TestBiasAndKnobWarnings:
    def test_the_topic_bias_is_bounded_and_defaults_to_zero(self) -> None:
        spec = _spec()
        assert [t.evidence_bias for t in spec.topics] == [0.0, 0.0]
        topic = {"name": "a", "description": "d", "profile": "alpha", "evidence_bias": 0.5}
        with pytest.raises(InvalidAgentSpecError):
            _spec(topics=[topic, {"name": "b", "description": "d", "profile": "beta"}])

    def test_the_bias_reaches_the_router(self) -> None:
        world = World()
        spec = _spec(
            topics=[
                {"name": "alpha", "description": "A", "profile": "alpha", "evidence_bias": 0.05},
                {"name": "beta", "description": "B", "profile": "beta"},
            ]
        )

        router = build_router(spec, chat_models_for(), probe=world.probe, profiles=PROFILES)

        assert [t.bias for t in router._topics] == [0.05, 0.0]  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "knob", [{"evidence_margin": 0.1}, {"evidence_aggregate": "max"}, {"probe_k": 2}]
    )
    def test_evidence_knobs_with_another_router_warn(
        self, knob: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            _spec(router="llm", **knob)

        assert "ignored" in caplog.text
        assert next(iter(knob)) in caplog.text

    def test_a_topic_bias_with_another_router_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        topics = [
            {"name": "alpha", "description": "A", "profile": "alpha", "evidence_bias": 0.0},
            {"name": "beta", "description": "B", "profile": "beta"},
        ]
        with caplog.at_level("WARNING"):
            _spec(router="auto", topics=topics)

        assert "evidence_bias" in caplog.text

    def test_the_secondary_confidence_with_the_evidence_router_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            _spec(secondary_min_confidence=0.6)

        assert "secondary_min_confidence" in caplog.text
        assert "evidence_margin" in caplog.text

    def test_a_consistent_spec_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            _spec(evidence_margin=0.02, probe_k=3)
            _spec(router="llm", secondary_min_confidence=0.6)

        assert caplog.text == ""


class TestSpecValidation:
    def test_defaults_keep_the_llm_path_untouched(self) -> None:
        spec = parse_agent_spec(
            {
                "id": "x",
                "description": "d",
                "topics": [
                    {"name": "a", "description": "d", "profile": "alpha"},
                    {"name": "b", "description": "d", "profile": "beta"},
                ],
            },
            origin="t",
        )
        assert spec.router == "auto"
        assert spec.uses_llm_router
        assert not spec.uses_evidence_router
        assert (spec.evidence_aggregate, spec.evidence_margin, spec.probe_k) == ("max", 0.02, 3)

    def test_the_evidence_router_needs_two_topics(self) -> None:
        with pytest.raises(InvalidAgentSpecError, match="'evidence' needs at least two topics"):
            parse_agent_spec(
                {
                    "id": "x",
                    "description": "d",
                    "router": "evidence",
                    "topics": [{"name": "a", "description": "d", "profile": "alpha"}],
                },
                origin="t",
            )

    def test_evidence_is_not_an_llm_router_and_fans_out_up_to_max_topics(self) -> None:
        spec = _spec(max_topics=2)
        assert spec.uses_evidence_router
        assert not spec.uses_llm_router
        assert spec.fan_out == 2

    @pytest.mark.parametrize(
        "change",
        [
            {"evidence_margin": -0.01},
            {"evidence_margin": 1.5},
            {"evidence_aggregate": "median"},
            {"probe_k": 0},
            {"probe_k": 21},
            {"router": "magic"},
        ],
    )
    def test_the_knobs_are_bounded(self, change: dict[str, Any]) -> None:
        with pytest.raises(InvalidAgentSpecError):
            _spec(**change)

    @pytest.mark.parametrize("aggregate", ["max", "mean_top_k"])
    def test_both_aggregates_are_accepted(self, aggregate: str) -> None:
        assert _spec(evidence_aggregate=aggregate).evidence_aggregate == aggregate
