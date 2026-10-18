"""The rag-chat graph end to end with fakes: runner, RunAgent, threads, HTTP and the container."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.exc import InterfaceError, OperationalError

from agintis.agent_kit.nodes.prompts import DEFAULT_UNCITED_SYNTHESIZE_PROMPT
from agintis.bootstrap.adapters import ChainedErrorClassifier
from agintis.bootstrap.container import Container
from agintis.main import create_app
from agintis.modules.agents.rag_chat import (
    AgentSpec,
    InvalidAgentSpecError,
    RagChatDeps,
    build_graph,
    parse_agent_spec,
    rag_chat_runner,
)
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    AgentUnavailableError,
    RunAgent,
    RunCompleted,
    RunFailed,
    RunInput,
    RunStarted,
    SourceRef,
    Sources,
    Step,
    Token,
)
from agintis.modules.agents_catalog.http import get_services
from agintis.modules.knowledge.api import MetadataField, RagProfile, RagProfileRegistry
from agintis.modules.knowledge.infrastructure.errors import RetrievalErrorClassifier
from agintis.modules.knowledge.infrastructure.tei import TeiUnavailableError
from agintis.platform.llm import ChatModels, ChatRole, LlmConfig, ProviderErrorClassifier
from agintis.shared_kernel.criteria import Filter
from agintis.shared_kernel.ids import RunId, TenantId, ThreadId
from tests.fakes.llm import RaisingModel, ScriptedChatModel, chat_models_for, rate_limit_error
from tests.fakes.retrieval import FakeRetrieval, make_hit
from tests.support.settings import make_settings
from tests.unit.test_agents_http import _frames, services_for

TENANT = TenantId("acme")
PROFILE = RagProfile(
    "docs",
    3,
    100,
    "simple",
    "structured",
    metadata_fields=(MetadataField("kind", "str", filterable=True),),
)
PROFILES = RagProfileRegistry([PROFILE])
SPEC = parse_agent_spec(
    {
        "id": "docs-rag",
        "description": "Answers from the docs.",
        "topics": [{"name": "docs", "description": "d", "profile": "docs"}],
    },
    origin="test",
)
HIT = make_hit("The deadline is 24 months.", uri="file:///a.pdf", title="Doc A", page=2)


def _analysis(query: str = "deadline?") -> dict[str, Any]:
    return {
        "standalone_query": query,
        "is_clear": True,
        "confidence": 0.9,
        "request_kind": "domain",
    }


def _filters(kind: str | None = None) -> dict[str, Any]:
    return {"kind": kind}


def _model(*turns: tuple[str, str | None, str]) -> ScriptedChatModel:
    """Per turn: (standalone query, extracted kind, final answer)."""
    return ScriptedChatModel(
        structured=[item for q, kind, _ in turns for item in (_analysis(q), _filters(kind))],
        replies=[answer for _, _, answer in turns],
    )


def _runner(
    model: BaseChatModel,
    retrieval: FakeRetrieval | None = None,
    saver: InMemorySaver | None = None,
    spec: AgentSpec = SPEC,
) -> Any:
    deps = RagChatDeps(chat_models_for(model), retrieval or FakeRetrieval((HIT,)))
    return rag_chat_runner(spec, PROFILES, deps, saver)


def _input(message: str = "What is the deadline?", thread: ThreadId | None = None) -> RunInput:
    return RunInput(TENANT, message, RunId("0123456789abcdef0123456789abcdef"), thread)


async def _events(runner: Any, run_input: RunInput) -> list[Any]:
    return [e async for e in runner.astream(run_input)]


class TestStatelessRun:
    async def test_streams_steps_sources_tokens_and_done_through_run_agent(self) -> None:
        model = _model(("What is the deadline?", "memo", "It is 24 months [1]."))
        retrieval = FakeRetrieval((HIT,))
        registry = AgentRegistry((_runner(model, retrieval),))
        run_agent = RunAgent(registry, ProviderErrorClassifier(), id_generator=lambda: "run-1")

        events = [e async for e in run_agent.start("docs-rag", TENANT, "What is the deadline?")]

        assert events[0] == RunStarted(RunId("run-1"), "docs-rag")
        steps = [e for e in events if isinstance(e, Step)]
        assert [s.node for s in steps if s.status == "started"] == [
            "analyze",
            "route",
            "topic",
            "topic/extract_filters",
            "topic/retrieve",
            "merge",
            "synthesize",
        ]
        assert {s.node for s in steps if s.status == "finished"} == {s.node for s in steps}
        assert [e for e in events if isinstance(e, Sources)] == [
            Sources((SourceRef("file:///a.pdf", "Doc A", "p. 2"),))
        ]
        # The answer is streamed; the analyzer / filter JSON is not.
        tokens = [e.text for e in events if isinstance(e, Token)]
        assert "".join(tokens) == "It is 24 months [1]."
        assert len(tokens) > 1
        assert isinstance(events[-1], RunCompleted)
        assert events[-1].finish_reason == "stop"
        assert events[-1].usage is not None  # every call's usage is aggregated
        # Ordering: sources arrive before the first token.
        kinds = [type(e) for e in events]
        assert kinds.index(Sources) < kinds.index(Token)

    async def test_internal_llm_output_never_reaches_the_client_as_tokens(self) -> None:
        model = _model(("What is the deadline?", "memo", "Fine."))

        events = await _events(_runner(model), _input())

        text = "".join(e.text for e in events if isinstance(e, Token))
        assert text == "Fine."
        assert "standalone_query" not in text
        assert "memo" not in text

    async def test_the_extracted_filter_reaches_retrieval_with_the_tenant(self) -> None:
        retrieval = FakeRetrieval((HIT,))
        model = _model(("deadline for memos?", "memo", "ok"))

        await _events(_runner(model, retrieval), _input())

        (call,) = retrieval.calls
        assert call.tenant_id == TENANT
        assert call.query == "deadline for memos?"
        assert call.extra_filter == Filter.eq("kind", "memo")

    async def test_the_synthesis_prompt_carries_the_numbered_sources(self) -> None:
        model = _model(("q", None, "ok"))

        await _events(_runner(model), _input("the question"))

        system, sources, human = model.seen[0]
        assert "deadline is 24 months" not in str(system.content)  # never in the system role
        assert sources.type == "human"
        assert "[1] Doc A (p. 2)\nThe deadline is 24 months." in str(sources.content)
        assert (human.type, human.content) == ("human", "the question")

    async def test_prompt_overrides_from_the_spec_are_used(self) -> None:
        spec = parse_agent_spec(
            {
                **SPEC.model_dump(),
                "prompts": {"analyze": "A-PROMPT", "synthesize": "S-PROMPT"},
            },
            origin="test",
        )
        model = _model(("q", None, "ok"))

        await _events(_runner(model, spec=spec), _input())

        assert model.structured_seen[0][0].content == "A-PROMPT"
        assert str(model.seen[0][0].content).startswith("S-PROMPT")

    async def test_citations_none_uses_the_uncited_prompt_and_still_emits_sources(self) -> None:
        spec = parse_agent_spec({**SPEC.model_dump(), "citations": "none"}, origin="test")
        model = _model(("q", None, "It is 24 months."))

        events = await _events(_runner(model, spec=spec), _input())

        assert model.seen[0][0].content == DEFAULT_UNCITED_SYNTHESIZE_PROMPT
        assert [e for e in events if isinstance(e, Sources)] == [
            Sources((SourceRef("file:///a.pdf", "Doc A", "p. 2"),))
        ]

    async def test_no_hits_still_answers(self) -> None:
        model = _model(("q", None, "I do not know."))
        no_rewrite = parse_agent_spec({**SPEC.model_dump(), "max_rewrites": 0}, origin="test")

        events = await _events(_runner(model, FakeRetrieval(), spec=no_rewrite), _input())

        assert not [e for e in events if isinstance(e, Sources)]
        assert "".join(e.text for e in events if isinstance(e, Token)) == "I do not know."

    def test_the_agent_is_stateless_without_a_saver_and_checks_the_llm(self) -> None:
        runner = _runner(_model())
        assert runner.supports_threads is False
        assert runner.agent_id == "docs-rag"
        runner.ensure_available()

        deps = RagChatDeps(ChatModels(None), FakeRetrieval())
        with pytest.raises(AgentUnavailableError) as info:
            rag_chat_runner(SPEC, PROFILES, deps).ensure_available()
        assert info.value.code == "llm_not_configured"

    def test_a_misconfigured_role_is_reported_as_unavailable(self) -> None:
        def failing(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("secret detail")

        models = ChatModels(
            LlmConfig(provider="openai", models={r: "m" for r in ChatRole}), builder=failing
        )
        with pytest.raises(AgentUnavailableError) as info:
            rag_chat_runner(SPEC, PROFILES, RagChatDeps(models, FakeRetrieval())).ensure_available()
        assert info.value.code == "llm_misconfigured"
        assert "secret" not in str(info.value)


class TestThreads:
    async def test_memory_is_kept_across_two_turns_and_stale_state_does_not_leak(self) -> None:
        saver = InMemorySaver()
        retrieval = FakeRetrieval((HIT,))
        model = _model(
            ("What is the deadline for memos?", "memo", "24 months [1]."),
            ("And for letters?", None, "Also 24 months [1]."),
        )
        runner = _runner(model, retrieval, saver)
        thread = ThreadId("t-1")

        await _events(runner, _input("What is the deadline for memos?", thread))
        await _events(runner, _input("And for letters?", thread))

        # Turn 2: the analyzer saw the whole conversation, the synthesis too.
        analyzer_second = model.structured_seen[2]
        assert [m.content for m in analyzer_second[1:]] == [
            "What is the deadline for memos?",
            "24 months [1].",
            "And for letters?",
        ]
        second = model.seen[1]
        assert [m.content for m in second[1:3]] == [
            "What is the deadline for memos?",
            "24 months.",  # numbering restarts every turn: the old [1] is stripped
        ]
        assert "<sources>" in str(second[3].content)
        assert second[4].content == "And for letters?"
        # Turn 1 filtered by kind; turn 2 extracted nothing and must NOT reuse that filter.
        assert [c.extra_filter for c in retrieval.calls] == [Filter.eq("kind", "memo"), None]
        assert [c.query for c in retrieval.calls] == [
            "What is the deadline for memos?",
            "And for letters?",
        ]
        deps = RagChatDeps(chat_models_for(model), retrieval)
        reader = build_graph(SPEC, PROFILES, deps, saver)  # same saver: reads the thread's state
        state = await reader.aget_state({"configurable": {"thread_id": thread}})
        assert [type(m) for m in state.values["messages"]] == [
            HumanMessage,
            AIMessage,
            HumanMessage,
            AIMessage,
        ]
        assert state.values["answer"] == "Also 24 months [1]."
        assert "filters" not in state.values  # private to the topic subgraph
        assert state.values["routed_topics"] == [{"topic": "docs", "confidence": 1.0}]
        assert [r["filter_level"] for r in state.values["topic_results"]] == ["none"]

    async def test_thread_runs_need_the_checkpointed_graph(self) -> None:
        runner = _runner(_model())
        with pytest.raises(AgentUnavailableError):
            await _events(runner, _input(thread=ThreadId("t")))

    def test_supports_threads_with_a_saver(self) -> None:
        assert _runner(_model(), saver=InMemorySaver()).supports_threads is True


@asynccontextmanager
async def _client(runner: Any) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(make_settings())
    app.dependency_overrides[get_services] = lambda: services_for(runner)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
    ):
        yield c


async def test_http_lists_the_rag_agent_and_streams_a_run() -> None:
    runner = _runner(_model(("What is the deadline?", "memo", "24 months [1].")))
    async with _client(runner) as client:
        listing = await client.get("/v1/agents")
        response = await client.post(
            "/v1/runs/stream",
            json={"agent_id": "docs-rag", "message": "What is the deadline?"},
            headers={"X-Tenant-Id": "acme"},
        )

    assert listing.json() == [{"id": "docs-rag", "description": "Answers from the docs."}]
    assert response.status_code == HTTPStatus.OK
    frames = _frames(response.text)
    assert [f["event"] for f in frames if f["event"] != "step"][:2] == ["run_started", "sources"]
    assert frames[-1]["event"] == "done"
    sources = next(f for f in frames if f["event"] == "sources")
    assert sources["data"]["items"][0]["source"] == "file:///a.pdf"
    tokens = "".join(f["data"]["text"] for f in frames if f["event"] == "token")
    assert tokens == "24 months [1]."
    assert "standalone_query" not in response.text


class TestContainerWiring:
    def _settings(self, tmp_path: Path, profiles: str, agents: str, spec: str) -> Any:
        profiles_dir, agents_dir = tmp_path / "profiles", tmp_path / "agents"
        profiles_dir.mkdir()
        agents_dir.mkdir()
        (profiles_dir / "docs.yaml").write_text(profiles, encoding="utf-8")
        (agents_dir / "docs.yaml").write_text(spec, encoding="utf-8")
        return make_settings(knowledge={"profiles_dir": profiles_dir}, agents={"dir": agents_dir})

    PROFILE_YAML = """\
key: docs
schema_version: 1
top_k: 3
chunk_max_tokens: 200
parser_key: simple
chunker_key: structured
metadata_fields:
  - {name: kind, type: str, source: 'const:memo', filterable: true}
"""
    SPEC_YAML = """\
id: docs-rag
description: Docs agent.
topics: [{name: docs, description: d, profile: docs, filters: {kind: "The kind"}}]
"""

    def test_each_spec_is_registered_next_to_direct_chat(self, tmp_path: Path) -> None:
        container = Container.build(self._settings(tmp_path, self.PROFILE_YAML, "", self.SPEC_YAML))

        registry = container.agents_catalog.registry
        assert [r.agent_id for r in registry.list()] == ["direct-chat", "docs-rag"]

    def test_without_an_agents_dir_only_direct_chat_exists(self) -> None:
        container = Container.build(make_settings())
        assert [r.agent_id for r in container.agents_catalog.registry.list()] == ["direct-chat"]

    def test_an_unknown_profile_aborts_the_boot(self, tmp_path: Path) -> None:
        spec = self.SPEC_YAML.replace("profile: docs", "profile: nope")
        with pytest.raises(InvalidAgentSpecError, match="unknown profile 'nope'"):
            Container.build(self._settings(tmp_path, self.PROFILE_YAML, "", spec))

    def test_a_filter_the_profile_does_not_offer_aborts_the_boot(self, tmp_path: Path) -> None:
        spec = self.SPEC_YAML.replace("{kind:", "{missing:")
        with pytest.raises(InvalidAgentSpecError, match="missing"):
            Container.build(self._settings(tmp_path, self.PROFILE_YAML, "", spec))

    def test_an_agent_id_clashing_with_direct_chat_aborts_the_boot(self, tmp_path: Path) -> None:
        spec = self.SPEC_YAML.replace("docs-rag", "direct-chat")
        with pytest.raises(ValueError, match="already registered"):
            Container.build(self._settings(tmp_path, self.PROFILE_YAML, "", spec))


class TestFailures:
    """Run-level behaviour of the hint nodes' failures and of classified outages."""

    def _run_agent(self, runner: Any) -> RunAgent:
        classifier = ChainedErrorClassifier(ProviderErrorClassifier(), RetrievalErrorClassifier())
        return RunAgent(AgentRegistry((runner,)), classifier, id_generator=lambda: "run-1")

    async def _failure(self, runner: Any) -> list[Any]:
        return [e async for e in self._run_agent(runner).start("docs-rag", TENANT, "q")]

    async def test_malformed_analyzer_output_degrades_and_the_run_still_answers(self) -> None:
        model = ScriptedChatModel(
            structured=["{not json", _filters("memo")],
            replies=["Still answered."],
        )
        retrieval = FakeRetrieval((HIT,))

        events = await self._failure(_runner(model, retrieval))

        assert "".join(e.text for e in events if isinstance(e, Token)) == "Still answered."
        assert not [e for e in events if isinstance(e, RunFailed)]
        assert retrieval.calls[0].query == "q"  # the raw question stood in for the rewrite

    async def test_a_provider_rate_limit_ends_the_run_with_one_llm_rate_limited_error(self) -> None:
        runner = _runner(RaisingModel(error=rate_limit_error()))

        events = await self._failure(runner)

        failures = [e for e in events if isinstance(e, RunFailed)]
        assert [(f.code, f.retryable) for f in failures] == [("llm_rate_limited", True)]
        assert events[-1] is failures[0]
        assert not [e for e in events if isinstance(e, Token | RunCompleted)]

    @pytest.mark.parametrize(
        "error",
        [
            TeiUnavailableError("embeddings down"),
            OperationalError(
                "SELECT 1", {}, Exception("canceling statement due to statement timeout")
            ),
            InterfaceError("SELECT 1", {}, Exception("connection is closed")),
        ],
    )
    async def test_retrieval_outages_are_one_retryable_retrieval_unavailable_error(
        self, error: Exception
    ) -> None:
        def explode(_call: Any) -> Any:
            raise error

        model = _model(("q", None, "never streamed"))
        events = await self._failure(_runner(model, FakeRetrieval(respond=explode)))

        failures = [e for e in events if isinstance(e, RunFailed)]
        assert [(f.code, f.retryable) for f in failures] == [("retrieval_unavailable", True)]
        assert "statement" not in failures[0].message  # fixed text, never exception text
        assert events[-1] is failures[0]
        assert not [e for e in events if isinstance(e, Token)]

    async def test_other_retrieval_exceptions_stay_internal(self) -> None:
        def explode(_call: Any) -> Any:
            raise ValueError("a bug")

        events = await self._failure(
            _runner(_model(("q", None, "x")), FakeRetrieval(respond=explode))
        )

        assert [e.code for e in events if isinstance(e, RunFailed)] == ["internal"]
