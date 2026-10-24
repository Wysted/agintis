import asyncio
from contextlib import aclosing
from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from agintis.agent_kit.runtime import (
    DEFAULT_RECURSION_LIMIT,
    STREAM_TOKENS_TAG,
    AgentContext,
    LangGraphRunner,
    emit,
    finish_reason_of,
    streamed,
)
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    AgentUnavailableError,
    Clarification,
    RunAgent,
    RunCompleted,
    RunFailed,
    RunInput,
    RunStarted,
    SourceRef,
    Sources,
    Step,
    Token,
    TokenUsage,
)
from agintis.shared_kernel.errors import FailureInfo
from agintis.shared_kernel.ids import RunId, TenantId, ThreadId
from tests.fakes.graphs import (
    BlockingProbe,
    Graph,
    StreamingModel,
    blocking_graph,
    config_probe_graph,
    emitting_graph,
    failing_graph,
    fan_out_graph,
    interrupting_graph,
    leaky_parent_graph,
    llm_graph,
    state_probe_graph,
    subgraph_graph,
    threaded,
)

RUN_ID = "0123456789abcdef0123456789abcdef"


def _input(run_id: str = RUN_ID, message: str = "hi") -> RunInput:
    return RunInput(TenantId("acme"), message, RunId(run_id))


def _runner(graph: Graph, **kwargs: Any) -> LangGraphRunner:
    return LangGraphRunner("toy", "A toy agent", graph, **kwargs)


async def _events(runner: LangGraphRunner, run_input: RunInput | None = None) -> list[Any]:
    return [e async for e in runner.astream(run_input or _input())]


def _tokens(events: list[Any]) -> list[Token]:
    return [e for e in events if isinstance(e, Token)]


def _steps(events: list[Any]) -> list[Step]:
    return [e for e in events if isinstance(e, Step)]


async def test_tagged_llm_streams_tokens_in_order() -> None:
    graph = llm_graph(StreamingModel(pieces=["Hel", "lo", "!"]))

    events = await _events(_runner(graph))

    assert _tokens(events) == [Token("Hel"), Token("lo"), Token("!")]


async def test_untagged_llm_gives_no_tokens_but_usage_is_aggregated() -> None:
    internal = StreamingModel(pieces=['{"route": "x"}'], input_tokens=10, output_tokens=1)
    answer = StreamingModel(pieces=["ok"], input_tokens=4, output_tokens=2, finish_reason="length")

    events = await _events(_runner(llm_graph(answer, internal)))

    assert _tokens(events) == [Token("ok")]
    assert events[-1] == RunCompleted("length", TokenUsage(14, 3, 17))


async def test_cached_input_tokens_are_summed_over_calls_and_default_to_zero() -> None:
    internal = StreamingModel(pieces=["x"], input_tokens=10, output_tokens=1, cached_tokens=6)
    answer = StreamingModel(pieces=["ok"], input_tokens=4, output_tokens=2, cached_tokens=3)

    events = await _events(_runner(llm_graph(answer, internal)))
    assert events[-1] == RunCompleted("stop", TokenUsage(14, 3, 17, cached_input_tokens=9))

    plain = await _events(_runner(llm_graph(StreamingModel(pieces=["ok"]))))
    assert plain[-1] == RunCompleted("stop", TokenUsage(4, 2, 6, cached_input_tokens=0))

    # one call reports details, the other does not: missing counts as 0
    mixed = await _events(_runner(llm_graph(answer, StreamingModel(pieces=["x"]))))
    assert mixed[-1].usage.cached_input_tokens == 3


async def test_finish_reason_comes_from_tagged_call_not_internal_one() -> None:
    internal = StreamingModel(pieces=["x"], finish_reason="tool_calls")
    answer = StreamingModel(pieces=["ok"], finish_reason="stop")

    events = await _events(_runner(llm_graph(answer, internal)))

    assert isinstance(events[-1], RunCompleted)
    assert events[-1].finish_reason == "stop"


async def test_default_finish_reason_and_no_usage_without_llm() -> None:
    events = await _events(_runner(emitting_graph()))

    assert events[-1] == RunCompleted("stop", None)


async def test_custom_events_pass_through_and_unknown_payloads_are_ignored() -> None:
    events = await _events(_runner(emitting_graph()))

    assert Step("custom-step", "started") in events
    assert Sources((SourceRef("doc.md", "Doc"),)) in events
    assert Clarification("Which one?", ("a", "b")) in events
    allowed = (Token, Step, Sources, Clarification, RunCompleted)
    assert all(isinstance(e, allowed) for e in events)


async def test_steps_cover_node_start_and_finish() -> None:
    events = await _events(_runner(llm_graph(StreamingModel(pieces=["a"]))))

    assert _steps(events) == [Step("answer", "started"), Step("answer", "finished")]


async def test_subgraph_tokens_and_namespaced_steps() -> None:
    events = await _events(_runner(subgraph_graph(StreamingModel(pieces=["su", "b"]))))

    assert _tokens(events) == [Token("su"), Token("b")]
    steps = _steps(events)
    assert Step("topic", "started") in steps
    assert Step("topic/inner_answer", "started") in steps
    assert Step("topic/inner_answer", "finished") in steps
    assert steps.index(Step("topic", "started")) < steps.index(
        Step("topic/inner_answer", "started")
    )


async def test_node_exception_propagates_unchanged() -> None:
    runner = _runner(failing_graph(ValueError("boom")))

    with pytest.raises(ValueError, match="boom"):
        await _events(runner)


async def test_run_agent_turns_node_failure_into_single_error_event() -> None:
    class Classifier:
        def classify(self, exc: BaseException) -> FailureInfo:
            return FailureInfo("internal", "Internal error while running the agent.", False)

    registry = AgentRegistry((_runner(failing_graph(RuntimeError("secret detail"))),))
    use_case = RunAgent(registry, Classifier())

    events = [e async for e in use_case.start("toy", TenantId("acme"), "hi")]

    assert isinstance(events[0], RunStarted)
    failures = [e for e in events if isinstance(e, RunFailed)]
    assert len(failures) == 1
    assert "secret detail" not in failures[0].message
    assert not any(isinstance(e, RunCompleted) for e in events)


async def test_closing_the_stream_cancels_the_graph_run() -> None:
    probe = BlockingProbe()
    runner = _runner(blocking_graph(probe))

    async def consume() -> None:
        async with aclosing(runner.astream(_input())) as stream:
            async for _ in stream:
                pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(probe.started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(probe.finalized.wait(), 5)
    assert probe.cancelled.is_set()


async def test_aclose_while_suspended_closes_the_graph_run() -> None:
    probe = BlockingProbe()
    stream = _runner(blocking_graph(probe)).astream(_input())

    async for event in stream:
        if event == Step("blocking", "started"):
            break  # suspended at a yield, like a client that went away
    await stream.aclose()

    await asyncio.wait_for(probe.finalized.wait(), 5)
    assert probe.cancelled.is_set()


async def test_default_input_mapper_builds_messages() -> None:
    seen: list[list[BaseMessage]] = []

    await _events(_runner(state_probe_graph(seen)), _input(message="hola"))

    assert seen == [[HumanMessage(content="hola", id=seen[0][0].id)]]


async def test_input_mapper_strategy_override() -> None:
    seen: list[list[BaseMessage]] = []
    runner = _runner(
        state_probe_graph(seen),
        input_mapper=lambda run_input: {"messages": [HumanMessage(run_input.message.upper())]},
    )

    await _events(runner, _input(message="hola"))

    assert [m.content for m in seen[0]] == ["HOLA"]


async def test_config_has_recursion_limit_and_tracer_run_id_but_no_tenant() -> None:
    seen: list[RunnableConfig] = []
    runtimes: list[Runtime[AgentContext]] = []

    await _events(_runner(config_probe_graph(seen, runtimes)))
    await _events(_runner(config_probe_graph(seen, runtimes), recursion_limit=7))

    first, second = seen
    assert first["recursion_limit"] == DEFAULT_RECURSION_LIMIT
    assert second["recursion_limit"] == 7
    execution_info = runtimes[0].execution_info
    assert execution_info is not None
    assert execution_info.run_id == str(UUID(RUN_ID))
    assert "tenant_id" not in first["configurable"]
    assert "run_id" not in first["configurable"]


async def test_non_uuid_run_id_is_tolerated() -> None:
    seen: list[RunnableConfig] = []
    runtimes: list[Runtime[AgentContext]] = []

    events = await _events(_runner(config_probe_graph(seen, runtimes)), _input("not-a-uuid"))

    assert isinstance(events[-1], RunCompleted)
    assert runtimes[0].context.run_id == "not-a-uuid"


async def test_nodes_read_the_typed_context() -> None:
    seen: list[RunnableConfig] = []
    runtimes: list[Runtime[AgentContext]] = []

    await _events(_runner(config_probe_graph(seen, runtimes)))

    assert [r.context for r in runtimes] == [AgentContext(TenantId("acme"), RunId(RUN_ID))]


async def test_context_factory_strategy_override() -> None:
    seen: list[RunnableConfig] = []
    runtimes: list[Runtime[AgentContext]] = []
    runner = _runner(
        config_probe_graph(seen, runtimes),
        context_factory=lambda ri: AgentContext(TenantId("other"), ri.run_id),
    )

    await _events(runner)

    assert runtimes[0].context.tenant_id == "other"


def test_ensure_available_delegates_to_the_injected_check() -> None:
    def unavailable() -> None:
        raise AgentUnavailableError("down", code="down")

    _runner(emitting_graph()).ensure_available()
    with pytest.raises(AgentUnavailableError):
        _runner(emitting_graph(), availability=unavailable).ensure_available()


def test_emit_writes_through_the_runtime_stream_writer() -> None:
    written: list[object] = []
    runtime: Runtime[Any] = Runtime(stream_writer=written.append)

    emit(runtime, Token("x"))

    assert written == [Token("x")]


async def test_non_streaming_tagged_model_gives_one_token_and_usage() -> None:
    model = StreamingModel(pieces=["full ", "answer"], disable_streaming=True, finish_reason="stop")

    events = await _events(_runner(llm_graph(model)))

    assert _tokens(events) == [Token("full answer")]
    assert events[-1] == RunCompleted("stop", TokenUsage(4, 2, 6))


async def test_non_streaming_untagged_model_counts_usage_only() -> None:
    internal = StreamingModel(
        pieces=["x"], disable_streaming=True, input_tokens=10, output_tokens=1
    )
    answer = StreamingModel(pieces=["ok"])

    events = await _events(_runner(llm_graph(answer, internal)))

    assert _tokens(events) == [Token("ok")]
    assert events[-1] == RunCompleted("stop", TokenUsage(14, 3, 17))


async def test_node_returned_message_is_not_counted_or_streamed() -> None:
    # The state message of a streamed call is deduped by id: usage is counted once.
    events = await _events(_runner(llm_graph(StreamingModel(pieces=["ok"]))))

    assert events[-1] == RunCompleted("stop", TokenUsage(4, 2, 6))


async def test_finish_reason_and_usage_in_separate_chunks() -> None:
    model = StreamingModel(pieces=["a"], separate_usage_chunk=True, finish_reason="length")

    events = await _events(_runner(llm_graph(model)))

    assert events[-1] == RunCompleted("length", TokenUsage(4, 2, 6))


def test_finish_reason_is_normalized_across_providers() -> None:
    def of(**metadata: str) -> str | None:
        return finish_reason_of(AIMessage(content="", response_metadata=metadata))

    assert of(stop_reason="end_turn") == "stop"
    assert of(stop_reason="max_tokens") == "length"
    assert of(stop_reason="stop_sequence") == "stop"
    assert of(done_reason="stop") == "stop"
    assert of(finish_reason="content_filter") == "content_filter"
    assert of() is None


async def test_tag_on_a_parent_runnable_leaks_to_inner_llm_calls() -> None:
    # Pins current langchain behaviour (tags are inherited by children): apply `streamed` to
    # leaf chat models only. If this starts failing, the inheritance rule changed upstream.
    events = await _events(_runner(leaky_parent_graph(StreamingModel(pieces=["SECRET"]))))

    assert _tokens(events) == [Token("SECRET")]


async def test_bind_tools_after_streamed_drops_the_tag_and_before_keeps_it() -> None:
    model = StreamingModel(pieces=["text"])
    tools: list[Any] = [{"name": "t"}]

    def graph(runnable: Any) -> Graph:
        return llm_graph(runnable)

    dropped = streamed(model).bind_tools(tools)  # type: ignore[attr-defined]
    kept = streamed(model.bind_tools(tools))  # type: ignore[arg-type]

    assert _tokens(await _events(_runner(graph(dropped)))) == []
    assert _tokens(await _events(_runner(graph(kept)))) == [Token("text")]


def test_streamed_applies_the_tag() -> None:
    bound = streamed(StreamingModel(pieces=[]))

    assert bound.config["tags"] == [STREAM_TOKENS_TAG]  # type: ignore[attr-defined]


async def test_interrupted_task_gets_no_finished_step() -> None:
    toy = threaded(interrupting_graph)

    events = await _events(
        toy.runner, RunInput(TenantId("acme"), "hi", RunId(RUN_ID), ThreadId("t1"))
    )

    assert _steps(events) == [Step("ask", "started")]
    assert isinstance(events[-1], RunCompleted)


async def test_failed_node_gets_no_finished_step() -> None:
    runner = _runner(failing_graph(ValueError("boom")))
    steps: list[Step] = []

    async def consume() -> None:
        async for event in runner.astream(_input()):
            if isinstance(event, Step):
                steps.append(event)

    with pytest.raises(ValueError, match="boom"):
        await consume()

    assert steps == [Step("boom", "started")]


async def test_send_fan_out_repeats_step_names_and_completes() -> None:
    runner = _runner(
        fan_out_graph(), input_mapper=lambda ri: {"messages": [], "topics": [], "topic": ""}
    )

    events = await _events(runner)

    steps = _steps(events)
    assert steps.count(Step("topic", "started")) == 2
    assert steps.count(Step("topic", "finished")) == 2
    assert isinstance(events[-1], RunCompleted)
