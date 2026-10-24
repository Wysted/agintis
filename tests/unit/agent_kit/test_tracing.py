"""Tracing end to end without network: a fake chat model through `TracedRunner` and an
in-memory span exporter."""

import asyncio
import json
import logging
import operator
from collections.abc import Iterator
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agintis.agent_kit.runtime import AgentContext, LangGraphRunner, TracedRunner, streamed
from agintis.modules.agents_catalog.api import RunInput, SourceRef, Token, TranscriptMessage
from agintis.platform.observability import NoTracing, OtlpConfig, TracingError
from agintis.platform.observability.otlp import OtlpTracing, endpoint_for_log
from agintis.shared_kernel.ids import RunId, TenantId, ThreadId
from tests.fakes.agents import FakeRunner
from tests.fakes.graphs import ChatState, Graph, StreamingModel, failing_graph, llm_graph

ENDPOINT = "http://localhost:6006/v1/traces"
RUN_ID = "0123456789abcdef0123456789abcdef"
REDACTED = "__REDACTED__"


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def _tracing(exporter: InMemorySpanExporter, *, hide_content: bool = False) -> OtlpTracing:
    return OtlpTracing(
        OtlpConfig(ENDPOINT, service_name="agintis-test", hide_content=hide_content),
        exporter=exporter,
    )


@pytest.fixture
def tracing(exporter: InMemorySpanExporter) -> Iterator[OtlpTracing]:
    adapter = _tracing(exporter)
    adapter.start()
    yield adapter
    adapter.shutdown()


def _runner(tracing: OtlpTracing | NoTracing) -> TracedRunner:
    graph = llm_graph(StreamingModel(pieces=["Hello ", "world"]))
    return TracedRunner(LangGraphRunner("toy", "A toy agent", graph), tracing)


def _input(thread: str | None = None) -> RunInput:
    return RunInput(
        TenantId("acme"), "secret question", RunId(RUN_ID), ThreadId(thread) if thread else None
    )


async def _drain(runner: TracedRunner, run_input: RunInput) -> list[Any]:
    return [event async for event in runner.astream(run_input)]


def _spans(tracing: OtlpTracing, exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    tracing.force_flush()
    return list(exporter.get_finished_spans())


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _metadata(span: ReadableSpan) -> dict[str, str]:
    raw = _attrs(span).get("metadata")
    return json.loads(raw) if raw else {}


def _input_for(tenant: str, run_id: str, thread: str | None = None) -> RunInput:
    return RunInput(TenantId(tenant), "q", RunId(run_id), ThreadId(thread) if thread else None)


class _Pieces(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topics: Annotated[list[str], operator.add]
    topic: str


class _TopicOut(TypedDict):
    topics: Annotated[list[str], operator.add]


def _fan_out_graph(delay: float) -> Graph:
    """analyze (LLM) -> Send x2 to a subgraph (LLM inside) -> synthesize (streamed LLM)."""
    model = StreamingModel(pieces=["x ", "y"])

    async def inner(state: _Pieces) -> dict[str, Any]:
        await asyncio.sleep(delay)
        await model.ainvoke("q")
        return {"topics": [state["topic"]]}

    sub_builder = StateGraph(_Pieces, context_schema=AgentContext, output_schema=_TopicOut)
    sub_builder.add_node("inner", inner)
    sub_builder.add_edge(START, "inner")
    sub_builder.add_edge("inner", END)
    sub = sub_builder.compile(checkpointer=False)

    async def analyze(state: _Pieces) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    def route(state: _Pieces) -> list[Send]:
        return [Send("topic", {"messages": [], "topics": [], "topic": t}) for t in ("a", "b")]

    async def synthesize(state: _Pieces) -> dict[str, Any]:
        await asyncio.sleep(delay)
        answer = streamed(StreamingModel(pieces=["Hello ", "world"]))
        return {"messages": [await answer.ainvoke(state["messages"])]}

    builder = StateGraph(_Pieces, context_schema=AgentContext)
    builder.add_node("analyze", analyze)
    builder.add_node("topic", sub)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "analyze")
    builder.add_conditional_edges("analyze", route, ["topic"])
    builder.add_edge("topic", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


def _slow_graph(started: asyncio.Event) -> Graph:
    """answer (streamed) -> slow: a node that never ends by itself."""
    model = streamed(StreamingModel(pieces=["Hello ", "world"]))

    async def answer(state: ChatState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    async def slow(state: ChatState) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {}

    builder = StateGraph(ChatState, context_schema=AgentContext)
    builder.add_node("answer", answer)
    builder.add_node("slow", slow)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", "slow")
    builder.add_edge("slow", END)
    return builder.compile()


async def _abort_from_another_task(runner: TracedRunner, run_input: RunInput) -> None:
    """Pull in task A until a Token arrives, close the stream from task B (a pump + a closer)."""
    stream = runner.astream(run_input)

    async def pull() -> None:
        async for event in stream:
            if isinstance(event, Token):
                return

    await asyncio.create_task(pull())
    await asyncio.create_task(stream.aclose())


class TestRunCorrelation:
    async def test_every_span_carries_the_run_metadata_and_one_trace_id(
        self, tracing: OtlpTracing, exporter: InMemorySpanExporter
    ) -> None:
        events = await _drain(_runner(tracing), _input())
        assert "".join(e.text for e in events if isinstance(e, Token)) == "Hello world"

        spans = _spans(tracing, exporter)
        assert spans
        for span in spans:
            metadata = json.loads(_attrs(span)["metadata"])
            assert metadata["tenant_id"] == "acme"
            assert metadata["agent_id"] == "toy"
            assert metadata["run_id"] == RUN_ID
            assert "session.id" not in _attrs(span)
        assert len({span.context.trace_id for span in spans}) == 1  # OTel ids are random
        resource = spans[0].resource.attributes
        assert resource["service.name"] == "agintis-test"
        assert resource["openinference.project.name"] == "agintis-test"

    async def test_thread_runs_are_grouped_in_a_session(
        self, tracing: OtlpTracing, exporter: InMemorySpanExporter
    ) -> None:
        graph = llm_graph(StreamingModel(pieces=["ok"]))
        runner = TracedRunner(LangGraphRunner("toy", "A toy agent", graph, graph), tracing)
        # The graph has no checkpointer (nothing is persisted): only the session wiring matters.
        spans_before = len(_spans(tracing, exporter))
        await _drain(runner, _input(thread="thread-7"))
        spans = _spans(tracing, exporter)[spans_before:]
        assert spans
        assert {_attrs(span).get("session.id") for span in spans} == {"thread-7"}

    async def test_the_scope_is_not_held_across_yields(
        self, tracing: OtlpTracing, exporter: InMemorySpanExporter
    ) -> None:
        """anext in one task and aclose in another must not fail on a context detach."""
        started = asyncio.Event()
        runner = TracedRunner(LangGraphRunner("toy", "t", _slow_graph(started)), tracing)
        await _abort_from_another_task(runner, _input())  # raises on a cross-task detach
        assert _spans(tracing, exporter)

    async def test_an_aborted_run_keeps_its_metadata_on_every_span_including_the_root(
        self, tracing: OtlpTracing, exporter: InMemorySpanExporter
    ) -> None:
        """OpenInference reads the context when a span ENDS: the root ends inside `aclose()`."""
        started = asyncio.Event()
        runner = TracedRunner(LangGraphRunner("toy", "t", _slow_graph(started)), tracing)
        await _abort_from_another_task(runner, _input_for("acme", RUN_ID))
        spans = _spans(tracing, exporter)
        roots = [span for span in spans if span.parent is None]
        assert [root.name for root in roots] == ["LangGraph"]
        for span in spans:
            meta = _metadata(span)
            assert (meta["tenant_id"], meta["agent_id"], meta["run_id"]) == ("acme", "toy", RUN_ID)

    async def test_fan_out_and_subgraph_spans_share_one_trace_and_the_metadata(
        self, tracing: OtlpTracing, exporter: InMemorySpanExporter
    ) -> None:
        runner = TracedRunner(LangGraphRunner("toy", "t", _fan_out_graph(0.01)), tracing)
        events = await _drain(runner, _input_for("acme", RUN_ID))
        assert "".join(e.text for e in events if isinstance(e, Token)) == "Hello world"
        spans = _spans(tracing, exporter)
        assert len(spans) > 6  # root + analyze + 2 topic branches (+ LLM calls) + synthesize
        assert len({span.context.trace_id for span in spans}) == 1
        for span in spans:
            assert _metadata(span)["run_id"] == RUN_ID, span.name
            assert _metadata(span)["tenant_id"] == "acme", span.name

    async def test_concurrent_interleaved_runs_do_not_mix_attributes(
        self, tracing: OtlpTracing, exporter: InMemorySpanExporter
    ) -> None:
        async def run(tenant: str, run_id: str, delay: float) -> None:
            runner = TracedRunner(LangGraphRunner("toy", "t", _fan_out_graph(delay)), tracing)
            async for _ in runner.astream(_input_for(tenant, run_id)):
                await asyncio.sleep(0)  # yield to the other run between events

        first, second = "1" * 32, "2" * 32
        await asyncio.gather(run("t1", first, 0.05), run("t2", second, 0.03))
        spans = _spans(tracing, exporter)
        traces: dict[int, set[tuple[str, str]]] = {}
        for span in spans:
            meta = _metadata(span)
            traces.setdefault(span.context.trace_id, set()).add((meta["tenant_id"], meta["run_id"]))
        assert len(traces) == 2
        assert sorted(next(iter(pairs)) for pairs in traces.values()) == [
            ("t1", first),
            ("t2", second),
        ]
        assert all(len(pairs) == 1 for pairs in traces.values())


class TestHideContent:
    async def test_content_is_redacted_when_hidden(self, exporter: InMemorySpanExporter) -> None:
        adapter = _tracing(exporter, hide_content=True)
        adapter.start()
        try:
            await _drain(_runner(adapter), _input())
            spans = _spans(adapter, exporter)
        finally:
            adapter.shutdown()
        values = " ".join(str(v) for span in spans for v in _attrs(span).values())
        assert "secret question" not in values
        assert "Hello world" not in values
        assert REDACTED in values

    async def test_content_is_recorded_by_default(
        self, tracing: OtlpTracing, exporter: InMemorySpanExporter
    ) -> None:
        await _drain(_runner(tracing), _input())
        values = " ".join(
            str(v) for span in _spans(tracing, exporter) for v in _attrs(span).values()
        )
        assert "secret question" in values
        assert "Hello world" in values


class TestLifecycle:
    def test_start_and_shutdown_are_idempotent(self, exporter: InMemorySpanExporter) -> None:
        adapter = _tracing(exporter)
        adapter.start()
        adapter.start()
        adapter.shutdown()
        adapter.shutdown()

    async def test_restart_works_and_shutdown_uninstruments(self) -> None:
        for _ in range(2):
            exporter = InMemorySpanExporter()
            adapter = _tracing(exporter)
            adapter.start()
            await _drain(_runner(adapter), _input())
            assert _spans(adapter, exporter)
            adapter.shutdown()
            # Uninstrumented: nothing reaches the provider any more.
            exporter.clear()
            await _drain(_runner(NoTracing()), _input())
            assert exporter.get_finished_spans() == ()

    def test_a_second_adapter_is_refused_while_one_is_active(self, tracing: OtlpTracing) -> None:
        with pytest.raises(TracingError, match="already instrumented"):
            _tracing(InMemorySpanExporter()).start()

    async def test_no_global_tracer_provider_is_installed(
        self, tracing: OtlpTracing, exporter: InMemorySpanExporter
    ) -> None:
        from opentelemetry import trace

        assert trace.get_tracer_provider() is not tracing._provider
        assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)


class TestPassThrough:
    async def test_the_wrapper_delegates_the_runner_port(self, tracing: OtlpTracing) -> None:
        runner = _runner(tracing)
        assert runner.agent_id == "toy"
        assert runner.description == "A toy agent"
        assert not runner.supports_threads
        runner.ensure_available()
        assert await runner.pending_interrupts(ThreadId("t")) == ()

    async def test_the_wrapper_delegates_the_transcript(self, tracing: OtlpTracing) -> None:
        turns = (
            TranscriptMessage("user", "hi"),
            TranscriptMessage("assistant", "hello [1]", (SourceRef("doc-1"),)),
        )
        runner = TracedRunner(FakeRunner(messages=turns), tracing)
        assert await runner.thread_messages(ThreadId("t")) == turns


SECRET = "my-secret-passage"


class TestErrorTextRedaction:
    async def _failed_spans(
        self, exporter: InMemorySpanExporter, *, hide_content: bool
    ) -> list[ReadableSpan]:
        adapter = _tracing(exporter, hide_content=hide_content)
        adapter.start()
        try:
            runner = TracedRunner(
                LangGraphRunner("toy", "t", failing_graph(ValueError(f"bad output: {SECRET}"))),
                adapter,
            )
            with pytest.raises(ValueError, match="bad output"):
                await _drain(runner, _input())
            return _spans(adapter, exporter)
        finally:
            adapter.shutdown()

    @staticmethod
    def _error_text(spans: list[ReadableSpan]) -> str:
        parts: list[str] = []
        for span in spans:
            parts.append(span.status.description or "")
            for event in span.events:
                parts.extend(str(value) for value in (event.attributes or {}).values())
        return " ".join(parts)

    async def test_error_text_is_redacted_when_hidden_but_the_type_survives(
        self, exporter: InMemorySpanExporter
    ) -> None:
        spans = await self._failed_spans(exporter, hide_content=True)
        errored = [span for span in spans if span.status.status_code.name == "ERROR"]
        assert errored
        assert SECRET not in self._error_text(spans)
        assert all(span.status.description == REDACTED for span in errored)
        exceptions = [e for span in spans for e in span.events if e.name == "exception"]
        assert exceptions
        for event in exceptions:
            attributes = dict(event.attributes or {})
            assert attributes["exception.type"] == "ValueError"
            assert attributes["exception.message"] == REDACTED
            assert attributes["exception.stacktrace"] == REDACTED

    async def _cancelled_spans(
        self, exporter: InMemorySpanExporter, *, hide_content: bool
    ) -> list[ReadableSpan]:
        """A client disconnect: the consumer task is cancelled while a node is still running."""
        adapter = _tracing(exporter, hide_content=hide_content)
        adapter.start()
        try:
            started = asyncio.Event()
            runner = TracedRunner(LangGraphRunner("toy", "t", _slow_graph(started)), adapter)

            async def consume() -> None:
                async for _ in runner.astream(_input_for("acme", RUN_ID)):
                    pass

            task = asyncio.create_task(consume())
            await asyncio.wait_for(started.wait(), timeout=10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return _spans(adapter, exporter)
        finally:
            adapter.shutdown()

    async def test_a_cancelled_run_root_span_error_text_is_redacted_when_hidden(
        self, exporter: InMemorySpanExporter
    ) -> None:
        spans = await self._cancelled_spans(exporter, hide_content=True)
        roots = [span for span in spans if span.parent is None]
        assert [root.name for root in roots] == ["LangGraph"]
        for span in spans:
            assert span.status.description in (None, "", REDACTED), span.name
            for event in span.events:
                attributes = dict(event.attributes or {})
                for key in ("exception.message", "exception.stacktrace"):
                    assert attributes.get(key, REDACTED) == REDACTED, span.name
        text = self._error_text(spans)
        assert "Task" not in text
        assert ".py" not in text

    async def test_a_cancelled_run_records_error_text_by_default(
        self, exporter: InMemorySpanExporter
    ) -> None:
        """Guards the redaction test above against passing vacuously."""
        spans = await self._cancelled_spans(exporter, hide_content=False)
        root = next(span for span in spans if span.parent is None)
        has_text = bool(root.status.description) or any(
            "exception.stacktrace" in dict(event.attributes or {}) for event in root.events
        )
        assert has_text

    async def test_error_text_is_recorded_by_default(self, exporter: InMemorySpanExporter) -> None:
        spans = await self._failed_spans(exporter, hide_content=False)
        assert SECRET in self._error_text(spans)


class TestOtlpExporterConfig:
    def test_the_timeout_is_passed_to_the_exporter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        class Recorder(InMemorySpanExporter):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__()
                seen.update(kwargs)

        monkeypatch.setattr("agintis.platform.observability.otlp.OTLPSpanExporter", Recorder)
        adapter = OtlpTracing(OtlpConfig(ENDPOINT, timeout_seconds=2.5))
        adapter.start()
        adapter.shutdown()
        assert seen["timeout"] == 2.5
        assert seen["endpoint"] == ENDPOINT

    def test_shutdown_does_not_force_flush_a_second_time(
        self, exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _tracing(exporter)
        adapter.start()
        assert adapter._provider is not None
        calls: list[object] = []

        def flush(*args: object, **kwargs: object) -> bool:
            calls.append(args)
            return True

        monkeypatch.setattr(adapter._provider, "force_flush", flush)
        adapter.shutdown()
        assert not calls

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("http://localhost:6006/v1/traces", "http://localhost:6006"),
            (
                "https://user:pw@cloud.example.com/api/public/otel/v1/traces?key=abc",
                "https://cloud.example.com",
            ),
            ("http://[::1]:4318/v1/traces", "http://[::1]:4318"),
            ("https://host:8443", "https://host:8443"),
        ],
    )
    def test_the_logged_endpoint_has_no_credentials_or_path(self, raw: str, expected: str) -> None:
        assert endpoint_for_log(raw) == expected

    def test_start_logs_only_scheme_host_and_port(
        self, exporter: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter = OtlpTracing(
            OtlpConfig("https://user:s3cret@cloud.example.com:4318/v1/traces?token=abc"),
            exporter=exporter,
        )
        with caplog.at_level(logging.INFO, logger="agintis.platform.observability.otlp"):
            adapter.start()
        adapter.shutdown()
        assert "https://cloud.example.com:4318" in caplog.text
        assert "s3cret" not in caplog.text
        assert "token" not in caplog.text
