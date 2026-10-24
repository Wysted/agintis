"""Toy compiled graphs (and streaming fake models) for the LangGraph runner tests."""

import asyncio
import itertools
import operator
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime
from langgraph.types import Send, interrupt

from agintis.agent_kit.runtime import (
    STREAM_TOKENS_TAG,
    AgentContext,
    LangGraphRunner,
    emit,
    streamed,
)
from agintis.modules.agents_catalog.api import Clarification, SourceRef, Sources, Step


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


Graph = Pregel[Any, Any, Any, Any]
Ctx = Runtime[AgentContext]


class StreamingModel(BaseChatModel):
    """Streams `pieces`; usage and finish reason arrive in the last chunk (or in two chunks).

    With `disable_streaming=True` (or when only `_generate` is used) it behaves like a
    non-streaming provider: one full `AIMessage` carrying usage and finish reason.
    """

    pieces: list[str]
    input_tokens: int = 4
    output_tokens: int = 2
    finish_reason: str = "stop"
    separate_usage_chunk: bool = False  # ChatOpenAI style: finish chunk, then usage-only chunk
    cached_tokens: int | None = None  # prompt-cache hits; None: the provider reports no details

    @property
    def _llm_type(self) -> str:
        return "streaming-model"

    def _usage(self) -> UsageMetadata:
        usage = UsageMetadata(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
        )
        if self.cached_tokens is not None:
            usage["input_token_details"] = {"cache_read": self.cached_tokens}
        return usage

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(
            content="".join(self.pieces),
            usage_metadata=self._usage(),
            response_metadata={"finish_reason": self.finish_reason},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for piece in self.pieces:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
        finish = {"finish_reason": self.finish_reason}
        if self.separate_usage_chunk:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", response_metadata=finish))
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", usage_metadata=self._usage())
            )
        else:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="", usage_metadata=self._usage(), response_metadata=finish
                )
            )

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, AIMessage]:
        return self.bind(tools=list(tools), **kwargs)


def llm_node(model: Runnable[Any, Any]) -> Any:
    async def node(state: ChatState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    return node


def _builder(schema: Any = ChatState) -> StateGraph[Any, Any, Any, Any]:
    return StateGraph(schema, context_schema=AgentContext)


def _compile(builder: StateGraph[Any, Any, Any, Any], *nodes: str, **kwargs: Any) -> Graph:
    builder.add_edge(START, nodes[0])
    for left, right in itertools.pairwise(nodes):
        builder.add_edge(left, right)
    builder.add_edge(nodes[-1], END)
    return builder.compile(**kwargs)


def llm_graph(
    answer: BaseChatModel | Runnable[Any, Any],
    internal: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Graph:
    """[internal (untagged)] -> answer (tagged with `streamed` unless already a runnable)."""
    builder = _builder()
    names = []
    if internal is not None:
        builder.add_node("analyze", llm_node(internal))
        names.append("analyze")
    model = streamed(answer) if isinstance(answer, BaseChatModel) else answer
    builder.add_node("answer", llm_node(model))
    names.append("answer")
    return _compile(builder, *names, checkpointer=checkpointer)


def leaky_parent_graph(inner: BaseChatModel) -> Graph:
    """A tagged PARENT runnable calling an untagged inner model: the tag is inherited."""

    async def call_inner(messages: Any) -> Any:
        return await inner.ainvoke(messages)

    parent = RunnableLambda(call_inner).with_config(tags=[STREAM_TOKENS_TAG])
    builder = _builder()
    builder.add_node("parent", llm_node(parent))
    return _compile(builder, "parent")


def emitting_graph() -> Graph:
    async def node(state: ChatState, runtime: Ctx) -> dict[str, Any]:
        emit(runtime, Step("custom-step", "started"))
        emit(runtime, Sources((SourceRef("doc.md", "Doc"),)))
        emit(runtime, Clarification("Which one?", ("a", "b")))
        runtime.stream_writer({"unknown": "payload"})
        runtime.stream_writer("also unknown")
        return {}

    builder = _builder()
    builder.add_node("emitter", node)
    return _compile(builder, "emitter")


def subgraph_graph(inner_model: BaseChatModel) -> Graph:
    inner = _builder()
    inner.add_node("inner_answer", llm_node(streamed(inner_model)))
    sub = _compile(inner, "inner_answer")

    async def prepare(state: ChatState) -> dict[str, Any]:
        return {}

    outer = _builder()
    outer.add_node("prepare", prepare)
    outer.add_node("topic", sub)
    return _compile(outer, "prepare", "topic")


def failing_graph(error: Exception) -> Graph:
    async def boom(state: ChatState) -> dict[str, Any]:
        raise error

    builder = _builder()
    builder.add_node("boom", boom)
    return _compile(builder, "boom")


class BlockingProbe:
    """Records that a blocking node started and whether it was cancelled/cleaned up."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.finalized = asyncio.Event()


def blocking_graph(probe: BlockingProbe) -> Graph:
    async def first(state: ChatState) -> dict[str, Any]:
        return {}

    async def block(state: ChatState, runtime: Ctx) -> dict[str, Any]:
        emit(runtime, Step("blocking", "started"))
        probe.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            probe.cancelled.set()
            raise
        finally:
            probe.finalized.set()
        return {}

    builder = _builder()
    builder.add_node("first", first)
    builder.add_node("block", block)
    return _compile(builder, "first", "block")


def config_probe_graph(
    seen: list[RunnableConfig],
    runtimes: list[Ctx],
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Graph:
    async def node(state: ChatState, runtime: Ctx) -> dict[str, Any]:
        seen.append(get_config())
        runtimes.append(runtime)
        return {}

    builder = _builder()
    builder.add_node("probe", node)
    return _compile(builder, "probe", checkpointer=checkpointer)


def state_probe_graph(
    seen: list[list[BaseMessage]], checkpointer: BaseCheckpointSaver[Any] | None = None
) -> Graph:
    async def node(state: ChatState) -> dict[str, Any]:
        seen.append(list(state["messages"]))
        return {}

    builder = _builder()
    builder.add_node("probe", node)
    return _compile(builder, "probe", checkpointer=checkpointer)


def interrupting_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Graph:
    async def ask(state: ChatState) -> dict[str, Any]:
        interrupt("which one?")
        return {}

    builder = _builder()
    builder.add_node("ask", ask)
    return _compile(builder, "ask", checkpointer=checkpointer)


def asking_graph(
    payload: object = "Which one?", checkpointer: BaseCheckpointSaver[Any] | None = None
) -> Graph:
    """`ask` interrupts with `payload`, then `answer` replies with the resume value."""

    async def ask(state: ChatState) -> dict[str, Any]:
        answer = interrupt(payload)
        return {"messages": [AIMessage(content=f"you chose {answer}")]}

    builder = _builder()
    builder.add_node("ask", ask)
    return _compile(builder, "ask", checkpointer=checkpointer)


class CitedState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    cited_sources: Annotated[dict[str, list[dict[str, Any]]], operator.or_]


def cited_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Graph:
    """Each turn: a system note (not a chat turn), then an answer citing `doc-<n>` (the 1st
    answer cites nothing); the answers are the ones with ids `a<n>`."""

    async def answer(state: CitedState) -> dict[str, Any]:
        n = sum(isinstance(m, AIMessage) for m in state["messages"]) + 1
        cited = (
            {} if n == 1 else {f"a{n}": [{"source": f"doc-{n}", "title": "Doc", "locator": None}]}
        )
        return {
            "messages": [SystemMessage("internal note"), AIMessage(f"answer {n}", id=f"a{n}")],
            "cited_sources": cited,
        }

    builder = _builder(CitedState)
    builder.add_node("answer", answer)
    return _compile(builder, "answer", checkpointer=checkpointer)


class TopicState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str


def two_questions_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Graph:
    """Fan-out of two tasks that BOTH interrupt (two pending interrupts), then answer."""

    def route(state: TopicState) -> list[Send]:
        return [Send("ask", {"messages": [], "topic": t}) for t in ("a", "b")]

    async def ask(state: TopicState) -> dict[str, Any]:
        answer = interrupt(f"Question {state['topic']}")
        return {"messages": [AIMessage(content=f"{state['topic']}={answer}")]}

    builder = _builder(TopicState)
    builder.add_node("ask", ask)
    builder.add_conditional_edges(START, route, ["ask"])
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=checkpointer)


def ask_then_fail_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Graph:
    """`ask` interrupts; once answered, the NEXT node fails (the resume is already consumed)."""

    async def ask(state: ChatState) -> dict[str, Any]:
        interrupt("Which one?")
        return {}

    async def explode(state: ChatState) -> dict[str, Any]:
        raise RuntimeError("boom after the answer")

    builder = _builder()
    builder.add_node("ask", ask)
    builder.add_node("explode", explode)
    return _compile(builder, "ask", "explode", checkpointer=checkpointer)


def interrupt_and_error_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Graph:
    """Two parallel nodes in ONE superstep: one interrupts, the other raises."""

    async def ask(state: ChatState) -> dict[str, Any]:
        interrupt("Which one?")
        return {}

    async def explode(state: ChatState) -> dict[str, Any]:
        raise RuntimeError("boom in the same superstep")

    builder = _builder()
    builder.add_node("ask", ask)
    builder.add_node("explode", explode)
    builder.add_edge(START, "ask")
    builder.add_edge(START, "explode")
    builder.add_edge("ask", END)
    builder.add_edge("explode", END)
    return builder.compile(checkpointer=checkpointer)


@dataclass(frozen=True)
class Threaded:
    """A runner built the way real agents are: plain graph + checkpointed graph, one saver."""

    runner: LangGraphRunner
    graph: Graph  # the checkpointed one: use it to read the state
    saver: BaseCheckpointSaver[Any]


def threaded(
    make: Callable[[BaseCheckpointSaver[Any] | None], Graph],
    agent_id: str = "toy",
    saver: BaseCheckpointSaver[Any] | None = None,
) -> Threaded:
    saver = saver or InMemorySaver()
    graph = make(saver)
    return Threaded(
        LangGraphRunner(agent_id, "A toy agent", make(None), thread_graph=graph), graph, saver
    )


class _InnerState(TypedDict):
    q: str
    answer: str


def _inner_asking_graph(depth: int) -> Graph:
    """A subgraph that interrupts; with `depth` > 1 it invokes another one INSIDE its node."""
    inner = _inner_asking_graph(depth - 1) if depth > 1 else None

    async def ask(state: _InnerState) -> dict[str, Any]:
        if inner is not None:
            out = await inner.ainvoke({"q": state["q"], "answer": ""})
            return {"answer": out["answer"]}
        return {"answer": interrupt(f"deep {state['q']}")}

    builder = StateGraph(_InnerState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile()


def subgraph_as_node_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Graph:
    """The parent adds a compiled subgraph that interrupts AS A NODE (shared `messages`)."""

    async def ask(state: ChatState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"sub={interrupt('Question in the subgraph')}")]}

    inner = StateGraph(ChatState)
    inner.add_node("ask", ask)
    inner.add_edge(START, "ask")
    inner.add_edge("ask", END)
    outer = _builder()
    outer.add_node("sub", inner.compile())
    return _compile(outer, "sub", checkpointer=checkpointer)


def nested_in_node_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Graph:
    """Fan-out of two nodes that each INVOKE a two-level nested subgraph that interrupts."""
    inner = _inner_asking_graph(2)

    def route(state: TopicState) -> list[Send]:
        return [Send("topic", {"messages": [], "topic": t}) for t in ("a", "b")]

    async def topic(state: TopicState) -> dict[str, Any]:
        out = await inner.ainvoke({"q": state["topic"], "answer": ""})
        return {"messages": [AIMessage(content=f"{state['topic']}={out['answer']}")]}

    builder = _builder(TopicState)
    builder.add_node("topic", topic)
    builder.add_conditional_edges(START, route, ["topic"])
    builder.add_edge("topic", END)
    return builder.compile(checkpointer=checkpointer)


class FanOutState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topics: Annotated[list[str], operator.add]
    topic: str


def fan_out_graph() -> Graph:
    def route(state: FanOutState) -> list[Send]:
        return [Send("topic", {"messages": [], "topics": [], "topic": t}) for t in ("a", "b")]

    async def topic(state: FanOutState) -> dict[str, Any]:
        return {"topics": [state["topic"]]}

    builder = _builder(FanOutState)
    builder.add_node("topic", topic)
    builder.add_conditional_edges(START, route, ["topic"])
    builder.add_edge("topic", END)
    return builder.compile()
