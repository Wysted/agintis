"""Generic LangGraph -> `AgentRunner` adapter: works for any compiled graph.

Stream mapping (`astream(..., stream_mode=[messages, tasks, custom], subgraphs=True, version="v2")`,
every part is `{"type", "ns", "data"}`):

- `messages`: `data = (message, metadata)` for EVERY LLM call of every node. Streamed calls
  arrive as `AIMessageChunk`s; non-streamed calls (`disable_streaming`, `_generate`-only models)
  as one full `AIMessage` (LLM runs carry `ls_provider` in the metadata, messages merely
  returned by a node do not and are ignored; langgraph dedupes by message id, so nothing is
  counted twice). Usage is aggregated from all of them. Text becomes `Token` only when the call
  carries `STREAM_TOKENS_TAG` (see `streamed`), so internal calls (analyzer, router) stay silent.
  `messages` mode forces streaming on every call, so the `stream_usage`/`base_url` gotcha affects
  the whole run; calls tagged `nostream` are excluded by langgraph, usage included.
- `tasks`: `data` has `input` when a node starts and `result`/`error`/`interrupts` when it ends,
  for the root graph and every subgraph (`ns`). Mapped to `Step(started|finished)`; the node name
  is prefixed with the subgraph path (`"topic/retrieve"`). These are PROGRESS HINTS, not a
  guaranteed started/finished pairing: failed and interrupted tasks get no `finished`, cached or
  re-applied tasks may start without a result, and `Send` fan-out repeats node names.
  A `step_label` strategy (default: none) may name a task from its input at START (e.g. the topic
  of a `Send` fan-out branch); the label is remembered by task id for the matching END and is
  inherited by the steps of its subgraph (their namespace carries the parent task id). The raw
  task input never reaches the client: only what the strategy returns, as a short string.
- `custom`: nodes write the event dataclasses themselves (`emit`); allowed types pass through,
  anything else is ignored.

Threads (state kept between runs): `RunInput.thread_id` becomes `configurable.thread_id` and the
run uses `thread_graph` (the same graph compiled WITH a checkpointer; a graph compiled with one
cannot run without a thread id, so stateless runs keep the plain `graph`). `RunInput.resume`
becomes `Command(resume={interrupt_id: value})`: always an explicit id map (LangGraph raises on a
bare value with several pending interrupts and takes a dict of 32-hex keys for an id map).
`pending_interrupts` reads the unanswered interrupts from the checkpoint (the source of truth for
the thread status); `thread_messages` reads the chat turns (human and AI messages, plus the
sources the state's `cited_sources` channel keeps per AI message id). A task that ends with
`interrupts` pauses the run: each distinct interrupt becomes a `Clarification` carrying its
`interrupt_id` (see `clarification_from_interrupt`) and the run finishes with
`RunCompleted(finish_reason="interrupted")`. LangGraph discards a pending interrupt when NEW input
(not a resume) reaches the thread: callers must resume it first.

Nothing is swallowed: node exceptions and `CancelledError` propagate; `RunAgent` classifies them.
Graph nodes must be `async`: a sync node runs in a thread and cannot be cancelled when the
client goes away.
"""

import logging
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import aclosing
from dataclasses import replace
from typing import Any, Protocol, cast
from uuid import UUID

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel
from langgraph.types import Command, StreamMode, StreamPart

from agintis.agent_kit.runtime.context import ContextFactory, agent_context
from agintis.agent_kit.runtime.events import NODE_EVENT_TYPES, STREAM_TOKENS_TAG
from agintis.agent_kit.runtime.finish_reason import finish_reason_of
from agintis.modules.agents_catalog.api import (
    DEFAULT_FINISH_REASON,
    INTERRUPTED_FINISH_REASON,
    AgentRunner,
    AgentUnavailableError,
    Clarification,
    RunCompleted,
    RunInput,
    RunnerEvent,
    SourceRef,
    Step,
    Token,
    TokenUsage,
    TranscriptMessage,
)
from agintis.shared_kernel.ids import ThreadId

logger = logging.getLogger(__name__)

DEFAULT_RECURSION_LIMIT = 25
STREAM_MODES: list[StreamMode] = ["messages", "tasks", "custom"]

# Separator of the subgraph path in Step node names.
_NAMESPACE_SEPARATOR = "/"
# Internal pseudo-nodes (e.g. `__start__`) are not steps.
_INTERNAL_NODE_PREFIX = "__"


def _always_available() -> None:
    return None


# Labels are short display names (a topic), never free text.
MAX_STEP_LABEL_CHARS = 100


class StepLabeler(Protocol):
    """Strategy: a display label for a task, from its node name and its input (None = none).

    The input can carry user text (a query): return only a safe, short name derived from it.
    """

    def __call__(self, node: str, task_input: Any) -> str | None: ...


def no_step_label(node: str, task_input: Any) -> str | None:
    return None


class InputMapper(Protocol):
    """Strategy: builds the graph input from the run input."""

    def __call__(self, run_input: RunInput) -> Any: ...


def messages_input(run_input: RunInput) -> dict[str, Any]:
    """Default mapper for graphs whose state has a `messages` channel."""
    if run_input.message is None:
        raise ValueError("messages_input needs a message run (resumes bypass the input mapper)")
    return {"messages": [HumanMessage(content=run_input.message)]}


# Shown when an interrupt carries a payload that is not a question (it is never echoed: it may
# hold internal state).
FALLBACK_QUESTION = "The agent needs more information to continue."


def clarification_from_interrupt(value: object, interrupt_id: str = "") -> Clarification:
    """Maps an interrupt payload to the question shown to the user (tagged with `interrupt_id`).

    Accepted: a `Clarification`, a plain non-empty string (the question), or a mapping
    `{"question": str, "options": [str, ...]?}`. Anything else (a malformed payload) yields a
    generic question and a warning: the run still pauses and can be resumed, and the raw payload
    is neither shown nor lost silently.
    """
    if isinstance(value, Clarification):
        return replace(value, interrupt_id=interrupt_id)
    if isinstance(value, str) and value.strip():
        return Clarification(value, (), interrupt_id)
    if isinstance(value, Mapping):
        question = value.get("question")
        options = value.get("options", ())
        if (
            isinstance(question, str)
            and question.strip()
            and isinstance(options, list | tuple)
            and all(isinstance(option, str) for option in options)
        ):
            return Clarification(question, tuple(options), interrupt_id)
    logger.warning("Interrupt payload of type %s is not a question", type(value).__name__)
    return Clarification(FALLBACK_QUESTION, (), interrupt_id)


def _step_name(ns: tuple[str, ...], name: str) -> str:
    # Namespace segments look like "node:task_id": keep only the node name.
    path = [segment.split(":", 1)[0] for segment in ns]
    return _NAMESPACE_SEPARATOR.join([*path, name])


class _RunAccumulator:
    """Per-run state: usage over all LLM calls, finish reason of the streamed answer."""

    def __init__(self) -> None:
        self.usage: UsageMetadata | None = None
        self.finish_reason: str | None = None
        self.interrupted = False
        self.interrupt_ids: set[str] = set()
        # task id -> label, set at task START and dropped at its END.
        self.step_labels: dict[str, str] = {}

    def completed(self) -> RunCompleted:
        usage = self.usage
        finish_reason = INTERRUPTED_FINISH_REASON if self.interrupted else self.finish_reason
        return RunCompleted(
            finish_reason=finish_reason or DEFAULT_FINISH_REASON,
            usage=TokenUsage(
                usage["input_tokens"],
                usage["output_tokens"],
                usage["total_tokens"],
                # `add_usage` sums the nested details; a provider without them reports none
                cached_input_tokens=(usage.get("input_token_details") or {}).get("cache_read") or 0,
            )
            if usage
            else None,
        )


def _inherited_label(ns: tuple[str, ...], state: _RunAccumulator) -> str | None:
    """The label of the nearest ancestor task: namespace segments are `node:task_id`."""
    for segment in reversed(ns):
        _, _, task_id = segment.partition(":")
        label = state.step_labels.get(task_id.split(":", 1)[0])
        if label is not None:
            return label
    return None


class LangGraphRunner(AgentRunner):
    """Runs a compiled LangGraph graph behind the `AgentRunner` port.

    `thread_graph` is the same graph compiled with a checkpointer; without it (and without a
    checkpointer on `graph` itself) the runner is stateless and `supports_threads` is False.
    """

    def __init__(
        self,
        agent_id: str,
        description: str,
        graph: Pregel[Any, Any, Any, Any],
        thread_graph: Pregel[Any, Any, Any, Any] | None = None,
        input_mapper: InputMapper = messages_input,
        context_factory: ContextFactory = agent_context,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        availability: Callable[[], None] = _always_available,
        step_label: StepLabeler = no_step_label,
    ) -> None:
        self._agent_id = agent_id
        self._description = description
        if isinstance(graph.checkpointer, BaseCheckpointSaver):
            # Such a graph cannot run without a thread id: it would fail MID-STREAM on every
            # stateless run. Pass the plain graph and the checkpointed one separately.
            raise ValueError(
                f"Agent {agent_id!r}: `graph` is compiled with a checkpointer; pass the plain "
                "graph as `graph` and the checkpointed one as `thread_graph`"
            )
        self._graph = graph
        self._thread_graph = thread_graph
        self._input_mapper = input_mapper
        self._context_factory = context_factory
        self._recursion_limit = recursion_limit
        self._availability = availability
        self._step_label = step_label

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def description(self) -> str:
        return self._description

    @property
    def supports_threads(self) -> bool:
        return self._thread_graph is not None

    def ensure_available(self) -> None:
        """Delegates to the injected `availability` check (raises `AgentUnavailableError`)."""
        self._availability()

    def _config(self, run_input: RunInput) -> RunnableConfig:
        """The single place where the run config is built (recursion limit, run id, thread id).

        Per-request data (tenant, run id) goes in `context=` (`AgentContext`), NEVER here: the
        configurable dict is persisted with every checkpoint and is not typed. Only the
        `thread_id` (the checkpoint key) belongs in it.
        `run_id` is the root tracer run id (Langfuse/LangSmith): reusing ours correlates a trace
        with the SSE run. LangChain requires a UUID, so it is set only when ours is one (the
        default generator is a uuid4 hex). langgraph reserves it (copied to `metadata.run_id`; the
        same run_id over an existing checkpoint is treated as a resume): one FRESH run_id per run,
        never reused across turns or resumes. Conversation continuity is `thread_id`.
        """
        config: RunnableConfig = {"recursion_limit": self._recursion_limit}
        try:
            config["run_id"] = UUID(run_input.run_id)
        except ValueError:
            logger.debug(
                "Run id %r is not a UUID; tracer run id left to LangChain", run_input.run_id
            )
        if run_input.thread_id is not None:
            config["configurable"] = {"thread_id": run_input.thread_id}
        return config

    def _graph_for(self, run_input: RunInput) -> Pregel[Any, Any, Any, Any]:
        if run_input.thread_id is None:
            return self._graph
        if self._thread_graph is None:
            raise AgentUnavailableError(
                f"Agent {self._agent_id!r} was built without a checkpointer",
                code="threads_unavailable",
            )
        return self._thread_graph

    def _input(self, run_input: RunInput) -> Any:
        if run_input.resume is not None:
            return Command(resume=dict(run_input.resume))
        return self._input_mapper(run_input)

    async def pending_interrupts(self, thread_id: ThreadId) -> tuple[Clarification, ...]:
        if self._thread_graph is None:
            return ()
        snapshot = await self._thread_graph.aget_state({"configurable": {"thread_id": thread_id}})
        return tuple(
            clarification_from_interrupt(interrupt.value, interrupt.id)
            for interrupt in snapshot.interrupts
        )

    async def thread_messages(self, thread_id: ThreadId) -> tuple[TranscriptMessage, ...]:
        if self._thread_graph is None:
            return ()
        snapshot = await self._thread_graph.aget_state({"configurable": {"thread_id": thread_id}})
        cited = snapshot.values.get("cited_sources", {})
        turns: list[TranscriptMessage] = []
        for message in snapshot.values.get("messages", []):
            if isinstance(message, HumanMessage):
                turns.append(TranscriptMessage("user", message.text))
            elif isinstance(message, AIMessage):
                sources = tuple(
                    SourceRef(s["source"], s["title"], s["locator"])
                    for s in cited.get(message.id, ())
                )
                turns.append(TranscriptMessage("assistant", message.text, sources))
        return tuple(turns)

    async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
        state = _RunAccumulator()
        # Pregel.astream is typed AsyncIterator but is an async generator: closable.
        stream = cast(
            AsyncGenerator[StreamPart[Any, Any]],
            self._graph_for(run_input).astream(
                self._input(run_input),
                self._config(run_input),
                context=self._context_factory(run_input),
                stream_mode=STREAM_MODES,
                subgraphs=True,
                version="v2",
            ),
        )
        async with aclosing(stream) as parts:
            async for part in parts:
                for event in self._map(part, state):
                    yield event
        yield state.completed()

    def _map(self, part: StreamPart[Any, Any], state: _RunAccumulator) -> list[RunnerEvent]:
        match part["type"]:
            case "messages":
                events: list[RunnerEvent | None] = [self._map_message(part["data"], state)]
            case "tasks":
                events = [
                    self._map_task(part["ns"], part["data"], state),
                    *self._map_interrupts(part["ns"], part["data"], state),
                ]
            case "custom":
                events = [self._map_custom(part["data"])]
            case _:
                events = []
        return [event for event in events if event is not None]

    @staticmethod
    def _map_interrupts(
        ns: tuple[str, ...], data: Mapping[str, Any], state: _RunAccumulator
    ) -> list[Clarification]:
        """One `Clarification` per distinct interrupt (parent and subgraph tasks both report it).

        The interrupt id is the identity; without one (not expected) the namespace-qualified task
        id and position stand in, so two tasks never collapse into one question.
        """
        if data.get("error") is not None:
            return []
        questions: list[Clarification] = []
        for index, interrupt in enumerate(data.get("interrupts") or ()):
            state.interrupted = True
            interrupt_id = str(interrupt.get("id") or "")
            key = interrupt_id or f"{'/'.join(ns)}|{data.get('id')}|{index}"
            if key in state.interrupt_ids:
                continue
            state.interrupt_ids.add(key)
            questions.append(clarification_from_interrupt(interrupt.get("value"), interrupt_id))
        return questions

    @staticmethod
    def _map_message(data: Any, state: _RunAccumulator) -> Token | None:
        message, metadata = data
        if not isinstance(message, AIMessage):
            return None
        # A full AIMessage is an LLM run only with `ls_provider`; otherwise a node returned it.
        if not isinstance(message, AIMessageChunk) and "ls_provider" not in metadata:
            return None
        if message.usage_metadata:
            state.usage = add_usage(state.usage, message.usage_metadata)
        if STREAM_TOKENS_TAG not in (metadata.get("tags") or ()):
            return None
        state.finish_reason = finish_reason_of(message) or state.finish_reason
        return Token(message.text) if message.text else None

    def _map_task(
        self, ns: tuple[str, ...], data: Mapping[str, Any], state: _RunAccumulator
    ) -> Step | None:
        name = data["name"]
        if name.startswith(_INTERNAL_NODE_PREFIX):
            return None
        task_id = data.get("id")
        if "input" in data:
            own = self._label_of(name, data["input"])
            if own is not None and isinstance(task_id, str):
                state.step_labels[task_id] = own
            return Step(_step_name(ns, name), "started", own or _inherited_label(ns, state))
        if data.get("error") is None and not data.get("interrupts"):
            own = state.step_labels.pop(task_id, None) if isinstance(task_id, str) else None
            return Step(_step_name(ns, name), "finished", own or _inherited_label(ns, state))
        return None

    def _label_of(self, node: str, task_input: Any) -> str | None:
        """The strategy's label, defensively: a labeler bug must not fail the run."""
        try:
            label = self._step_label(node, task_input)
        except Exception as exc:
            logger.warning("Step labeler failed for node %r (%s)", node, type(exc).__name__)
            return None
        if not isinstance(label, str):
            return None
        return label.strip()[:MAX_STEP_LABEL_CHARS] or None

    @staticmethod
    def _map_custom(data: object) -> RunnerEvent | None:
        if isinstance(data, NODE_EVENT_TYPES):
            return data
        logger.debug("Ignoring custom stream payload of type %s", type(data).__name__)
        return None
