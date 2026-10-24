"""Example agent: plain chat as a one-node LangGraph graph (no RAG).

Shows the minimal shape of an agent on top of `agent_kit.runtime`: a callable-class node that
streams the synthesis model through `streamed(...)`, compiled and wrapped in `LangGraphRunner`.

Memory: the state is `MessagesState`, so with a checkpointer a thread keeps the whole
conversation (each run appends the new user message and the reply). The graph is compiled twice
from one builder: plain for stateless runs (a graph compiled WITH a checkpointer refuses to run
without a thread id) and with the checkpointer for thread runs.

The history is UNBOUNDED: the whole conversation goes to the model on every turn. Trimming or
summarizing is a node concern (planned with the 3.5 nodes), not done here.
"""

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime

from agintis.agent_kit.runtime import AgentContext, LangGraphRunner, streamed
from agintis.agent_kit.usage import UsageRecorder
from agintis.modules.agents_catalog.api import AgentUnavailableError
from agintis.platform.llm import ChatModels, ChatRole, LlmNotConfiguredError

AGENT_ID = "direct-chat"
DESCRIPTION = "Plain chat with the configured language model (no retrieval)."
NODE_NAME = "chat"


class ChatNode:
    """Streams the configured model for `role`.

    Takes `ChatModels` (not a model) and resolves it per call, so a disabled LLM does not break
    boot nor the ingest/eval CLIs.
    """

    def __init__(self, chat_models: ChatModels, role: ChatRole = ChatRole.SYNTHESIS) -> None:
        self._chat_models = chat_models
        self._role = role

    async def __call__(
        self, state: MessagesState, runtime: Runtime[AgentContext]
    ) -> dict[str, list[AIMessage]]:
        model = streamed(self._chat_models.get(self._role))
        reply = await model.ainvoke(state["messages"])
        return {"messages": [reply]}


def build_graph(
    chat_models: ChatModels,
    role: ChatRole = ChatRole.SYNTHESIS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Pregel[Any, Any, Any, Any]:
    return _builder(chat_models, role).compile(checkpointer=checkpointer)


def _builder(chat_models: ChatModels, role: ChatRole) -> StateGraph[Any, Any, Any, Any]:
    builder = StateGraph(MessagesState, context_schema=AgentContext)
    builder.add_node(NODE_NAME, ChatNode(chat_models, role))
    builder.add_edge(START, NODE_NAME)
    builder.add_edge(NODE_NAME, END)
    return builder


def direct_chat_runner(
    chat_models: ChatModels,
    role: ChatRole = ChatRole.SYNTHESIS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    usage: UsageRecorder | None = None,
) -> LangGraphRunner:
    """The runner, with a pre-stream check that reports a disabled/misconfigured LLM (503).

    With a `checkpointer` the agent also supports threads (multi-turn memory).

    `ChatModels.get()` builds (and caches) the model without network access.
    """

    def ensure_available() -> None:
        try:
            chat_models.get(role)
        except LlmNotConfiguredError as exc:
            raise AgentUnavailableError(
                "The language model is not configured", code="llm_not_configured"
            ) from exc
        except Exception as exc:
            raise AgentUnavailableError(
                "The language model is misconfigured", code="llm_misconfigured"
            ) from exc

    builder = _builder(chat_models, role)
    return LangGraphRunner(
        AGENT_ID,
        DESCRIPTION,
        builder.compile(),
        thread_graph=builder.compile(checkpointer=checkpointer) if checkpointer else None,
        availability=ensure_available,
        usage=usage,
    )
