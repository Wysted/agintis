"""SynthesizeNode: the answer, grounded on the numbered sources and streamed to the client."""

import logging
import re
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agintis.agent_kit import source_text
from agintis.agent_kit.nodes.history import MAX_HISTORY_MESSAGES, last_human_text, recent_history
from agintis.agent_kit.nodes.prompts import DEFAULT_SYNTHESIZE_PROMPT
from agintis.agent_kit.nodes.sources import SourceLimits, select_sources
from agintis.agent_kit.nodes.state import CitedSourceItem, RagState, SourceItem
from agintis.agent_kit.runtime import AgentContext, emit, streamed
from agintis.modules.agents_catalog.api import SourceRef, Sources
from agintis.platform.llm import ChatModels, ChatRole

logger = logging.getLogger(__name__)

_STRAY_SAMPLE_TOKENS = 3
_STRAY_SAMPLE_CHARS = 20

NO_SOURCES_TEXT = source_text.NO_SOURCES_TEXT
SOURCES_LEAD = "Numbered sources for the question that follows (numbering restarts every turn):"
_SOURCES_OPEN = "<sources>"
_SOURCES_CLOSE = "</sources>"
# `[3]` markers of earlier answers: numbering restarts every turn, so a stale one would be read
# as a citation of this turn's sources.
_CITATION = re.compile(r"[ \t]?\[\d{1,3}\]")


def render_source(number: int, item: SourceItem) -> str:
    """One numbered source, exactly as the model sees it (see `agent_kit.source_text`)."""
    return source_text.render_source(number, item)


def render_sources(sources: list[SourceItem]) -> str:
    """The sources in order, numbered from 1; expects the output of `select_sources`."""
    return source_text.render_sources(sources)


def strip_citations(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The messages with `[n]` markers removed from the AI ones (copies; state is untouched)."""
    cleaned: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, AIMessage):
            text = _CITATION.sub("", message.text)
            if text != message.text:
                message = message.model_copy(update={"content": text})
        cleaned.append(message)
    return cleaned


class SynthesizeNode:
    """Streams the SYNTHESIS model (the only `streamed` call of the RAG path).

    Messages sent: the system prompt ALONE (instructions only), the earlier conversation, a
    user-role message with the rendered sources, then the user's question. Retrieved text is
    untrusted data: in the system role it would carry the weight of instructions, in the user
    role it is just more user-visible context (and the prompt says to never obey it). The
    sources sit right before the question (recency, and the earlier turns stay contiguous);
    `<`/`>` are neutralized so a passage cannot fake the block's end. Earlier answers lose their
    `[n]` markers. The reply is appended to `messages` (thread memory) and its text is written
    to `answer`.

    It also emits the ONE `Sources` event, built from the very list it numbers (after
    `select_sources`) and sent right before the model call, so the client gets it before the first
    token on every path to the answer (including the rewrite loop ending without a new merge) and
    its order matches the `[n]` of the answer. Nothing is emitted when there are no sources.
    The same items are written to `cited_sources` under the reply's message id, so a transcript
    can show them again later.
    """

    def __init__(
        self,
        chat_models: ChatModels,
        prompt: str = DEFAULT_SYNTHESIZE_PROMPT,
        limits: SourceLimits = SourceLimits(),  # noqa: B008 (frozen)
        max_history_messages: int = MAX_HISTORY_MESSAGES,
    ) -> None:
        self._chat_models = chat_models
        self._prompt = prompt
        self._limits = limits
        self._max_history = max_history_messages

    def sources_message(self, rendered: str) -> HumanMessage:
        """The sources message for `render_sources(select_sources(...))`."""
        return HumanMessage(f"{SOURCES_LEAD}\n{_SOURCES_OPEN}\n{rendered}\n{_SOURCES_CLOSE}")

    def selected_sources(self, state: RagState) -> list[SourceItem]:
        return select_sources(state.get("sources") or [], self._limits)

    def prompt_messages(self, state: RagState, rendered: str) -> list[AnyMessage]:
        history = strip_citations(recent_history(state["messages"], self._max_history))
        earlier, question = history[:-1], history[-1:]
        return [
            SystemMessage(self._prompt),
            *earlier,
            self.sources_message(rendered),
            *question,
        ]

    async def __call__(self, state: RagState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        model = streamed(self._chat_models.get(ChatRole.SYNTHESIS))
        selected = self.selected_sources(state)
        cited = [
            CitedSourceItem(source=s["source"], title=s["title"], locator=s["locator"])
            for s in selected
        ]
        if cited:
            emit(
                runtime,
                Sources(tuple(SourceRef(c["source"], c["title"], c["locator"]) for c in cited)),
            )
        rendered = render_sources(selected)
        reply = await model.ainvoke(self.prompt_messages(state, rendered))
        # shortcut: detect and log only (tokens are already streamed), add a non-streaming retry
        # if mixed scripts show up in production
        stray = source_text.foreign_script_tokens(
            reply.text, f"{last_human_text(state['messages'])}\n{rendered}"
        )
        if stray:
            # Count plus a few truncated tokens: answer text is user content, keep the log small.
            sample = [t[:_STRAY_SAMPLE_CHARS] for t in stray[:_STRAY_SAMPLE_TOKENS]]
            logger.warning("Answer has %d word(s) in a foreign script, e.g. %r", len(stray), sample)
        if not reply.id:  # the sources are keyed by it
            reply = reply.model_copy(update={"id": str(uuid4())})
        update: dict[str, Any] = {"messages": [reply], "answer": reply.text}
        if cited:  # an empty write would still re-serialize the thread's whole map
            update["cited_sources"] = {reply.id: cited}
        return update
