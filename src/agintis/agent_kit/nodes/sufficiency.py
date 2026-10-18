"""Sufficiency of the merged sources (Strategy) and the query rewrite that follows an insufficient
attempt.

Why not just count: dense and hybrid retrieval ALWAYS return `top_k` hits for a non-empty profile,
so "fewer than N sources" practically never happens and a count-based loop never fires.
`RelevanceSufficiency` therefore judges the hits' own evidence (see `source_is_relevant`).
`CountSufficiency` stays as the simple placeholder (it is what the loop looked like in the first
3.5b cut and is useful with fakes).
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from agintis.agent_kit.nodes.history import last_human_text
from agintis.agent_kit.nodes.prompts import DEFAULT_REWRITE_PROMPT
from agintis.agent_kit.nodes.state import RagState, SourceItem
from agintis.agent_kit.nodes.structured import bind_structured, invoke_structured
from agintis.agent_kit.runtime import AgentContext
from agintis.modules.knowledge.api import DEFAULT_MAX_QUERY_CHARS
from agintis.platform.llm import ChatModels, ChatRole

logger = logging.getLogger(__name__)

# Defaults of the loop: one rewrite at most, one relevant source is enough.
DEFAULT_MAX_REWRITES = 1
DEFAULT_MIN_SOURCES = 1
# Cross-encoder scores (TEI rerank is a sigmoid, 0..1): relevant passages score far above this,
# unrelated ones near 0. A starting point to calibrate with the eval, not a measured value.
DEFAULT_MIN_RERANK_SCORE = 0.1
type SufficiencyKind = Literal["count", "relevance"]
DEFAULT_SUFFICIENCY: SufficiencyKind = "relevance"
type Next = Literal["rewrite", "synthesize"]
REWRITE: Next = "rewrite"
SYNTHESIZE: Next = "synthesize"


class Sufficiency(ABC):
    """Decides, after a merge, whether another (rewritten) attempt is worth it.

    Used as the routing function after the merge (`rewrite` or `synthesize`); synthesis already
    says it does not know when the sources do not cover the question, so an insufficient FINAL
    attempt still goes to it.
    """

    def __init__(self, min_sources: int, max_rewrites: int) -> None:
        if min_sources < 1:
            raise ValueError("min_sources must be at least 1")
        if max_rewrites < 0:
            raise ValueError("max_rewrites must not be negative")
        self._min_sources = min_sources
        self._max_rewrites = max_rewrites

    @abstractmethod
    def enough(self, sources: Sequence[SourceItem]) -> bool:
        """Do the merged sources answer the question well enough?"""

    def needs_rewrite(self, sources: Sequence[SourceItem], attempts: int) -> bool:
        return attempts < self._max_rewrites and not self.enough(sources)

    async def __call__(self, state: RagState) -> Next:
        rewrite = self.needs_rewrite(state.get("sources") or [], state.get("attempts") or 0)
        return REWRITE if rewrite else SYNTHESIZE


class CountSufficiency(Sufficiency):
    """Fewer than `min_sources` merged sources. Placeholder: retrieval returns `top_k` hits for
    any non-empty profile, so with real data this only fires on an empty result."""

    def __init__(
        self, min_sources: int = DEFAULT_MIN_SOURCES, max_rewrites: int = DEFAULT_MAX_REWRITES
    ) -> None:
        super().__init__(min_sources, max_rewrites)

    def enough(self, sources: Sequence[SourceItem]) -> bool:
        return len(sources) >= self._min_sources


def source_is_relevant(item: SourceItem, min_rerank_score: float) -> bool | None:
    """The hit's own relevance evidence, strongest first; None = no evidence at all.

    1. a cross-encoder score (reranker on): relevant when `rerank_score >= min_rerank_score`;
    2. no reranker (dense-only, hybrid or bm25_hybrid): no usable signal. A cosine similarity
       has no stable threshold across queries and models, and `lexical_rank` means little either
       way: with `hybrid` it says every stem of the whole question matched in one chunk
       (`plainto_tsquery` AND), which natural-language questions rarely satisfy (treating its
       absence as "irrelevant" made the rewrite loop fire on most well-answered questions);
       with `bm25_hybrid` it only says SOME query term matched, which nearly every hit does.
    """
    score = item.get("rerank_score")
    if score is not None:
        return score >= min_rerank_score
    return None


class RelevanceSufficiency(Sufficiency):
    """Sufficient when at least `min_sources` sources are relevant (`source_is_relevant`).

    - Nothing retrieved at all (every topic empty after the filter fallback ladder: empty
      corpus or profile): never rewrite, a broader query cannot help.
    - A source without any evidence (no reranker: dense or either hybrid) counts as relevant,
      so this strategy degrades to "always sufficient" there (debug log): with no signal a
      rewrite would be a coin flip that costs a model call and a second retrieval.
    - In a mixed fan-out (one topic with a reranker, another without) a no-signal source counts
      as relevant, so the topic without a reranker masks negative rerank evidence of the other.
    """

    def __init__(
        self,
        min_sources: int = DEFAULT_MIN_SOURCES,
        max_rewrites: int = DEFAULT_MAX_REWRITES,
        min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
    ) -> None:
        super().__init__(min_sources, max_rewrites)
        if not 0.0 <= min_rerank_score <= 1.0:
            raise ValueError("min_rerank_score must be between 0 and 1")
        self._min_rerank_score = min_rerank_score

    def enough(self, sources: Sequence[SourceItem]) -> bool:
        if not sources:
            return True
        verdicts = [source_is_relevant(s, self._min_rerank_score) for s in sources]
        if all(v is None for v in verdicts):
            logger.debug("No relevance signal on the sources (no reranker)")
            return True
        return sum(1 for v in verdicts if v is not False) >= self._min_sources


SUFFICIENCY_STRATEGIES: Mapping[str, Callable[[int, int, float], Sufficiency]] = {
    "count": lambda min_sources, max_rewrites, _score: CountSufficiency(min_sources, max_rewrites),
    "relevance": RelevanceSufficiency,
}


def build_sufficiency(
    kind: str,
    min_sources: int = DEFAULT_MIN_SOURCES,
    max_rewrites: int = DEFAULT_MAX_REWRITES,
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
) -> Sufficiency:
    try:
        factory = SUFFICIENCY_STRATEGIES[kind]
    except KeyError:
        raise ValueError(
            f"Unknown sufficiency {kind!r} (known: {sorted(SUFFICIENCY_STRATEGIES)})"
        ) from None
    return factory(min_sources, max_rewrites, min_rerank_score)


class QueryRewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="A different search query for the same question.")


def _delimited(text: str) -> str:
    """User text with markup neutralized, so it cannot close or fake a tag of the prompt."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


class RewriteQueryNode:
    """Uses the ANALYZER model (internal output, never `streamed`) to write a broader or
    alternative search query, replaces `query` with it (cut to `max_query_chars`) and counts the
    attempt. The question and the previous query go in delimited fields: user text cannot spoof
    the label of the other one.

    Degrades (unparseable or empty output) to the PREVIOUS query. A rewrite that leaves the query
    unchanged would repeat the same search, so the node then goes straight to `fallback_node`
    (synthesis) instead of `retry_node` (route): the loop ends instead of burning attempts.
    Provider and network errors propagate.
    """

    def __init__(
        self,
        chat_models: ChatModels,
        prompt: str = DEFAULT_REWRITE_PROMPT,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        retry_node: str = "route",
        fallback_node: str = "synthesize",
    ) -> None:
        if max_query_chars < 1:
            raise ValueError("max_query_chars must be at least 1")
        self._chat_models = chat_models
        self._prompt = prompt
        self._max_query_chars = max_query_chars
        self._retry_node = retry_node
        self._fallback_node = fallback_node
        self._structured: Runnable[Any, Any] | None = None  # bound lazily, once

    async def __call__(self, state: RagState, runtime: Runtime[AgentContext]) -> Command[Any]:
        previous = state.get("query") or last_human_text(state["messages"])
        original = state.get("standalone_query") or previous
        if self._structured is None:
            self._structured = bind_structured(
                self._chat_models.get(ChatRole.ANALYZER), QueryRewrite
            )
        outcome = await invoke_structured(
            self._structured,
            [
                SystemMessage(self._prompt),
                HumanMessage(
                    f"<question>{_delimited(original)}</question>\n"
                    f"<previous_query>{_delimited(previous)}</previous_query>\n"
                    "It returned too few results. Write a different search query."
                ),
            ],
        )
        query = previous
        if isinstance(outcome.parsed, QueryRewrite) and outcome.parsed.query.strip():
            query = outcome.parsed.query.strip()[: self._max_query_chars]
        changed = query != previous
        if not changed:
            logger.info("Query rewrite left the query unchanged: ending the rewrite loop")
        return Command(
            update={"query": query, "attempts": (state.get("attempts") or 0) + 1},
            goto=self._retry_node if changed else self._fallback_node,
        )
