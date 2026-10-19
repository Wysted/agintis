"""MergeSourcesNode: the sources of every routed topic as ONE numbered list.

Scores of different topics (profiles) are on different scales and not comparable, so topics are
merged by RANK, not by score, with a WEIGHTED round-robin (`interleave`), deterministic:
- groups in router order (primary topic first), the evidence kept from earlier rewrite attempts
  as the LAST group (lowest priority in each round);
- each round takes `weight` consecutive items of a group: the primary topic `primary_weight`
  (default 2), every other group 1. A plain 1:1 round-robin lets an irrelevant secondary topic
  push the right topic's hits down; the weight keeps the primary's best hits in front while a
  second topic still contributes;
- duplicates are dropped, first occurrence wins (chunk id; items without one by source, locator
  and text). Across topics this is DEFENSIVE: profiles are unique per agent spec, so two topics
  never return the same chunk; it matters for the kept previous evidence, which a new attempt
  can find again.
Then the usual `select_sources` limits cut the list.

Optional tie-break (`chat_models` given, spec `merge_tiebreak: llm`): on EVERY merge where 2+
routed topics returned sources, the ROUTER model reads each one's top `TIEBREAK_PASSAGES`
passages, labelled A, B, C... in router order (no topic names), and names the letter that
answers; that topic becomes the weighted primary, the others keep their router order. So a turn
costs up to 1 + `max_rewrites` ROUTER calls. The candidates are near-tied only in the evidence
router's normal bands (within `evidence_margin`); its `low` scope band sends up to
`low_scope_max_topics` candidates that are not, and `router: llm` has no margin at all. Why: the
evidence router's max cosine favours big profiles in near-ties and no score transform fixed it
offline; reading the passages did. The previous evidence is never a candidate. An unparsable
reply keeps the router order; provider errors propagate (like the router's). Fewer than 2
candidates: no model call.
"""

import logging
import re
import string
from collections.abc import Callable, Hashable, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agintis.agent_kit.nodes.history import last_human_text
from agintis.agent_kit.nodes.prompts import DEFAULT_MERGE_TIEBREAK_PROMPT
from agintis.agent_kit.nodes.sources import SourceLimits, select_sources
from agintis.agent_kit.nodes.state import RagState, SourceItem
from agintis.agent_kit.runtime import AgentContext
from agintis.platform.llm import ChatModels, ChatRole

logger = logging.getLogger(__name__)

# Items per round the primary topic contributes (the others contribute one).
DEFAULT_PRIMARY_WEIGHT = 2
# Tie-break: passages per candidate topic the ROUTER model reads, and their cut in characters
# (the values it was measured with).
TIEBREAK_PASSAGES = 2
TIEBREAK_PASSAGE_CHARS = 1500

type TopicSources = tuple[str, list[SourceItem]]


def interleave[T](
    groups: Sequence[Sequence[T]],
    key: Callable[[T], Hashable],
    weights: Sequence[int] | None = None,
) -> list[T]:
    """Weighted round-robin merge of ranked lists (best first each); `weights[i]` = items of
    group i per round (default 1 each). The first occurrence of a `key` wins."""
    weight = list(weights) if weights is not None else [1] * len(groups)
    if len(weight) != len(groups) or any(w < 1 for w in weight):
        raise ValueError("weights need one positive integer per group")
    merged: list[T] = []
    seen: set[Hashable] = set()
    position = [0] * len(groups)
    while any(position[i] < len(group) for i, group in enumerate(groups)):
        for i, group in enumerate(groups):
            taken = group[position[i] : position[i] + weight[i]]
            position[i] += weight[i]
            for item in taken:
                if (item_key := key(item)) not in seen:
                    seen.add(item_key)
                    merged.append(item)
    return merged


def primary_weights(groups: int, primary_weight: int) -> list[int]:
    """`primary_weight` for the first group, 1 for the rest."""
    return [primary_weight] + [1] * (groups - 1) if groups else []


def source_key(item: SourceItem) -> Hashable:
    chunk_id = item.get("chunk_id")
    if chunk_id:
        return ("chunk", chunk_id)
    return ("text", item["source"], item["locator"], item["text"])


def tiebreak_messages(query: str, candidates: Sequence[TopicSources]) -> list[BaseMessage]:
    """The tie-break prompt: the question, then each candidate's top passages under its letter."""
    # shortcut: passage delimiters are not neutralized (a chunk containing '=== Collection X ==='
    # can shift labels; the worst case is a wrong primary with the same source set), upgrade by
    # neutralizing via agent_kit render helpers if seen
    blocks = [
        f"=== Collection {letter} ===\n"
        + "\n\n".join(s["text"][:TIEBREAK_PASSAGE_CHARS] for s in sources[:TIEBREAK_PASSAGES])
        for letter, (_, sources) in zip(string.ascii_uppercase, candidates, strict=False)
    ]
    return [
        SystemMessage(DEFAULT_MERGE_TIEBREAK_PROMPT),
        HumanMessage(f"Question: {query}\n\n" + "\n\n".join(blocks)),
    ]


def merge_sources(
    results: Sequence[TopicSources],
    limits: SourceLimits,
    previous: Sequence[SourceItem] = (),
    primary_weight: int = DEFAULT_PRIMARY_WEIGHT,
) -> list[SourceItem]:
    """`results` = (topic, sources) pairs in router order (primary first); `previous` = evidence
    of earlier attempts, merged last."""
    groups = [sources for _, sources in results]
    weights = primary_weights(len(groups), primary_weight)
    if previous:
        groups.append(list(previous))
        weights.append(1)
    return select_sources(interleave(groups, source_key, weights), limits)


class MergeSourcesNode:
    """Writes `sources` (the merged, capped selection) and `previous_sources` (the same list, the
    evidence a rewrite attempt must not discard). Always rewrites `sources` (a thread keeps the
    previous turn's). The `Sources` event is NOT sent here: `SynthesizeNode` emits it from the
    list it numbers, because not every path to the answer goes through a merge.
    """

    def __init__(
        self,
        limits: SourceLimits,
        primary_weight: int = DEFAULT_PRIMARY_WEIGHT,
        chat_models: ChatModels | None = None,
    ) -> None:
        """`chat_models` turns the ROUTER-model tie-break on (None = off, no model call)."""
        if primary_weight < 1:
            raise ValueError("primary_weight must be at least 1")
        self._limits = limits
        self._primary_weight = primary_weight
        self._chat_models = chat_models

    async def _tiebreak(self, state: RagState, results: list[TopicSources]) -> list[TopicSources]:
        candidates = [r for r in results if r[1]]
        if self._chat_models is None or len(candidates) < 2:
            return results
        letters = string.ascii_uppercase[: len(candidates)]
        query = state.get("query") or last_human_text(state["messages"])
        reply = await self._chat_models.get(ChatRole.ROUTER).ainvoke(
            tiebreak_messages(query, candidates)
        )
        match = re.search(rf"\b([{letters}])\b", reply.text.strip().upper())
        if match is None:
            logger.info("Merge tie-break: unparsable reply, router order kept")
            return results
        chosen = candidates[letters.index(match.group(1))]
        reordered = [chosen, *(r for r in results if r is not chosen)]
        if chosen is not results[0]:
            logger.info(
                "Merge tie-break: %s -> %s",
                ", ".join(t for t, _ in results),
                ", ".join(t for t, _ in reordered),
            )
        return reordered

    async def __call__(self, state: RagState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
        by_topic = {r["topic"]: r["sources"] for r in state.get("topic_results") or []}
        order = [r["topic"] for r in state.get("routed_topics") or [] if r["topic"] in by_topic]
        order += [t for t in by_topic if t not in order]  # defensive: results without a route
        results = await self._tiebreak(state, [(t, by_topic[t]) for t in order])
        sources = merge_sources(
            results,
            self._limits,
            state.get("previous_sources") or [],
            self._primary_weight,
        )
        return {"sources": sources, "previous_sources": sources}
