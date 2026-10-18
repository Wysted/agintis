"""Build an `AnswerRecord` from the final state of a RAG graph run."""

import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage

from agintis.agent_kit.judge.records import (
    AnswerRecord,
    CitedSource,
    ExpectedScope,
    ScopeVerdict,
)
from agintis.agent_kit.nodes import CitationStyle, RagState, SourceLimits, select_sources

_SURROGATES = re.compile("[\ud800-\udfff]")


def scrub(text: str) -> str:
    """Lone surrogates (broken model output) replaced by U+FFFD: they cannot be written as UTF-8."""
    return _SURROGATES.sub("\ufffd", text)


def final_answer(state: Mapping[str, Any]) -> str:
    """The answer text of a finished turn (`answer`, else the last AI message)."""
    answer = state.get("answer")
    if isinstance(answer, str) and answer:
        return answer
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage):
            return message.text
    return ""


def record_from_state(
    state: RagState | Mapping[str, Any],
    *,
    id: str,
    question: str,
    agent: str,
    limits: SourceLimits,
    expected_scope: ExpectedScope | None = None,
    models: Mapping[str, str] | None = None,
    meta: Mapping[str, Any] | None = None,
    citations: CitationStyle,
) -> AnswerRecord:
    """The record of one finished run.

    The sources are the SAME selection `SynthesizeNode` numbered and showed the model
    (`select_sources` with the agent's limits over the merged `sources`, text cut included), so
    `[n]` in the answer is source `n` here. A refused turn has none.
    """
    selected = select_sources(state.get("sources") or [], limits)
    level, score = state.get("scope_level"), state.get("scope_score")
    scope = (
        ScopeVerdict(level, None if score is None else float(score)) if level is not None else None
    )
    return AnswerRecord(
        id=id,
        question=question,
        answer=scrub(final_answer(state)),
        agent=agent,
        expected_scope=expected_scope,
        scope=scope,
        sources=tuple(
            CitedSource(
                n=n,
                title=None if item["title"] is None else scrub(item["title"]),
                uri=scrub(item["source"]),
                text=scrub(item["text"]),
                locator=None if item["locator"] is None else scrub(item["locator"]),
            )
            for n, item in enumerate(selected, start=1)
        ),
        models=dict(models or {}),
        meta=dict(meta or {}),
        refusal_reason=state.get("refusal_reason"),
        citations=citations,
    )
