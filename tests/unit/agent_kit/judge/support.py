"""Builders shared by the judge tests."""

import threading
import time
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import BaseModel

from agintis.agent_kit.judge import AnswerRecord, CitedSource, ScopeVerdict
from agintis.agent_kit.judge.llm import (
    CitationSupportOutput,
    ClaimCheck,
    FaithfulnessOutput,
    RelevanceOutput,
    SentenceCheck,
)
from tests.fakes.llm import ScriptedChatModel

SOURCES = (
    CitedSource(1, "Norma A", "file:///a.pdf", "El plazo es de 24 meses.", "p. 3"),
    CitedSource(2, None, "file:///b.pdf", "La tasa es del 10%."),
)


def make_record(
    answer: str = "El plazo es de 24 meses [1]. La tasa es del 10% [2].",
    *,
    id: str = "q1",
    expected_scope: Any = "in",
    scope: ScopeVerdict | None = None,
    sources: tuple[CitedSource, ...] = SOURCES,
    error: str | None = None,
    refusal_reason: str | None = None,
) -> AnswerRecord:
    return AnswerRecord(
        id=id,
        question="¿Cuál es el plazo?",
        answer=answer,
        agent="rag",
        expected_scope=expected_scope,
        scope=scope,
        sources=sources,
        models={"synthesis": "default (openai:m)"},
        meta={"git": "abc"},
        error=error,
        refusal_reason=refusal_reason,
        citations="inline",
    )


# (answer span, source number, evidence span) of the default record: verbatim on both sides.
_CLAIM_PIECES = (
    ("El plazo es de 24 meses", 1, "El plazo es de 24 meses."),
    ("La tasa es del 10%", 2, "La tasa es del 10%."),
)
_COMPARISON = "polaridad igual; condición igual; alcance igual; cifras iguales; actor igual"
_SENTENCES = ("El plazo es de 24 meses [1].", "La tasa es del 10% [2].")


def claim(label: bool | str = True, position: int = 1, **overrides: Any) -> ClaimCheck:
    """A claim over the default record (alternating its two facts); bool -> supported or
    contradicted. Every field can be overridden."""
    quote, source_n, evidence = _CLAIM_PIECES[(position - 1) % 2]
    fields: dict[str, Any] = {
        "claim_quote": quote,
        "source_ns": [source_n],
        "evidence_quote": evidence,
        "comparison": _COMPARISON,
        "label": ("supported" if label else "contradicted") if isinstance(label, bool) else label,
    }
    return ClaimCheck.model_validate({**fields, **overrides})


def claims(*labels: bool | str, is_refusal: bool = False) -> FaithfulnessOutput:
    return FaithfulnessOutput(
        claims=[claim(label, i) for i, label in enumerate(labels, start=1)],
        is_refusal=is_refusal,
    )


def sentence_check(label: bool | str = True, index: int = 1, **overrides: Any) -> SentenceCheck:
    fields: dict[str, Any] = {
        "index": index,
        "sentence_quote": _SENTENCES[(index - 1) % 2],
        "cited": [(index - 1) % 2 + 1],
        "evidence_quote": _CLAIM_PIECES[(index - 1) % 2][2],
        "comparison": _COMPARISON,
        "states_missing_info": False,
        "label": ("supported" if label else "not_supported") if isinstance(label, bool) else label,
    }
    return SentenceCheck.model_validate({**fields, **overrides})


def checks(*labels: bool | str, indexes: list[int] | None = None) -> CitationSupportOutput:
    idx = indexes or list(range(1, len(labels) + 1))
    return CitationSupportOutput(
        checks=[sentence_check(label, i) for i, label in zip(idx, labels, strict=True)]
    )


def relevance(verdict: bool, is_refusal: bool = False) -> RelevanceOutput:
    return RelevanceOutput(is_refusal=is_refusal, verdict=verdict, rationale="rationale")


def judge_model(*answers: BaseModel | str) -> ScriptedChatModel:
    """A judge model that pops one structured answer per call, whatever the schema."""
    return ScriptedChatModel(structured=list(answers))


class TrackingModel(ScriptedChatModel):
    """Answers every structured call with `value`, tracking the calls in flight and the run
    tags / metadata each call carried."""

    value: Any = None
    delay: float = 0.0
    lock: Any = None
    state: Any = None  # {"now": int, "peak": int, "runs": list[dict]}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        with self.lock:
            self.state["now"] += 1
            self.state["peak"] = max(self.state["peak"], self.state["now"])
            if run_manager is not None:
                self.state["runs"].append(
                    {"tags": list(run_manager.tags), "metadata": dict(run_manager.metadata)}
                )
        try:
            time.sleep(self.delay)
            return super()._generate(messages, stop, run_manager, **kwargs)
        finally:
            with self.lock:
                self.state["now"] -= 1


def tracking_model(value: BaseModel, delay: float = 0.0) -> TrackingModel:
    state: dict[str, Any] = {"now": 0, "peak": 0, "runs": []}
    return TrackingModel(
        responder=lambda _schema, _messages: value,
        value=value,
        delay=delay,
        lock=threading.Lock(),
        state=state,
    )
