"""What a judge returns: a status, a binary verdict, a rationale and the cost of the call."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class JudgeName(StrEnum):
    FAITHFULNESS = "faithfulness"
    CITATION_SUPPORT = "citation_support"
    ANSWER_RELEVANCE = "answer_relevance"
    REFUSAL_CORRECTNESS = "refusal_correctness"
    SCRIPT_MIX = "script_mix"


class JudgeStatus(StrEnum):
    OK = "ok"  # `verdict` is set
    NOT_APPLICABLE = "not_applicable"  # the judge does not apply to this record
    ERROR = "error"  # the judge could not decide (provider failure, unparseable output...)
    RUN_FAILED = "run_failed"  # the agent run behind the record failed: nothing to judge


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage of the model calls behind a verdict (zero for deterministic ones)."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.total_tokens + other.total_tokens,
            self.calls + other.calls,
        )

    def to_json(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


@dataclass(frozen=True, slots=True)
class JudgeResult:
    judge: JudgeName
    status: JudgeStatus
    verdict: bool | None
    rationale: str
    # "llm" | "deterministic": which way the verdict was reached.
    method: str
    # Why it is not ok, or why a decided verdict is what it is (counted in the aggregates):
    # not_applicable -> "refusal" | "model_refusal" | "no_citations" | "citations_disabled" (the
    # agent's `citations: none`) | "no_expected_scope" | "empty_answer"; error -> "no_claims" |
    # "judge_error" | ...; run_failed -> "run_failed";
    # ok -> the guard code of a failed verdict ("claim_not_verbatim", "in_scope_refused"...).
    code: str | None = None
    # Judge-specific evidence (per-claim list, per-sentence list, offending markers...).
    details: Mapping[str, Any] = field(default_factory=dict)
    usage: Usage = Usage()

    @classmethod
    def decided(
        cls,
        judge: JudgeName,
        verdict: bool,
        rationale: str,
        method: str,
        details: Mapping[str, Any] | None = None,
        usage: Usage = Usage(),  # noqa: B008 (frozen)
        *,
        code: str | None = None,
    ) -> "JudgeResult":
        return cls(
            judge,
            JudgeStatus.OK,
            verdict,
            rationale,
            method,
            code,
            details=details or {},
            usage=usage,
        )

    @classmethod
    def not_applicable(
        cls,
        judge: JudgeName,
        reason: str,
        code: str,
        method: str = "deterministic",
        usage: Usage = Usage(),  # noqa: B008 (frozen)
        details: Mapping[str, Any] | None = None,
    ) -> "JudgeResult":
        return cls(
            judge,
            JudgeStatus.NOT_APPLICABLE,
            None,
            reason,
            method,
            code,
            details=details or {},
            usage=usage,
        )

    @classmethod
    def failed(
        cls,
        judge: JudgeName,
        reason: str,
        usage: Usage = Usage(),  # noqa: B008 (frozen)
        *,
        method: str = "llm",
        code: str = "judge_error",
    ) -> "JudgeResult":
        return cls(judge, JudgeStatus.ERROR, None, reason, method, code, usage=usage)

    @classmethod
    def run_failed(cls, judge: JudgeName, reason: str) -> "JudgeResult":
        """The record has no answer because the agent run failed (not the judge's failure)."""
        return cls(judge, JudgeStatus.RUN_FAILED, None, reason, "deterministic", "run_failed")

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "method": self.method,
            "code": self.code,
            "details": dict(self.details),
            "usage": self.usage.to_json(),
        }
