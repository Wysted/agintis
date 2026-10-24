"""Run a set of judges over records: error isolation, subset selection and aggregation."""

import asyncio
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from langchain_core.language_models import BaseChatModel

from agintis.agent_kit.judge.deterministic import RefusalCorrectnessJudge, ScriptMixJudge
from agintis.agent_kit.judge.llm import (
    AnswerRelevanceJudge,
    CitationSupportJudge,
    FaithfulnessJudge,
    JudgeCall,
    JudgeCallError,
    JudgeOutputError,
)
from agintis.agent_kit.judge.ports import AnswerJudge, SignalAwareJudge
from agintis.agent_kit.judge.records import AnswerRecord
from agintis.agent_kit.judge.results import JudgeName, JudgeResult, JudgeStatus, Usage
from agintis.agent_kit.usage import ModelUsageCounter

logger = logging.getLogger(__name__)


def build_judges(
    model: BaseChatModel,
    model_label: str,
    names: Sequence[JudgeName] | None = None,
    *,
    semaphore: asyncio.Semaphore | None = None,
    run_usage: ModelUsageCounter | None = None,
) -> list[AnswerJudge]:
    """The requested judges (default: all of them) in the canonical order, over one judge model.

    `semaphore` bounds the model calls in flight across ALL the judges of a run; `run_usage`
    collects their usage per model.
    """
    wanted = set(names or JudgeName)
    call = JudgeCall(model, model_label, semaphore, run_usage)
    available: dict[JudgeName, AnswerJudge] = {
        JudgeName.FAITHFULNESS: FaithfulnessJudge(call),
        JudgeName.CITATION_SUPPORT: CitationSupportJudge(call),
        JudgeName.ANSWER_RELEVANCE: AnswerRelevanceJudge(call),
        JudgeName.REFUSAL_CORRECTNESS: RefusalCorrectnessJudge(),
        JudgeName.SCRIPT_MIX: ScriptMixJudge(),
    }
    return [judge for name, judge in available.items() if name in wanted]


def failure_reason(exc: BaseException) -> str:
    """A safe description of a failure for a report: class (and HTTP status), never the text
    (provider messages may echo request details)."""
    if isinstance(exc, JudgeOutputError):  # our own message
        return str(exc)
    if isinstance(exc, JudgeCallError) and exc.__cause__ is not None:
        exc = exc.__cause__
    status = getattr(exc, "status_code", None)
    return type(exc).__name__ + (f" (status {status})" if isinstance(status, int) else "")


async def _safely(
    judge: AnswerJudge, record: AnswerRecord, model_refusal: bool | None = None
) -> JudgeResult:
    try:
        if isinstance(judge, SignalAwareJudge):
            return await judge.judge_with(record, model_refusal)
        return await judge.judge(record)
    except Exception as exc:  # one failing judge or record never aborts a run
        logger.warning(
            "judge %s failed on %s: %s", judge.name.value, record.id, failure_reason(exc)
        )
        usage = exc.usage if isinstance(exc, JudgeOutputError | JudgeCallError) else Usage()
        return JudgeResult.failed(judge.name, failure_reason(exc), usage, method=judge.method)


def _flag(results: Iterable[JudgeResult], judge: JudgeName) -> bool | None:
    for r in results:
        if (
            r.judge is judge
            and "heuristic_refusal" in r.details
            and "model_is_refusal" in r.details
        ):
            return bool(r.details["model_is_refusal"])
    return None


def model_refusal_signal(
    results: Iterable[JudgeResult],
) -> tuple[bool | None, str | None, bool | None]:
    """`(signal, source, disagree)`: what the judge model said about `is_refusal`.

    The signal is `answer_relevance`'s (it runs for every non-empty answer, so the signal does
    not depend on whether the answer happened to have claims); faithfulness' (the raw flag, also
    when it comes with claims or citations: v8 judges the topic, not the claim count) is the
    fallback only when relevance did not run or errored. `disagree` says whether both reported
    and differed (None if not both did); never an OR across judges. `(None, None, None)` when
    neither reported.
    """
    results = list(results)
    relevance = _flag(results, JudgeName.ANSWER_RELEVANCE)
    faithfulness = _flag(results, JudgeName.FAITHFULNESS)
    disagree = None if relevance is None or faithfulness is None else relevance != faithfulness
    if relevance is not None:
        return relevance, JudgeName.ANSWER_RELEVANCE.value, disagree
    if faithfulness is not None:
        return faithfulness, JudgeName.FAITHFULNESS.value, disagree
    return None, None, None


async def judge_record(
    record: AnswerRecord, judges: Sequence[AnswerJudge]
) -> dict[JudgeName, JudgeResult]:
    """Every judge over one record; an exception becomes an `error` result.

    The judges that need no other judge's output run concurrently first. The signal-aware ones
    (`refusal_correctness`) then run with what the judge model said about refusals
    (`model_refusal_signal`), so their verdict can be `hybrid`; if the LLM judges were not
    requested the signal is None and they stay deterministic.

    A dump record whose run failed (`record.error`) has no answer to judge: every judge reports
    `run_failed` with that reason (counted apart from judge errors), without calling the model.
    """
    if record.error is not None:
        return {
            j.name: JudgeResult.run_failed(j.name, f"agent run failed: {record.error}")
            for j in judges
        }
    independent = [j for j in judges if not isinstance(j, SignalAwareJudge)]
    dependent = [j for j in judges if isinstance(j, SignalAwareJudge)]
    first = await asyncio.gather(*(_safely(judge, record) for judge in independent))
    signal, source, disagree = model_refusal_signal(first)
    second = [
        replace(
            result,
            details={**result.details, "signal_source": source, "llm_judges_disagree": disagree},
        )
        if result.status is JudgeStatus.OK
        else result
        for result in [await _safely(judge, record, signal) for judge in dependent]
    ]
    results = {result.judge: result for result in [*first, *second]}
    return {j.name: results[j.name] for j in judges}  # the requested order


def _refusal_signals(results: Sequence[JudgeResult]) -> dict[str, int] | None:
    """How the judge model's `is_refusal` and the text heuristic compare (None for judges that
    do not ask the model). `model_only` / `heuristic_only` are the disagreements to audit; since v8
    `model_only` also counts every CITED refusal (the heuristic says "not a refusal" for any cited
    answer), so it is not by itself a judge-error signal."""
    compared = [r.details for r in results if "heuristic_refusal" in r.details]
    if not compared:
        return None
    model = [d["model_is_refusal"] for d in compared]
    heuristic = [d["heuristic_refusal"] for d in compared]
    pairs = list(zip(model, heuristic, strict=True))
    return {
        "compared": len(compared),
        "both": sum(1 for m, h in pairs if m and h),
        "model_only": sum(1 for m, h in pairs if m and not h),
        "heuristic_only": sum(1 for m, h in pairs if h and not m),
        "neither": sum(1 for m, h in pairs if not m and not h),
    }


def _heuristic_verdicts(results: Sequence[JudgeResult]) -> dict[str, Any] | None:
    """The `refusal_correctness` pass rate the text heuristic ALONE (with the scope verdict)
    would give, next to the hybrid verdict that includes the judge model's signal. None for the
    other judges."""
    decided = [
        r.details["heuristic_verdict"]
        for r in results
        if r.status is JudgeStatus.OK and "heuristic_verdict" in r.details
    ]
    if not decided:
        return None
    passed = sum(1 for v in decided if v)
    return {
        "decided": len(decided),
        "passed": passed,
        "pass_rate": passed / len(decided),
    }


def _model_signal(results: Sequence[JudgeResult]) -> dict[str, Any] | None:
    """For `refusal_correctness`: where the model signal came from, how often the two LLM judges
    that report `is_refusal` disagreed, and how many decided records fell back to the
    deterministic method (no model signal), so the two methods are never mixed silently."""
    decided = [r for r in results if r.status is JudgeStatus.OK and "signal_source" in r.details]
    if not decided:
        return None
    sources = Counter(r.details["signal_source"] for r in decided)
    compared = [r for r in decided if r.details["llm_judges_disagree"] is not None]
    return {
        "decided": len(decided),
        "hybrid": sum(1 for r in decided if r.method == "hybrid"),
        "deterministic_fallbacks": sum(1 for r in decided if r.method != "hybrid"),
        "sources": {str(k): v for k, v in sorted(sources.items(), key=lambda kv: str(kv[0]))},
        "relevance_vs_faithfulness": {
            "compared": len(compared),
            "disagree": sum(1 for r in compared if r.details["llm_judges_disagree"]),
        },
    }


def _model_refusals(results: Sequence[JudgeResult]) -> dict[str, int] | None:
    """Faithfulness `model_refusal` results (n/a): how many cite a source and how many list
    claims with no citation, which NO judge checks (`unchecked_claims`). None when there are none
    (answer_relevance also codes `model_refusal` but carries no claims, so it is skipped)."""
    flagged = [
        r.details for r in results if r.code == "model_refusal" and "claims_listed" in r.details
    ]
    if not flagged:
        return None
    return {
        "total": len(flagged),
        "cited_refusal": sum(1 for d in flagged if d["cited_refusal"]),
        "unchecked_claims": sum(
            1 for d in flagged if d["claims_listed"] > 0 and not d["cited_refusal"]
        ),
    }


def _sum_counts(results: Sequence[JudgeResult], key: str) -> dict[str, int] | None:
    """Totals of a `{name: count}` details entry (`retries`, `guard`) over the results that carry
    it; None when none does."""
    carried = [r.details[key] for r in results if isinstance(r.details.get(key), dict)]
    if not carried:
        return None
    total: Counter[str] = Counter()
    for counts in carried:
        total.update(counts)
    return dict(sorted(total.items()))


def _uncovered_protected(results: Sequence[JudgeResult]) -> dict[str, int] | None:
    """Report-only: the records whose claim quotes left negations / numbers / quantifiers of the
    answer uncovered, and how many tokens in total (None for judges that do not report it)."""
    counts = [
        r.details["uncovered_protected"] for r in results if "uncovered_protected" in r.details
    ]
    if not counts:
        return None
    return {"records": sum(1 for c in counts if c), "tokens": sum(counts)}


def aggregate(
    judgments: Sequence[Mapping[JudgeName, JudgeResult]], names: Sequence[JudgeName]
) -> dict[str, dict[str, Any]]:
    """Per judge: counts by status, the reasons behind the non-ok ones, pass rate over the
    decided ones and token usage. `errors` are judge failures only; `run_failed` (the agent run
    behind the record failed) is counted apart. `reasons` counts every `JudgeResult.code`
    (e.g. `no_claims`: the faithfulness errors where the model listed no claim). `retries`
    (parse / verbatim) and `verbatim_guard` (claims and evidence quotes that were not verbatim)
    count the judge's self-corrections."""
    summary: dict[str, dict[str, Any]] = {}
    for name in names:
        results = [j[name] for j in judgments if name in j]
        decided = [r for r in results if r.status is JudgeStatus.OK]
        passed = sum(1 for r in decided if r.verdict)
        usage = sum((r.usage for r in results), Usage())
        codes = Counter(r.code for r in results if r.code is not None)
        summary[name.value] = {
            "records": len(results),
            "decided": len(decided),
            "passed": passed,
            "failed": len(decided) - passed,
            "pass_rate": passed / len(decided) if decided else None,
            "not_applicable": sum(1 for r in results if r.status is JudgeStatus.NOT_APPLICABLE),
            "errors": sum(1 for r in results if r.status is JudgeStatus.ERROR),
            "run_failed": sum(1 for r in results if r.status is JudgeStatus.RUN_FAILED),
            "reasons": dict(sorted(codes.items())),
            "refusal_signals": _refusal_signals(results),
            "model_refusals": _model_refusals(results),
            "refusal_correctness_heuristic": _heuristic_verdicts(results),
            "refusal_model_signal": _model_signal(results),
            "retries": _sum_counts(results, "retries"),
            "verbatim_guard": _sum_counts(results, "guard"),
            "uncovered_protected": _uncovered_protected(results),
            "usage": usage.to_json(),
        }
    return summary
