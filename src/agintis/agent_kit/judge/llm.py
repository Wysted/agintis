"""LLM implementations of the answer judges (structured output, Spanish prompts).

Faithfulness, citation support and answer relevance. Each one gates itself deterministically first
(not applicable, or a failure that needs no model), asks the judge model ONE structured question
and computes the aggregate verdict in code: the model lists per-claim / per-sentence checks (with
VERBATIM quotes, backed by the deterministic guard of `guard.py`) and the verdict is
`all(supported)`, never the model's own summary.

The model is any `BaseChatModel` that supports structured output (the judge profile of
`EVAL_JUDGE_PROFILE`); a call is bound once with `bind_structured` (a non-streamed copy) and runs
under a `JudgeCall`, which bounds concurrency, counts tokens and tags the run for the tracer.
"""

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from agintis.agent_kit import source_text
from agintis.agent_kit.judge import citations, guard, prompts, refusal
from agintis.agent_kit.judge.citations import (
    CitedSentence,
    citation_numbers,
    cited_sentences,
    dangling_citations,
)
from agintis.agent_kit.judge.guard import match_evidence, match_quote, match_sentence
from agintis.agent_kit.judge.records import AnswerRecord, CitedSource
from agintis.agent_kit.judge.refusal import (
    looks_like_refusal,
    refused_by_scope,
    states_missing_information,
)
from agintis.agent_kit.judge.render import (
    citation_messages,
    faithfulness_messages,
    relevance_messages,
)
from agintis.agent_kit.judge.results import JudgeName, JudgeResult, Usage
from agintis.agent_kit.judge.usage import response_usage
from agintis.agent_kit.nodes.structured import bind_structured, invoke_structured

JUDGE_TAG = "agintis:judge"
JUDGE_COMPONENT = "answer_judge"
MAX_DETAIL_CHARS = 400
MAX_PARSE_ATTEMPTS = 2  # one retry on a schema parse failure
MAX_QUOTES_IN_NOTE = 5  # offending quotes echoed in the corrective message
UNVERIFIABLE = "unverifiable"  # label of a claim / sentence whose quote is not in the answer
CLAIM_NOT_VERBATIM = "claim_not_verbatim"
EVIDENCE_NOT_VERBATIM = "evidence_not_verbatim"
INVALID_SOURCE_NS = "invalid_source_ns"


class JudgeOutputError(RuntimeError):
    """The judge model answered something that cannot be used (unparseable output). Carries the
    tokens the call used, so a run's usage totals stay honest."""

    def __init__(self, message: str, usage: Usage = Usage()) -> None:  # noqa: B008 (frozen)
        super().__init__(message)
        self.usage = usage


class JudgeCallError(RuntimeError):
    """The judge model call failed (provider / network); `__cause__` is the original error. Carries
    the usage of the failed call so the run's totals count it."""

    def __init__(self, usage: Usage) -> None:
        super().__init__("the judge model call failed")
        self.usage = usage


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimCheck(_Strict):
    """One claim of the answer. The field ORDER is the order the model writes them in: it copies
    first, compares second and only then labels."""

    claim_quote: str = Field(
        description="VERBATIM contiguous span of the answer with the full claim (condition and "
        "qualifiers included). Never paraphrased."
    )
    source_ns: list[int] = Field(
        description="Numbers of the sources compared (they may support the claim jointly); "
        "empty if none."
    )
    evidence_quote: str = Field(
        description="VERBATIM span(s) of those sources that support or contradict it, "
        "non-contiguous fragments joined with '...'; empty if none."
    )
    comparison: str = Field(
        description="Short axis-by-axis check: polarity, condition/time, scope/quantifier, "
        "numbers/dates, entity/actor."
    )
    label: Literal["supported", "contradicted", "not_found"]


class FaithfulnessOutput(_Strict):
    claims: list[ClaimCheck] = Field(description="Every verifiable claim of the answer, checked.")
    is_refusal: bool = Field(
        description="True iff the answer gives no information on the topic asked (see the "
        "refusal definition in the instructions)."
    )


class SentenceCheck(_Strict):
    index: int = Field(description="The index of the sentence, as given.")
    sentence_quote: str = Field(description="VERBATIM sentence, copied from the answer.")
    cited: list[int] = Field(description="Numbers of the sources the sentence cites.")
    evidence_quote: str = Field(
        description="VERBATIM span(s) of the cited source(s), fragments joined with '...'; "
        "empty if none."
    )
    comparison: str = Field(
        description="Short axis-by-axis check: polarity, condition/time, scope/quantifier, "
        "numbers/dates, entity/actor."
    )
    states_missing_info: bool = Field(
        description="True only if the sentence just says that information is missing from the "
        "sources (then the evidence may be empty)."
    )
    label: Literal["supported", "partial", "not_supported"]


class CitationSupportOutput(_Strict):
    checks: list[SentenceCheck] = Field(description="Exactly one entry per sentence given.")


class RelevanceOutput(_Strict):
    is_refusal: bool = Field(
        description="True iff the answer gives no information on the topic asked (see the "
        "refusal definition in the instructions)."
    )
    verdict: bool = Field(description="True if the answer addresses the question asked.")
    rationale: str = Field(description="One or two sentences.")


class UsageCounter(BaseCallbackHandler):
    """Sums the token usage the chat model reports (works for any model, with or without a name)."""

    run_inline = True

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.calls += 1  # a failed call still hit the provider (and may have been billed)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        call = response_usage(response)
        self.input_tokens += call.input_tokens
        self.output_tokens += call.output_tokens
        self.total_tokens += call.total_tokens
        self.calls += 1

    @property
    def usage(self) -> Usage:
        return Usage(self.input_tokens, self.output_tokens, self.total_tokens, self.calls)


@dataclass(frozen=True, slots=True)
class Asked[S]:
    """What `JudgeCall.ask` returns: the parsed output, the usage of ALL attempts and how many
    parse retries it took (0 or 1)."""

    parsed: S
    usage: Usage
    parse_retries: int = 0


class JudgeCall:
    """The one way the LLM judges talk to the model.

    Binds the structured schema once per judge, limits the calls in flight (`semaphore`, shared by
    the whole run) and tags every run so judge traces are distinguishable in the tracer: tag
    `agintis:judge`, run name `judge.<name>` and metadata {agintis_component, judge, record_id,
    judged_agent, judge_model}. `model_label` is the "profile (provider:model)" label.
    """

    def __init__(
        self,
        model: BaseChatModel,
        model_label: str,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._model = model
        self._label = model_label
        self._semaphore = semaphore or asyncio.Semaphore(1)
        self._bound: dict[type[BaseModel], Runnable[Any, Any]] = {}

    async def ask[S: BaseModel](
        self,
        judge: JudgeName,
        record: AnswerRecord,
        schema: type[S],
        messages: Sequence[BaseMessage],
    ) -> "Asked[S]":
        """The parsed answer, its usage and the parse retries it took.

        An output that does not parse is retried ONCE with the same messages plus a short
        corrective note; the usage of every attempt is counted. `JudgeOutputError` when it still
        does not parse and `JudgeCallError` (cause = the provider error) when a call fails."""
        runnable = self._bound.get(schema)
        if runnable is None:
            runnable = self._bound[schema] = bind_structured(self._model, schema)
        total = Usage()
        for attempt in range(MAX_PARSE_ATTEMPTS):
            sent = list(messages)
            if attempt:
                sent.append(HumanMessage(prompts.PARSE_RETRY_NOTE))
            try:
                parsed, usage = await self._once(judge, record, schema, runnable, sent)
            except JudgeOutputError as exc:
                total += exc.usage
                if attempt == MAX_PARSE_ATTEMPTS - 1:
                    raise JudgeOutputError(str(exc), total) from None
                continue
            except JudgeCallError as exc:
                raise JudgeCallError(total + exc.usage) from exc.__cause__
            return Asked(parsed, total + usage, attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _once[S: BaseModel](
        self,
        judge: JudgeName,
        record: AnswerRecord,
        schema: type[S],
        runnable: Runnable[Any, Any],
        messages: Sequence[BaseMessage],
    ) -> tuple[S, Usage]:
        counter = UsageCounter()
        config: RunnableConfig = {
            "callbacks": [counter],
            "run_name": f"judge.{judge.value}",
            "tags": [JUDGE_TAG, f"judge:{judge.value}"],
            "metadata": {
                "agintis_component": JUDGE_COMPONENT,
                "judge": judge.value,
                "record_id": record.id,
                "judged_agent": record.agent,
                "judge_model": self._label,
            },
        }
        try:
            async with self._semaphore:
                outcome = await invoke_structured(runnable, messages, config)
        except Exception as exc:
            raise JudgeCallError(counter.usage) from exc
        if not isinstance(outcome.parsed, schema):
            raise JudgeOutputError(
                "the judge model output did not parse into the schema", counter.usage
            )
        return outcome.parsed, counter.usage


def refusal_signals(out_is_refusal: bool, record: AnswerRecord) -> dict[str, bool]:
    """The two refusal signals side by side (judge model vs text heuristic), kept in the result
    details so the aggregates can report how often they disagree."""
    return {
        "model_is_refusal": out_is_refusal,
        "heuristic_refusal": looks_like_refusal(record.answer, citations=record.citations),
    }


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_DETAIL_CHARS else text[: MAX_DETAIL_CHARS - 1] + "…"


_SYNTHETIC_RECORD = AnswerRecord(
    id="meta",
    question="¿Qué plazo fija el reglamento ficticio?",
    answer=(
        "El reglamento fija un plazo de diez horas [1]. No se admiten prórrogas [2]. "
        "Usa `a[3]` para el plazo [1]. Por ejemplo:\n```python\nplazo = 10\n```\n[1]"
    ),
    agent="meta",
    sources=(
        CitedSource(1, "Reglamento A", "file:///a", "El plazo es de diez horas.", "art. 1"),
        CitedSource(2, None, "file:///b", "No se admiten prórrogas salvo fuerza mayor."),
    ),
    citations="inline",
)


def _rendered_messages() -> list[list[str]]:
    """The messages the judges build for a fixed synthetic record (layout, tags, order)."""
    record = _SYNTHETIC_RECORD
    batches = [
        faithfulness_messages(prompts.FAITHFULNESS_PROMPT, record),
        citation_messages(prompts.CITATION_SUPPORT_PROMPT, record, cited_sentences(record.answer)),
        relevance_messages(prompts.ANSWER_RELEVANCE_PROMPT, record),
    ]
    return [[f"{m.type}:{m.text}" for m in batch] for batch in batches]


def judge_meta() -> dict[str, str]:
    """What makes two judging runs comparable: the hand-bumped `judge_version` and a hash of the
    prompts, the corrective notes, the output schemas, the guard parameters (`guard.parameters`),
    the sentence splitter's abbreviations, the refusal heuristic (`refusal.parameters`), the
    script-mix rule (`source_text.script_parameters`) and the
    messages rendered for a fixed synthetic record (report meta)."""
    payload = json.dumps(
        {
            "prompts": [
                prompts.FAITHFULNESS_PROMPT,
                prompts.CITATION_SUPPORT_PROMPT,
                prompts.ANSWER_RELEVANCE_PROMPT,
                prompts.PARSE_RETRY_NOTE,
                prompts.VERBATIM_RETRY_NOTE,
            ],
            "schemas": [
                m.model_json_schema()
                for m in (FaithfulnessOutput, CitationSupportOutput, RelevanceOutput)
            ],
            "guard": guard.parameters(),
            "abbreviations": sorted(citations.ABBREVIATIONS),
            "refusal": refusal.parameters(),
            "script": source_text.script_parameters(),
            "judges": [name.value for name in JudgeName],
            "rendered": _rendered_messages(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "judge_version": prompts.JUDGE_VERSION,
        "judge_prompts_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _source_texts(record: AnswerRecord, numbers: Sequence[int]) -> list[str]:
    """Everything the judge saw of the named sources: the passage text and the rendered block
    (`[n] title (locator)` header + text), so evidence may quote the title or the locator."""
    by_number: Mapping[int, CitedSource] = {s.n: s for s in record.sources}
    return [
        t for n in numbers if n in by_number for t in (by_number[n].text, by_number[n].render())
    ]


GUARD_FAILURES = ("claims_not_verbatim", "invalid_source_ns", "evidence_not_verbatim")
GUARD_CODES = {
    "claims_not_verbatim": CLAIM_NOT_VERBATIM,
    "invalid_source_ns": INVALID_SOURCE_NS,
    "evidence_not_verbatim": EVIDENCE_NOT_VERBATIM,
}
MAX_WATCHED_TOKENS_LISTED = 20


def _guarded(
    model_label: str,
    quote_match: guard.Match | None,
    evidence: str,
    source_texts: Sequence[str],
    downgrade_to: str,
    *,
    sources_unknown: bool = False,
    evidence_optional: bool = False,
) -> tuple[str, str | None, dict[str, Any]]:
    """Apply the evidence side of the guard to one claim / sentence: `(label, reason, match
    info)`. A quote that is not in the answer (`quote_match` None) is `unverifiable`; a
    `supported` whose evidence is not in the sources is downgraded to `downgrade_to`
    (`evidence_not_verbatim`, or `invalid_source_ns` when it names no existing source). An empty
    evidence is allowed when `evidence_optional` (a sentence that only states missing
    information). Tolerant matches are reported in the info."""
    if quote_match is None:
        return UNVERIFIABLE, CLAIM_NOT_VERBATIM, {"quote_match": None, "evidence_match": None}
    if model_label != "supported":
        return model_label, None, {"quote_match": quote_match, "evidence_match": None}
    if sources_unknown:
        info = {"quote_match": quote_match, "evidence_match": None}
        return downgrade_to, INVALID_SOURCE_NS, info
    if evidence_optional and not evidence.strip():
        return model_label, None, {"quote_match": quote_match, "evidence_match": None}
    evidence_match = match_evidence(evidence, source_texts)
    info = {"quote_match": quote_match, "evidence_match": evidence_match}
    if evidence_match is None:
        return downgrade_to, EVIDENCE_NOT_VERBATIM, info
    return model_label, None, info


def _guard_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        **{name: sum(1 for r in rows if r["reason"] == code) for name, code in GUARD_CODES.items()},
        "claims_fuzzy": sum(1 for r in rows if r["quote_match"] == "fuzzy"),
        "evidence_fuzzy": sum(1 for r in rows if r["evidence_match"] == "fuzzy"),
    }


def _guard_code(guard_counts: Mapping[str, int]) -> str | None:
    """The reason code of a failed verdict that the guard shaped (the strongest first)."""
    for name in GUARD_FAILURES:
        if guard_counts[name]:
            return GUARD_CODES[name]
    return None


def _guard_rationale(guard_counts: Mapping[str, int]) -> str:
    return "".join(f"; {guard_counts[n]} {n}" for n in GUARD_FAILURES if guard_counts[n])


def _dedupe(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per identical claim quote (the model may list a claim twice and inflate the
    counts); when the copies disagree the first non-supported one wins."""
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = guard.normalize(row["claim_quote"])
        kept = chosen.get(key)
        if kept is None or (kept["label"] == "supported" and row["label"] != "supported"):
            chosen[key] = row
    return list(chosen.values())


def _verbatim_note(quotes: Sequence[str]) -> HumanMessage:
    shown = "\n".join(f"- {_clip(q)}" for q in quotes[:MAX_QUOTES_IN_NOTE])
    return HumanMessage(prompts.VERBATIM_RETRY_NOTE.format(quotes=shown))


async def _ask_with_verbatim_retry[S: BaseModel](
    call: JudgeCall,
    name: JudgeName,
    record: AnswerRecord,
    schema: type[S],
    messages: Sequence[BaseMessage],
    offending_of: Callable[[S], list[str]],
) -> tuple[S, Usage, dict[str, int]]:
    """Ask once; if some quote is not a (tolerant) match of the text it claims to copy the
    judgment is unreliable (the model repaired the claim while copying it): ask ONCE more with a
    corrective message listing the `offending_of` quotes. The retried output replaces the first
    one whole. Returns the output, the usage of every call and the retries
    `{"parse": n, "verbatim": 0|1}`."""
    first = await call.ask(name, record, schema, messages)
    bad = offending_of(first.parsed)
    if not bad:
        return first.parsed, first.usage, {"parse": first.parse_retries, "verbatim": 0}
    try:
        second = await call.ask(name, record, schema, [*messages, _verbatim_note(bad)])
    except JudgeOutputError as exc:
        raise JudgeOutputError(str(exc), first.usage + exc.usage) from None
    except JudgeCallError as exc:
        raise JudgeCallError(first.usage + exc.usage) from exc.__cause__
    retries = {"parse": first.parse_retries + second.parse_retries, "verbatim": 1}
    return second.parsed, first.usage + second.usage, retries


class FaithfulnessJudge:
    """Is every claim of the answer supported by the sources the synthesis saw?

    The model copies each claim VERBATIM from the answer, copies its evidence from the sources it
    names (`source_ns`, jointly), compares them axis by axis and labels it; the verdict is
    computed HERE: true iff every claim is `supported` (the model emits no overall verdict). The
    verbatim guard (`guard.py`) backs the copy: a claim quote that is not in the answer is retried
    once and then `unverifiable` (the verdict is false, code `claim_not_verbatim`); a `supported`
    claim whose evidence quote is not in its sources is downgraded to `not_found` (code
    `evidence_not_verbatim`, or `invalid_source_ns` when none of its `source_ns` exists).
    Identical claim quotes count once. `uncovered_protected` reports (no verdict effect) the
    negations, numbers and quantifiers of the answer that no claim quote covers.

    Not applicable when the agent's own refuse path answered (scope `out`: nothing is claimed) or
    when the judge model reports the answer is a refusal (`is_refusal`; the text is not guessed
    here). A refusal is an answer with no information on the asked topic (v8), so a flagged
    refusal stays n/a even when it lists or cites adjacent-topic facts (`cited_refusal` and
    `claims_listed` in the details count them). Cited facts are still verified by
    `citation_support` (it runs on any answer with `[n]` markers, independently of this flag);
    UNCITED facts of a flagged refusal are checked by no judge, only counted (the aggregate's
    `model_refusals.unchecked_claims`). For a `citations: none` agent a flagged refusal that
    lists claims is judged like any answer (`model_is_refusal` stays in the details): nothing
    else would check them. The model and heuristic signals are both recorded
    (`refusal_signals`; the heuristic says "not a refusal" for any cited answer, so
    `model_only` counts every cited v8 refusal: not a judge-error signal). An answer
    with sources missing is still judged (the claims cannot be supported by no sources); a model
    that lists no claims for a non-refusal is an `error` (`no_claims`), never a silent pass.
    """

    name = JudgeName.FAITHFULNESS
    method = "llm"

    def __init__(self, call: JudgeCall, prompt: str = prompts.FAITHFULNESS_PROMPT) -> None:
        self._call = call
        self._prompt = prompt

    async def judge(self, record: AnswerRecord) -> JudgeResult:
        if refused_by_scope(record):
            return JudgeResult.not_applicable(self.name, "the agent refused (scope out)", "refusal")
        if not record.answer.strip():
            return JudgeResult.not_applicable(self.name, "the answer is empty", "empty_answer")
        out, usage, retries = await _ask_with_verbatim_retry(
            self._call,
            self.name,
            record,
            FaithfulnessOutput,
            faithfulness_messages(self._prompt, record),
            lambda o: [
                c.claim_quote for c in o.claims if match_quote(c.claim_quote, record.answer) is None
            ],
        )
        signals = refusal_signals(out.is_refusal, record)
        # With `citations: none` no judge would check the claims of a flagged refusal (there is
        # no citation_support), so they are judged here; the flag stays in the details.
        if out.is_refusal and (record.citations == "inline" or not out.claims):
            # v8: a refusal may cite or list adjacent-topic facts; citation_support checks those.
            return JudgeResult.not_applicable(
                self.name,
                "the judge model reports the answer is a refusal",
                "model_refusal",
                "llm",
                usage,
                {
                    **signals,
                    "retries": retries,
                    "cited_refusal": bool(citation_numbers(record.answer)),
                    "claims_listed": len(out.claims),
                },
            )
        if not out.claims:
            return JudgeResult.failed(
                self.name, "the judge model listed no claims", usage, code="no_claims"
            )
        rows = _dedupe([self._row(c, record) for c in out.claims])
        guard_counts = _guard_counts(rows)
        unsupported = [r for r in rows if r["label"] != "supported"]
        verdict = not unsupported
        uncovered = guard.uncovered_protected(record.answer, [c.claim_quote for c in out.claims])
        details: dict[str, Any] = {
            "claims": rows,
            "unsupported": len(unsupported),
            "guard": guard_counts,
            "retries": retries,
            "uncovered_protected": len(uncovered),
            "uncovered_protected_tokens": uncovered[:MAX_WATCHED_TOKENS_LISTED],
            **signals,
        }
        code = None if verdict else _guard_code(guard_counts)
        rationale = f"{len(rows) - len(unsupported)}/{len(rows)} claims supported" + (
            _guard_rationale(guard_counts)
        )
        return JudgeResult.decided(self.name, verdict, rationale, "llm", details, usage, code=code)

    @staticmethod
    def _row(claim: ClaimCheck, record: AnswerRecord) -> dict[str, Any]:
        known = {s.n for s in record.sources}
        label, reason, matches = _guarded(
            claim.label,
            match_quote(claim.claim_quote, record.answer),
            claim.evidence_quote,
            _source_texts(record, claim.source_ns),
            "not_found",
            sources_unknown=bool(claim.source_ns) and not known.intersection(claim.source_ns),
        )
        return {
            **matches,
            "claim_quote": claim.claim_quote,
            "source_ns": claim.source_ns,
            "evidence_quote": _clip(claim.evidence_quote),
            "comparison": _clip(claim.comparison),
            "label": label,
            "model_label": claim.label,
            "reason": reason,
        }


class CitationSupportJudge:
    """ALCE-style: does the source set a sentence cites support that sentence?

    Order of business: no `[n]` marker -> not applicable (`citations_disabled` for an agent with
    `citations: none`, else `no_citations`; markers it emits anyway, e.g. from a custom
    synthesize prompt, are judged as usual); a marker pointing to a source that does
    not exist -> deterministic failure (no model call); otherwise the CODE enumerates the cited
    sentences (`cited_sentences`: the same splitter the precheck uses) and the model judges
    exactly that numbered list, each next to the sources IT cites (several markers = the sources
    together), so it cannot skip, merge or invent sentences; a model that still skips or repeats
    an index is an `error`. Per sentence it copies the sentence and the evidence VERBATIM,
    compares axis by axis and labels `supported | partial | not_supported`. The verdict is computed
    here: true iff every sentence is `supported` (`partial` counts as false). The verbatim guard
    works like faithfulness' except that the sentence quote is matched against ITS OWN enumerated
    sentence (covering at least 90% of it): a quote not found is retried once, then `unverifiable`
    (false, `claim_not_verbatim`); a `supported` whose evidence is not in the cited sources
    becomes `not_supported` (`evidence_not_verbatim`). A sentence that only states that
    information is missing sets `states_missing_info`: its evidence may then be empty (counted in
    `missing_info`), but the flag is honoured ONLY when the sentence, markers stripped, reads as
    "the sources do not say it" (`refusal.states_missing_information`); otherwise it is ignored,
    the evidence guard applies and `missing_info_flag_rejected` counts it. That a honoured
    sentence is `supported` rests on the model's assertion that the cited sources lack the
    topic.
    """

    name = JudgeName.CITATION_SUPPORT
    method = "llm"

    def __init__(self, call: JudgeCall, prompt: str = prompts.CITATION_SUPPORT_PROMPT) -> None:
        self._call = call
        self._prompt = prompt

    async def judge(self, record: AnswerRecord) -> JudgeResult:
        sentences = cited_sentences(record.answer)
        if not sentences and record.citations == "none":
            return JudgeResult.not_applicable(
                self.name, "the agent answers without citations", "citations_disabled"
            )
        if not sentences:
            return JudgeResult.not_applicable(
                self.name, "the answer carries no citations", "no_citations"
            )
        dangling = dangling_citations(record.answer, len(record.sources))
        if dangling:
            return JudgeResult.decided(
                self.name,
                False,
                f"citation(s) {list(dangling)} point to sources that do not exist "
                f"({len(record.sources)} sources)",
                "deterministic",
                {"dangling": list(dangling), "sources": len(record.sources)},
            )
        by_index = {s.index: s for s in sentences}

        def offending(o: CitationSupportOutput) -> list[str]:
            return [
                c.sentence_quote
                for c in o.checks
                if c.index not in by_index
                or match_sentence(c.sentence_quote, by_index[c.index].text) is None
            ]

        out, usage, retries = await _ask_with_verbatim_retry(
            self._call,
            self.name,
            record,
            CitationSupportOutput,
            citation_messages(self._prompt, record, sentences),
            offending,
        )
        checks = {c.index: c for c in out.checks}
        expected = {s.index for s in sentences}
        if len(out.checks) != len(checks) or set(checks) != expected:
            return JudgeResult.failed(
                self.name, "the judge model did not return one check per cited sentence", usage
            )
        rows = [self._row(s, checks[s.index], record) for s in sentences]
        guard_counts = _guard_counts(rows)
        failed = [r for r in rows if r["label"] != "supported"]
        verdict = not failed
        details: dict[str, Any] = {
            "sentences": rows,
            "unsupported": len(failed),
            "partial": sum(1 for r in rows if r["label"] == "partial"),
            "missing_info": sum(1 for r in rows if r["states_missing_info"]),
            "missing_info_flag_rejected": sum(1 for r in rows if r["missing_info_flag_rejected"]),
            "guard": guard_counts,
            "retries": retries,
        }
        code = None if verdict else _guard_code(guard_counts)
        rationale = f"{len(rows) - len(failed)}/{len(rows)} cited sentences supported" + (
            _guard_rationale(guard_counts)
        )
        return JudgeResult.decided(self.name, verdict, rationale, "llm", details, usage, code=code)

    @staticmethod
    def _row(sentence: CitedSentence, check: SentenceCheck, record: AnswerRecord) -> dict[str, Any]:
        missing_info = check.states_missing_info and states_missing_information(sentence.text)
        label, reason, matches = _guarded(
            check.label,
            match_sentence(check.sentence_quote, sentence.text),
            check.evidence_quote,
            _source_texts(record, sentence.numbers),
            "not_supported",
            evidence_optional=missing_info,
        )
        return {
            **matches,
            "sentence": sentence.text,
            "cited": list(sentence.numbers),
            "sentence_quote": check.sentence_quote,
            "evidence_quote": _clip(check.evidence_quote),
            "comparison": _clip(check.comparison),
            "states_missing_info": missing_info,
            "missing_info_flag_rejected": check.states_missing_info and not missing_info,
            "label": label,
            "model_label": check.label,
            "reason": reason,
        }


class AnswerRelevanceJudge:
    """Does the answer address the question?

    An empty answer fails without a model call. Not applicable when the agent refused (scope
    `out`) or when the judge model reports a refusal (the answer gives no information on the
    asked topic) for a question with no expected scope or an out-of-scope one
    (`refusal_correctness`' business).
    The one exception, whatever produced the refusal (the scope check or the answer text as the
    judge model reads it): an in-scope question (`expected_scope == "in"`) that was refused FAILS
    (`in_scope_refused`): a refusal cannot be an answer that addresses the question.
    """

    name = JudgeName.ANSWER_RELEVANCE
    method = "llm"

    def __init__(self, call: JudgeCall, prompt: str = prompts.ANSWER_RELEVANCE_PROMPT) -> None:
        self._call = call
        self._prompt = prompt

    async def judge(self, record: AnswerRecord) -> JudgeResult:
        # The rule, one sentence: a refusal is n/a EXCEPT for an expected-in question, where it is
        # false, whether it came from the scope check or from the answer text.
        if refused_by_scope(record):
            if record.expected_scope == "in":
                return JudgeResult.decided(
                    self.name,
                    False,
                    "an in-scope question was refused (scope out)",
                    "deterministic",
                    code="in_scope_refused",
                )
            return JudgeResult.not_applicable(self.name, "the agent refused (scope out)", "refusal")
        if not record.answer.strip():
            return JudgeResult.decided(self.name, False, "the answer is empty", "deterministic")
        asked = await self._call.ask(
            self.name, record, RelevanceOutput, relevance_messages(self._prompt, record)
        )
        out, usage = asked.parsed, asked.usage
        signals = {
            **refusal_signals(out.is_refusal, record),
            "retries": {"parse": asked.parse_retries},
        }
        if out.is_refusal and record.expected_scope == "in":
            return JudgeResult.decided(
                self.name,
                False,
                "an in-scope question was refused",
                "llm",
                signals,
                usage,
                code="in_scope_refused",
            )
        if out.is_refusal:
            return JudgeResult.not_applicable(
                self.name,
                "the judge model reports the answer is a refusal",
                "model_refusal",
                "llm",
                usage,
                signals,
            )
        return JudgeResult.decided(
            self.name, out.verdict, _clip(out.rationale), "llm", dict(signals), usage
        )
