from dataclasses import replace
from typing import Any

import pytest

from agintis.agent_kit.judge import (
    JudgeName,
    JudgeResult,
    JudgeStatus,
    ScopeVerdict,
    aggregate,
    build_judges,
    failure_reason,
    judge_record,
)
from agintis.agent_kit.judge.llm import FaithfulnessOutput
from tests.fakes.llm import RaisingModel, ScriptedChatModel, rate_limit_error
from tests.unit.agent_kit.judge.support import checks, claim, claims, make_record, relevance

LABEL = "judge (deepseek:deepseek-chat)"


def _by_schema(schema: object, _messages: list[object]) -> object:
    name = getattr(schema, "__name__", "")
    return {
        "FaithfulnessOutput": claims(True, True),
        "CitationSupportOutput": checks(True, True),
        "RelevanceOutput": relevance(True),
    }[name]


def _model() -> ScriptedChatModel:
    return ScriptedChatModel(responder=_by_schema)


def test_build_judges_defaults_to_all_judges_in_canonical_order() -> None:
    judges = build_judges(_model(), LABEL)
    assert [j.name for j in judges] == list(JudgeName)


def test_build_judges_honours_the_subset() -> None:
    judges = build_judges(_model(), LABEL, [JudgeName.REFUSAL_CORRECTNESS, JudgeName.FAITHFULNESS])
    assert [j.name for j in judges] == [JudgeName.FAITHFULNESS, JudgeName.REFUSAL_CORRECTNESS]


async def test_a_record_goes_through_every_judge() -> None:
    results = await judge_record(make_record(), build_judges(_model(), LABEL))

    assert {n: r.verdict for n, r in results.items()} == {
        JudgeName.FAITHFULNESS: True,
        JudgeName.CITATION_SUPPORT: True,
        JudgeName.ANSWER_RELEVANCE: True,
        JudgeName.REFUSAL_CORRECTNESS: True,
        JudgeName.SCRIPT_MIX: True,
    }


async def test_a_provider_failure_is_an_error_result_and_the_other_judges_still_run() -> None:
    # The deterministic judge is unaffected by the dead model; the LLM ones report the error.
    judges = build_judges(RaisingModel(error=rate_limit_error()), LABEL)
    results = await judge_record(make_record(), judges)

    assert results[JudgeName.REFUSAL_CORRECTNESS].status is JudgeStatus.OK
    for name in (JudgeName.FAITHFULNESS, JudgeName.CITATION_SUPPORT, JudgeName.ANSWER_RELEVANCE):
        assert results[name].status is JudgeStatus.ERROR
        assert results[name].rationale == "RateLimitError (status 429)"


async def test_one_unparseable_judge_does_not_touch_the_others() -> None:
    def respond(schema: object, _messages: list[object]) -> object:
        if getattr(schema, "__name__", "") == "FaithfulnessOutput":
            return "not json at all"
        return _by_schema(schema, _messages)

    results = await judge_record(
        make_record(), build_judges(ScriptedChatModel(responder=respond), LABEL)
    )

    assert results[JudgeName.FAITHFULNESS].status is JudgeStatus.ERROR
    assert "did not parse" in results[JudgeName.FAITHFULNESS].rationale
    assert results[JudgeName.FAITHFULNESS].usage.calls == 2  # one parse retry, both counted
    assert results[JudgeName.ANSWER_RELEVANCE].verdict is True


async def test_a_record_whose_run_failed_is_run_failed_for_every_judge_without_calls() -> None:
    model = _model()
    results = await judge_record(
        make_record("", error="RateLimitError"), build_judges(model, LABEL)
    )
    assert {r.status for r in results.values()} == {JudgeStatus.RUN_FAILED}
    assert {r.code for r in results.values()} == {"run_failed"}
    assert all("RateLimitError" in r.rationale for r in results.values())
    assert model.structured_seen == []


def test_failure_reason_never_echoes_the_exception_text() -> None:
    assert failure_reason(ValueError("secret sk-123")) == "ValueError"
    assert failure_reason(rate_limit_error()) == "RateLimitError (status 429)"


def _result(name: JudgeName, status: JudgeStatus, verdict: bool | None) -> JudgeResult:
    return JudgeResult(name, status, verdict, "r", "llm")


def test_aggregate_counts_statuses_and_pass_rate_over_the_decided_ones() -> None:
    n = JudgeName.FAITHFULNESS
    judgments = [
        {n: _result(n, JudgeStatus.OK, True)},
        {n: _result(n, JudgeStatus.OK, True)},
        {n: _result(n, JudgeStatus.OK, False)},
        {n: _result(n, JudgeStatus.NOT_APPLICABLE, None)},
        {n: _result(n, JudgeStatus.ERROR, None)},
    ]
    stats = aggregate(judgments, [n])[n.value]

    assert stats["records"] == 5
    assert (stats["decided"], stats["passed"], stats["failed"]) == (3, 2, 1)
    assert stats["pass_rate"] == pytest.approx(2 / 3)
    assert (stats["not_applicable"], stats["errors"]) == (1, 1)


def test_aggregate_without_decided_records_has_no_rate() -> None:
    n = JudgeName.CITATION_SUPPORT
    stats = aggregate([{n: _result(n, JudgeStatus.NOT_APPLICABLE, None)}], [n])[n.value]
    assert stats["pass_rate"] is None


async def test_aggregate_sums_token_usage() -> None:
    judgments = [
        await judge_record(make_record(id=f"r{i}"), build_judges(_model(), LABEL)) for i in range(3)
    ]
    stats = aggregate(judgments, list(JudgeName))
    assert stats["faithfulness"]["usage"] == {
        "input_tokens": 9,
        "output_tokens": 6,
        "total_tokens": 15,
        "calls": 3,
    }
    assert stats["refusal_correctness"]["usage"]["calls"] == 0


# --- review fixes -------------------------------------------------------------------------------


async def test_run_failed_records_are_counted_apart_and_are_not_judge_errors() -> None:
    judges = build_judges(_model(), LABEL)
    judgments = [
        await judge_record(make_record(id="ok"), judges),
        await judge_record(make_record("", id="boom", error="OperationalError"), judges),
    ]
    stats = aggregate(judgments, list(JudgeName))

    for name in JudgeName:
        assert stats[name.value]["run_failed"] == 1
        assert stats[name.value]["errors"] == 0
        assert stats[name.value]["reasons"]["run_failed"] == 1
        assert stats[name.value]["records"] == 2


async def test_no_claims_is_counted_separately_and_stays_an_error() -> None:
    model = ScriptedChatModel(
        responder=lambda schema, _m: (
            claims()
            if getattr(schema, "__name__", "") == "FaithfulnessOutput"
            else _by_schema(schema, _m)
        )
    )
    results = await judge_record(
        make_record(), build_judges(model, LABEL, [JudgeName.FAITHFULNESS])
    )
    stats = aggregate([results], [JudgeName.FAITHFULNESS])["faithfulness"]

    assert results[JudgeName.FAITHFULNESS].status is JudgeStatus.ERROR
    assert stats["errors"] == 1
    assert stats["reasons"] == {"no_claims": 1}


async def test_not_applicable_reasons_are_counted() -> None:
    judges = build_judges(_model(), LABEL)
    judgments = [
        await judge_record(make_record("Sin citas.", id="a", expected_scope=None), judges),
        await judge_record(
            make_record("Fijo.", id="b", scope=ScopeVerdict("out", None), sources=()), judges
        ),
    ]
    stats = aggregate(judgments, list(JudgeName))

    assert stats["citation_support"]["reasons"] == {"no_citations": 2}
    # record a's answer is not what the scripted claims quote: the guard counts it
    assert stats["faithfulness"]["reasons"] == {"claim_not_verbatim": 1, "refusal": 1}
    assert stats["refusal_correctness"]["reasons"] == {"no_expected_scope": 1}


async def test_a_failing_deterministic_judge_is_not_reported_as_llm() -> None:
    class Boom:
        name = JudgeName.REFUSAL_CORRECTNESS
        method = "deterministic"

        async def judge(self, record: object) -> JudgeResult:
            raise RuntimeError("bug")

    result = (await judge_record(make_record(), [Boom()]))[JudgeName.REFUSAL_CORRECTNESS]
    assert (result.status, result.method) == (JudgeStatus.ERROR, "deterministic")


async def test_a_failed_provider_call_still_counts_in_the_usage() -> None:
    class FailsInGenerate(ScriptedChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise rate_limit_error()

    judges = build_judges(FailsInGenerate(), LABEL, [JudgeName.FAITHFULNESS])
    result = (await judge_record(make_record(), judges))[JudgeName.FAITHFULNESS]

    assert result.status is JudgeStatus.ERROR
    assert result.usage.calls == 1
    assert result.usage.total_tokens == 0


async def test_the_aggregate_reports_how_the_model_and_the_heuristic_disagree() -> None:
    def respond(schema: object, messages: list[Any]) -> object:
        refusal = "REFUSE" in messages[-1].text
        name = getattr(schema, "__name__", "")
        if name == "FaithfulnessOutput":
            return claims(is_refusal=refusal) if refusal else claims(True)
        return _by_schema(schema, messages) if not refusal else relevance(False, is_refusal=True)

    judges = build_judges(
        ScriptedChatModel(responder=respond),
        LABEL,
        [JudgeName.FAITHFULNESS, JudgeName.ANSWER_RELEVANCE],
    )
    records = [
        make_record("No lo sé.", id="both"),  # model + heuristic (no REFUSE marker: model says no)
        make_record("REFUSE esto no se puede orientar.", id="model-only", expected_scope="in"),
        make_record("REFUSE no lo sé.", id="both-flagged"),
        make_record("El plazo es de 24 meses [1].", id="neither"),
    ]
    stats = aggregate([await judge_record(r, judges) for r in records], list(JudgeName))

    signals = stats["faithfulness"]["refusal_signals"]
    assert signals == {
        "compared": 4,
        "both": 1,  # "REFUSE no lo sé."
        "model_only": 1,  # "REFUSE esto ..."
        "heuristic_only": 1,  # "No lo sé." (the model did not flag it)
        "neither": 1,
    }
    assert stats["answer_relevance"]["reasons"]["in_scope_refused"] == 2  # both "in" refusals
    assert stats["refusal_correctness"]["refusal_signals"] is None


# --- the hybrid refusal signal --------------------------------------------------------------------

NEAR_DOMAIN = "Los documentos no mencionan ese plazo."  # the heuristic misses it, a model reads it


def _refusing_model() -> ScriptedChatModel:
    def respond(schema: object, messages: list[Any]) -> object:
        name = getattr(schema, "__name__", "")
        if name == "FaithfulnessOutput":
            return claims(is_refusal=True)
        return relevance(False, is_refusal=True)

    return ScriptedChatModel(responder=respond)


async def test_refusal_correctness_is_hybrid_when_the_llm_judges_ran() -> None:
    record = make_record(NEAR_DOMAIN, expected_scope="out", sources=())
    judges = build_judges(_refusing_model(), LABEL)

    result = (await judge_record(record, judges))[JudgeName.REFUSAL_CORRECTNESS]

    assert result.method == "hybrid"
    assert result.verdict is True  # the model caught the refusal the heuristic missed
    assert result.details["heuristic_verdict"] is False
    assert result.details["model_is_refusal"] is True
    assert "judge model" in result.rationale


async def test_refusal_correctness_stays_deterministic_without_the_llm_judges() -> None:
    record = make_record(NEAR_DOMAIN, expected_scope="out", sources=())
    judges = build_judges(_refusing_model(), LABEL, [JudgeName.REFUSAL_CORRECTNESS])

    result = (await judge_record(record, judges))[JudgeName.REFUSAL_CORRECTNESS]

    assert result.method == "deterministic"
    assert result.verdict is False  # heuristic only: the near-domain refusal is missed
    assert result.details["model_is_refusal"] is None


async def test_the_model_signal_also_catches_an_in_scope_refusal() -> None:
    record = make_record(NEAR_DOMAIN, expected_scope="in", sources=())
    results = await judge_record(record, build_judges(_refusing_model(), LABEL))

    assert results[JudgeName.REFUSAL_CORRECTNESS].verdict is False  # refused an answerable one
    assert results[JudgeName.ANSWER_RELEVANCE].verdict is False


async def test_a_model_that_says_answer_does_not_turn_an_answer_into_a_refusal() -> None:
    record = make_record("El plazo es de 24 meses [1].", expected_scope="out")
    results = await judge_record(record, build_judges(_model(), LABEL))
    refusal = results[JudgeName.REFUSAL_CORRECTNESS]

    assert refusal.method == "hybrid"
    assert refusal.verdict is False  # an out-of-scope question that was answered
    assert refusal.details["model_is_refusal"] is False


async def test_without_a_model_verdict_the_signal_is_none() -> None:
    class Dead(ScriptedChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise rate_limit_error()

    record = make_record(NEAR_DOMAIN, expected_scope="out", sources=())
    results = await judge_record(record, build_judges(Dead(), LABEL))

    assert results[JudgeName.FAITHFULNESS].status is JudgeStatus.ERROR
    assert results[JudgeName.REFUSAL_CORRECTNESS].method == "deterministic"


async def test_the_aggregate_shows_the_heuristic_only_pass_rate_next_to_the_hybrid_one() -> None:
    judges = build_judges(_refusing_model(), LABEL)
    records = [
        make_record(NEAR_DOMAIN, id="a", expected_scope="out", sources=()),
        make_record("No lo sé.", id="b", expected_scope="out", sources=()),
    ]
    stats = aggregate([await judge_record(r, judges) for r in records], list(JudgeName))
    entry = stats["refusal_correctness"]

    assert entry["pass_rate"] == 1.0  # hybrid: both refusals recognised
    assert entry["refusal_correctness_heuristic"] == {
        "decided": 2,
        "passed": 1,  # the heuristic alone misses the near-domain one
        "pass_rate": 0.5,
    }
    assert entry["refusal_model_signal"]["hybrid"] == 2
    assert stats["faithfulness"]["refusal_correctness_heuristic"] is None


# --- the model signal is relevance's, never an OR -------------------------------------------------


def _split_model(
    relevance_refusal: bool | None, faithfulness_refusal: bool, claim_list: tuple[bool, ...] = ()
) -> ScriptedChatModel:
    """relevance_refusal None = the relevance call fails."""

    def respond(schema: object, messages: list[Any]) -> object:
        name = getattr(schema, "__name__", "")
        if name == "FaithfulnessOutput":
            return claims(*claim_list, is_refusal=faithfulness_refusal)
        if relevance_refusal is None:
            raise RuntimeError("relevance down")
        return relevance(not relevance_refusal, is_refusal=relevance_refusal)

    return ScriptedChatModel(responder=respond)


async def test_a_refusal_flagged_by_faithfulness_with_claims_stays_a_refusal() -> None:
    # v8: the topic decides, not the claim count; citation_support checks the cited facts.
    record = make_record("No sé X, pero Y [1].", expected_scope="out")
    model = _split_model(None, True, claim_list=(True,))  # relevance down: faithfulness fallback

    results = await judge_record(record, build_judges(model, LABEL))

    faith = results[JudgeName.FAITHFULNESS]
    assert faith.details["model_is_refusal"] is True  # the raw flag, not corrected by claims
    assert faith.details["cited_refusal"] is True
    refusal = results[JudgeName.REFUSAL_CORRECTNESS]
    assert refusal.verdict is True  # an out-of-scope question refused
    assert refusal.details["model_is_refusal"] is True


async def test_relevance_is_the_signal_and_faithfulness_never_ors_into_it() -> None:
    record = make_record("Los documentos no mencionan ese plazo.", expected_scope="out")
    model = _split_model(False, True)  # faithfulness says refusal (no claims), relevance does not

    refusal = (await judge_record(record, build_judges(model, LABEL)))[
        JudgeName.REFUSAL_CORRECTNESS
    ]

    assert refusal.details["model_is_refusal"] is False
    assert refusal.details["signal_source"] == "answer_relevance"
    assert refusal.details["llm_judges_disagree"] is True
    assert refusal.verdict is False  # relevance read it as an answer


async def test_faithfulness_is_only_the_fallback_when_relevance_did_not_run() -> None:
    record = make_record("Los documentos no mencionan ese plazo.", expected_scope="out")
    model = _split_model(None, True)

    results = await judge_record(record, build_judges(model, LABEL))

    assert results[JudgeName.ANSWER_RELEVANCE].status is JudgeStatus.ERROR
    refusal = results[JudgeName.REFUSAL_CORRECTNESS]
    assert refusal.details["signal_source"] == "faithfulness"
    assert refusal.details["llm_judges_disagree"] is None
    assert refusal.verdict is True


async def test_the_aggregate_reports_disagreements_sources_and_deterministic_fallbacks() -> None:
    both = build_judges(_split_model(False, True), LABEL)
    agree = build_judges(_split_model(True, True), LABEL)
    only_refusal = build_judges(_split_model(True, True), LABEL, [JudgeName.REFUSAL_CORRECTNESS])
    judgments = [
        await judge_record(make_record("Texto.", id="d", expected_scope="out"), both),
        await judge_record(make_record("No lo sé.", id="a", expected_scope="out"), agree),
        await judge_record(make_record("No lo sé.", id="f", expected_scope="out"), only_refusal),
    ]
    entry = aggregate(judgments, list(JudgeName))["refusal_correctness"]["refusal_model_signal"]

    assert entry["decided"] == 3
    assert (entry["hybrid"], entry["deterministic_fallbacks"]) == (2, 1)
    assert entry["sources"] == {"answer_relevance": 2, "None": 1}
    assert entry["relevance_vs_faithfulness"] == {"compared": 2, "disagree": 1}
    assert aggregate(judgments, list(JudgeName))["faithfulness"]["refusal_model_signal"] is None


async def test_the_aggregate_totals_the_retries_and_the_verbatim_guard() -> None:
    bad = claims(True)
    bad.claims[0].claim_quote = "El plazo ya no es de 24 meses"  # a repaired claim: never verbatim
    judges = build_judges(
        ScriptedChatModel(responder=lambda _s, _m: bad), LABEL, [JudgeName.FAITHFULNESS]
    )
    judgments = [await judge_record(make_record(id=f"r{i}"), judges) for i in range(2)]
    stats = aggregate(judgments, [JudgeName.FAITHFULNESS])["faithfulness"]

    assert stats["retries"] == {"parse": 0, "verbatim": 2}
    assert stats["verbatim_guard"] == {
        "claims_fuzzy": 0,
        "claims_not_verbatim": 2,
        "evidence_fuzzy": 0,
        "evidence_not_verbatim": 0,
        "invalid_source_ns": 0,
    }
    assert stats["reasons"] == {"claim_not_verbatim": 2}
    assert stats["passed"] == 0


async def test_the_aggregate_reports_the_uncovered_protected_tokens() -> None:
    answer = "El socio no podrá transferir su puesto antes de los 60 días [1]."
    out = FaithfulnessOutput(
        claims=[claim(True, claim_quote="podrá transferir su puesto", source_ns=[1],
                      evidence_quote="El plazo es de 24 meses.")],
        is_refusal=False,
    )  # fmt: skip
    judges = build_judges(
        ScriptedChatModel(responder=lambda _s, _m: out), LABEL, [JudgeName.FAITHFULNESS]
    )
    judgments = [await judge_record(make_record(answer, id="r1"), judges)]
    stats = aggregate(judgments, [JudgeName.FAITHFULNESS])["faithfulness"]
    assert stats["uncovered_protected"] == {"records": 1, "tokens": 3}


async def test_the_aggregate_counts_model_refusals_with_uncited_claims_as_unchecked() -> None:
    cited = make_record("No sé X, pero Y [1].", expected_scope="out")
    uncited = make_record("No lo sé; el plazo es de 30 días.", expected_scope="out")
    plain = make_record("No lo sé.", expected_scope="out")
    model = _split_model(None, True, claim_list=(True,))
    judges = build_judges(model, LABEL)
    judgments = [await judge_record(r, judges) for r in (cited, uncited)]
    plain_model = _split_model(None, True)
    judgments.append(await judge_record(plain, build_judges(plain_model, LABEL)))

    stats = aggregate(judgments, [JudgeName.FAITHFULNESS])

    assert stats["faithfulness"]["model_refusals"] == {
        "total": 3,
        "cited_refusal": 1,
        "unchecked_claims": 1,
    }


async def test_an_uncited_refusal_of_an_agent_without_citations_has_its_claims_checked() -> None:
    # with `citations: none` no citation_support would see the fabricated 30 days
    fabricated = replace(
        make_record("No lo sé; el plazo es de 30 días.", expected_scope="out"), citations="none"
    )
    out = FaithfulnessOutput(
        claims=[claim("contradicted", claim_quote="el plazo es de 30 días", source_ns=[1],
                      evidence_quote="El plazo es de 24 meses.")],
        is_refusal=True,
    )  # fmt: skip
    judges = build_judges(
        ScriptedChatModel(responder=lambda _s, _m: out), LABEL, [JudgeName.FAITHFULNESS]
    )
    plain = replace(make_record("No lo sé.", expected_scope="out"), citations="none")
    plain_judges = build_judges(_split_model(None, True), LABEL, [JudgeName.FAITHFULNESS])

    judgments = [await judge_record(fabricated, judges), await judge_record(plain, plain_judges)]

    faith = judgments[0][JudgeName.FAITHFULNESS]
    assert (faith.status, faith.verdict) == (JudgeStatus.OK, False)
    assert faith.details["model_is_refusal"] is True
    assert faith.details["unsupported"] == 1
    stats = aggregate(judgments, [JudgeName.FAITHFULNESS])
    # a refusal listing no claims stays n/a
    assert stats["faithfulness"]["model_refusals"] == {
        "total": 1,
        "cited_refusal": 0,
        "unchecked_claims": 0,
    }
