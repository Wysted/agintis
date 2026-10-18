import asyncio
import dataclasses
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agintis.agent_kit.judge import (
    AnswerRelevanceJudge,
    CitationSupportJudge,
    CitedSource,
    FaithfulnessJudge,
    JudgeCall,
    JudgeName,
    JudgeOutputError,
    JudgeStatus,
    RefusalCorrectnessJudge,
    ScopeVerdict,
    ScriptMixJudge,
)
from agintis.agent_kit.judge.llm import (
    CitationSupportOutput,
    ClaimCheck,
    FaithfulnessOutput,
    SentenceCheck,
)
from tests.fakes.llm import ScriptedChatModel
from tests.unit.agent_kit.judge.support import (
    checks,
    claim,
    claims,
    judge_model,
    make_record,
    relevance,
    sentence_check,
    tracking_model,
)

LABEL = "judge (deepseek:deepseek-chat)"


def _call(model: ScriptedChatModel) -> JudgeCall:
    return JudgeCall(model, LABEL)


# --- faithfulness -------------------------------------------------------------------------------


class TestFaithfulness:
    async def test_all_claims_supported_is_a_pass_and_lists_the_claims(self) -> None:
        model = judge_model(claims(True, True))
        result = await FaithfulnessJudge(_call(model)).judge(make_record())

        assert result.status is JudgeStatus.OK
        assert result.verdict is True
        assert result.method == "llm"
        assert [c["label"] for c in result.details["claims"]] == ["supported", "supported"]
        assert result.details["guard"] == {
            "claims_not_verbatim": 0,
            "evidence_not_verbatim": 0,
            "invalid_source_ns": 0,
            "claims_fuzzy": 0,
            "evidence_fuzzy": 0,
        }
        assert result.details["retries"] == {"parse": 0, "verbatim": 0}
        assert "code" not in result.details
        assert result.usage.calls == 1
        assert result.usage.total_tokens == 5

    @pytest.mark.parametrize("label", ["contradicted", "not_found"])
    async def test_one_unsupported_claim_fails_computed_in_code(self, label: str) -> None:
        model = judge_model(claims(True, label))
        result = await FaithfulnessJudge(_call(model)).judge(make_record())

        assert result.verdict is False
        assert result.details["unsupported"] == 1
        assert result.rationale.startswith("1/2 claims supported")

    async def test_the_model_has_no_verdict_field_and_the_schema_order_is_copy_compare_label(
        self,
    ) -> None:
        assert list(ClaimCheck.model_fields) == [
            "claim_quote",
            "source_ns",
            "evidence_quote",
            "comparison",
            "label",
        ]
        assert list(FaithfulnessOutput.model_fields) == ["claims", "is_refusal"]
        assert list(SentenceCheck.model_fields) == [
            "index",
            "sentence_quote",
            "cited",
            "evidence_quote",
            "comparison",
            "states_missing_info",
            "label",
        ]
        assert list(CitationSupportOutput.model_fields) == ["checks"]

    async def test_the_prompt_is_spanish_and_the_data_goes_in_a_user_message(self) -> None:
        model = judge_model(claims(True), claims(True))  # not verbatim: asked twice
        record = make_record("Dato <script> [1].")
        await FaithfulnessJudge(_call(model)).judge(record)

        system, user = model.structured_seen[0]
        assert isinstance(system, SystemMessage)
        assert "fidelidad" in system.text
        assert isinstance(user, HumanMessage)
        assert "El plazo es de 24 meses." in user.text  # the chunk text the synthesis saw
        assert "[2] file:///b.pdf" in user.text  # untitled source: the uri labels it
        assert "<script>" not in user.text  # markup neutralized
        assert "&lt;script&gt;" in user.text
        # sources first, then the answer, then the question
        text = user.text
        assert text.index("<fuentes>") < text.index("<respuesta>") < text.index("<pregunta>")

    async def test_a_scope_out_answer_is_not_applicable_and_costs_no_call(self) -> None:
        model = judge_model()
        record = make_record("Mensaje fijo.", scope=ScopeVerdict("out", 0.1), sources=())
        result = await FaithfulnessJudge(_call(model)).judge(record)

        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.code == "refusal"
        assert model.structured_seen == []

    async def test_the_text_heuristic_never_gates_it_the_judge_model_reports_the_refusal(
        self,
    ) -> None:
        # A phrase the heuristic would flag, but the model says it is an answer: it is judged.
        answer = "No sé cuál es el plazo en general, pero la ley dice 24 meses."
        out = FaithfulnessOutput(
            claims=[claim(True, claim_quote="la ley dice 24 meses")], is_refusal=False
        )
        model = judge_model(out)
        result = await FaithfulnessJudge(_call(model)).judge(make_record(answer))
        assert result.status is JudgeStatus.OK
        assert len(model.structured_seen) == 1

    async def test_is_refusal_together_with_claims_is_not_applicable(self) -> None:
        # v8: the flag is the refusal; the listed claims are only counted (citation_support
        # verifies the cited sentences independently).
        out = claims(True, False, is_refusal=True)
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(
            make_record("No sé X, pero El plazo es de 24 meses [1]. La tasa es del 10% [2].")
        )
        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.details["model_is_refusal"] is True
        assert result.details["claims_listed"] == 2

    async def test_a_refusal_reported_by_the_model_is_not_applicable_with_a_counted_reason(
        self,
    ) -> None:
        model = judge_model(claims(is_refusal=True))
        result = await FaithfulnessJudge(_call(model)).judge(make_record("No lo sé.", sources=()))

        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.code == "model_refusal"
        assert result.usage.calls == 1  # the call was made and is counted

    async def test_an_empty_answer_is_not_applicable(self) -> None:
        result = await FaithfulnessJudge(_call(judge_model())).judge(make_record("  "))
        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.code == "empty_answer"

    async def test_a_model_that_lists_no_claims_is_an_error_not_a_pass(self) -> None:
        result = await FaithfulnessJudge(_call(judge_model(claims()))).judge(make_record())
        assert result.status is JudgeStatus.ERROR
        assert result.code == "no_claims"
        assert result.verdict is None
        assert result.usage.calls == 1  # the spent call is still counted

    async def test_an_answer_without_sources_is_still_judged(self) -> None:
        model = judge_model(claims("not_found"))
        result = await FaithfulnessJudge(_call(model)).judge(
            make_record("El plazo es de 24 meses.", sources=())
        )
        assert result.verdict is False
        assert "(ninguna fuente)" in model.structured_seen[0][1].text

    # --- parse retry ---

    async def test_unparseable_output_is_retried_once_with_a_corrective_note(self) -> None:
        model = judge_model('{"claims": "not a list"}', claims(True))
        result = await FaithfulnessJudge(_call(model)).judge(make_record())

        assert result.verdict is True
        assert result.details["retries"] == {"parse": 1, "verbatim": 0}
        assert result.usage.calls == 2  # both attempts are counted
        first, second = model.structured_seen
        assert len(second) == len(first) + 1
        assert second[:-1] == first
        assert isinstance(second[-1], HumanMessage)
        assert "esquema" in second[-1].text

    async def test_unparseable_twice_raises_with_the_usage_of_both_attempts(self) -> None:
        model = judge_model('{"claims": "not a list"}', "still not json")
        with pytest.raises(JudgeOutputError) as caught:
            await FaithfulnessJudge(_call(model)).judge(make_record())
        assert caught.value.usage.calls == 2

    # --- verbatim guard ---

    async def test_a_claim_quote_not_in_the_answer_is_retried_once_with_the_offending_quote(
        self,
    ) -> None:
        # the claim was "repaired" toward the source while extracting it: not a span of the answer
        repaired = FaithfulnessOutput(
            claims=[claim(True, claim_quote="Si aún no se ha cumplido el plazo, vale 24 meses")],
            is_refusal=False,
        )
        model = judge_model(repaired, claims(True))
        result = await FaithfulnessJudge(_call(model)).judge(make_record())

        assert result.verdict is True  # the retried output replaced the first one
        assert result.details["retries"] == {"parse": 0, "verbatim": 1}
        assert result.usage.calls == 2
        note = model.structured_seen[1][-1]
        assert isinstance(note, HumanMessage)
        assert "Si aún no se ha cumplido el plazo" in note.text
        assert "textualmente" in note.text

    async def test_a_claim_still_not_verbatim_after_the_retry_is_unverifiable_and_fails(
        self,
    ) -> None:
        repaired = FaithfulnessOutput(
            claims=[claim(True), claim(True, 2, claim_quote="La tasa fue del 10%")],
            is_refusal=False,
        )
        model = judge_model(repaired, repaired)
        result = await FaithfulnessJudge(_call(model)).judge(make_record())

        assert len(model.structured_seen) == 2  # ONE retry, never more
        assert result.status is JudgeStatus.OK
        assert result.verdict is False
        assert [c["label"] for c in result.details["claims"]] == ["supported", "unverifiable"]
        assert result.details["claims"][1]["model_label"] == "supported"
        assert result.details["claims"][1]["reason"] == "claim_not_verbatim"
        assert result.code == "claim_not_verbatim"
        assert result.details["guard"]["claims_not_verbatim"] == 1
        assert result.details["retries"]["verbatim"] == 1

    async def test_a_supported_claim_with_an_evidence_quote_not_in_its_source_is_downgraded(
        self,
    ) -> None:
        invented = FaithfulnessOutput(
            claims=[claim(True, evidence_quote="El plazo es de 36 meses.")], is_refusal=False
        )
        model = judge_model(invented)
        result = await FaithfulnessJudge(_call(model)).judge(make_record())

        assert len(model.structured_seen) == 1  # an evidence miss is not retried
        assert result.verdict is False
        row = result.details["claims"][0]
        assert (row["label"], row["model_label"]) == ("not_found", "supported")
        assert row["reason"] == "evidence_not_verbatim"
        assert result.code == "evidence_not_verbatim"
        assert result.details["guard"]["evidence_not_verbatim"] == 1

    @pytest.mark.parametrize(
        ("evidence", "source_n"),
        [
            ("", [1]),
            ("La tasa es del 10%.", [1]),
            ("El plazo es de 24 meses.", [7]),
            ("El plazo es de 24 meses.", []),
        ],
    )
    async def test_supported_needs_evidence_found_in_the_named_source(
        self, evidence: str, source_n: list[int]
    ) -> None:
        out = FaithfulnessOutput(
            claims=[claim(True, evidence_quote=evidence, source_ns=source_n)], is_refusal=False
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(make_record())
        assert result.verdict is False
        assert result.details["claims"][0]["label"] == "not_found"

    async def test_a_claim_that_combines_sources_is_checked_against_their_union(self) -> None:
        out = FaithfulnessOutput(
            claims=[
                claim(
                    True,
                    claim_quote="El plazo es de 24 meses",
                    source_ns=[1, 2],
                    evidence_quote="El plazo es de 24 meses. ... La tasa es del 10%.",
                )
            ],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(make_record())
        assert result.verdict is True
        assert result.details["claims"][0]["source_ns"] == [1, 2]

    async def test_a_fragment_found_only_in_an_unlisted_source_is_not_evidence(self) -> None:
        out = FaithfulnessOutput(
            claims=[
                claim(
                    True,
                    source_ns=[1],
                    evidence_quote="El plazo es de 24 meses. ... La tasa es del 10%.",
                )
            ],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(make_record())
        assert result.details["claims"][0]["label"] == "not_found"

    async def test_fragmented_evidence_with_a_footnote_digit_is_supported(self) -> None:
        source = CitedSource(
            1,
            None,
            "file:///a",
            "Vector de tasas publicado. Otro texto. La tasa equivalente3 deriva.",
        )
        out = FaithfulnessOutput(
            claims=[
                claim(
                    True,
                    claim_quote="La tasa equivalente deriva",
                    source_ns=[1],
                    evidence_quote="Vector de tasas publicado ... La tasa equivalente3 deriva",
                )
            ],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(
            make_record("La tasa equivalente deriva [1].", sources=(source,))
        )
        assert result.verdict is True

    async def test_tolerant_matches_are_counted_and_do_not_fail_the_claim(self) -> None:
        answer = (
            "El Ente Gestor informa a la Tesorería para que efectúe el depósito dentro del "
            "plazo señalado por el Reglamento [1]."
        )
        source = CitedSource(
            1,
            None,
            "file:///a",
            "Texto 1 Nota al pie interna. El Ente Gestor informa a la Tesorería para que efectúe "
            "el depósito dentro del plazo señalado por el Reglamento vigente de la entidad.",
        )
        out = FaithfulnessOutput(
            claims=[
                claim(
                    True,
                    claim_quote=answer.replace("para que", "para que éste"),
                    source_ns=[1],
                    evidence_quote="El Ente Gestor informa a la Tesorería para que efectúe "
                    "el depósito dentro del plazo señalado por el Reglamento vigente",
                )
            ],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(
            make_record(answer, sources=(source,))
        )
        assert result.verdict is True
        assert result.details["guard"]["claims_fuzzy"] == 1
        assert result.details["claims"][0]["quote_match"] == "fuzzy"

    async def test_a_claim_naming_only_unknown_sources_is_invalid_source_ns(self) -> None:
        out = FaithfulnessOutput(claims=[claim(True, source_ns=[7, 9])], is_refusal=False)
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(make_record())

        row = result.details["claims"][0]
        assert (row["label"], row["reason"]) == ("not_found", "invalid_source_ns")
        assert result.code == "invalid_source_ns"
        assert result.details["guard"]["invalid_source_ns"] == 1

    async def test_one_known_source_among_unknown_ones_is_judged_normally(self) -> None:
        out = FaithfulnessOutput(claims=[claim(True, source_ns=[1, 9])], is_refusal=False)
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(make_record())
        assert result.verdict is True

    async def test_identical_claims_count_once_and_the_worst_label_wins(self) -> None:
        out = FaithfulnessOutput(
            claims=[claim(True), claim(True), claim("contradicted"), claim(True, 2)],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(make_record())

        assert [c["label"] for c in result.details["claims"]] == ["contradicted", "supported"]
        assert result.details["unsupported"] == 1

    async def test_a_truncating_claim_quote_reports_the_uncovered_negation_but_keeps_the_verdict(
        self,
    ) -> None:
        answer = "El socio no podrá transferir su puesto antes de los 60 días [1]."
        source = CitedSource(1, None, "file:///a", "El socio no podrá transferir su puesto.")
        out = FaithfulnessOutput(
            claims=[
                claim(
                    True,
                    claim_quote="podrá transferir su puesto",
                    evidence_quote="podrá transferir su puesto",
                    source_ns=[1],
                )
            ],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(
            make_record(answer, sources=(source,))
        )

        assert result.verdict is True  # report-only: no verdict change yet
        assert result.details["uncovered_protected"] == 3
        assert result.details["uncovered_protected_tokens"] == ["no", "antes", "60"]

    async def test_a_flagged_refusal_that_cites_adjacent_facts_is_not_applicable(self) -> None:
        model = judge_model(claims(is_refusal=True))
        result = await FaithfulnessJudge(_call(model)).judge(make_record())  # cites [1] [2]
        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.code == "model_refusal"
        assert result.details["cited_refusal"] is True
        assert result.details["claims_listed"] == 0
        assert result.usage.calls == 1

    async def test_a_flagged_refusal_that_lists_claims_is_still_a_refusal(self) -> None:
        model = judge_model(claims(True, is_refusal=True))
        result = await FaithfulnessJudge(_call(model)).judge(make_record())
        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.details["claims_listed"] == 1
        assert result.details["model_is_refusal"] is True

    async def test_an_uncited_flagged_refusal_is_not_a_cited_refusal(self) -> None:
        model = judge_model(claims(is_refusal=True))
        result = await FaithfulnessJudge(_call(model)).judge(
            make_record("No encuentro esa información en las fuentes.")
        )
        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.details["cited_refusal"] is False

    async def test_a_claim_quote_without_the_marker_matches_a_marked_sentence(self) -> None:
        answer = "El Ente Gestor informa a la Tesorería [1] para que efectúe el depósito [2]."
        out = FaithfulnessOutput(
            claims=[
                claim(
                    True,
                    claim_quote=(
                        "El Ente Gestor informa a la Tesorería para que efectúe el depósito."
                    ),
                    evidence_quote="El plazo es de 24 meses.",
                    source_ns=[1],
                )
            ],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(make_record(answer))
        assert result.details["claims"][0]["quote_match"] == "exact"

    async def test_a_non_supported_label_keeps_its_label_whatever_the_evidence(self) -> None:
        out = FaithfulnessOutput(
            claims=[claim("contradicted", evidence_quote="not in any source")], is_refusal=False
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(make_record())
        row = result.details["claims"][0]
        assert (row["label"], row["reason"]) == ("contradicted", None)
        assert "code" not in result.details

    async def test_quotes_survive_typographic_differences_only(self) -> None:
        answer = "El comité dijo \u201chola\u201d   y\nluego  \u2018salió\u2019 [1]."
        source = CitedSource(1, None, "file:///a", "Dijo \"hola\" y luego 'salió'.")
        out = FaithfulnessOutput(
            claims=[
                claim(
                    True,
                    claim_quote='El Comité dijo "hola"',  # case differs: NOT accepted
                    evidence_quote='Dijo "hola" y luego',
                    source_ns=[1],
                )
            ],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(out, out))).judge(
            make_record(answer, sources=(source,))
        )
        assert result.details["claims"][0]["label"] == "unverifiable"

        ok = FaithfulnessOutput(
            claims=[
                claim(
                    True,
                    claim_quote="comité dijo \"hola\" y luego 'salió'",
                    evidence_quote="Dijo \"hola\" y luego 'salió'.",
                    source_ns=[1],
                )
            ],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(ok))).judge(
            make_record(answer, sources=(source,))
        )
        assert result.verdict is True

    async def test_a_quote_copied_from_the_neutralized_rendering_matches_the_raw_text(
        self,
    ) -> None:
        source = CitedSource(1, None, "file:///a", "Si a < b entonces c.")
        out = FaithfulnessOutput(
            claims=[
                claim(
                    True,
                    claim_quote="si a &lt; b",
                    evidence_quote="Si a &lt; b entonces c.",
                    source_ns=[1],
                )
            ],
            is_refusal=False,
        )
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(
            make_record("Es verdad que si a < b [1].", sources=(source,))
        )
        assert result.verdict is True

    async def test_a_failed_verbatim_retry_call_still_counts_the_first_call(self) -> None:
        repaired = FaithfulnessOutput(claims=[claim(True, claim_quote="x")], is_refusal=False)
        model = judge_model(repaired, "not json", "not json")
        with pytest.raises(JudgeOutputError) as caught:
            await FaithfulnessJudge(_call(model)).judge(make_record())
        assert caught.value.usage.calls == 3  # first call + two failed parse attempts


# --- citation support ---------------------------------------------------------------------------


class TestCitationSupport:
    async def test_every_cited_sentence_supported_passes(self) -> None:
        model = judge_model(checks(True, True))
        result = await CitationSupportJudge(_call(model)).judge(make_record())

        assert result.verdict is True
        assert result.method == "llm"
        assert [s["cited"] for s in result.details["sentences"]] == [[1], [2]]
        assert [s["label"] for s in result.details["sentences"]] == ["supported", "supported"]

    async def test_one_unsupported_sentence_fails(self) -> None:
        result = await CitationSupportJudge(_call(judge_model(checks(True, False)))).judge(
            make_record()
        )
        assert result.verdict is False
        assert result.details["unsupported"] == 1
        assert result.rationale.startswith("1/2 cited sentences supported")

    async def test_partial_counts_as_false(self) -> None:
        result = await CitationSupportJudge(_call(judge_model(checks(True, "partial")))).judge(
            make_record()
        )
        assert result.verdict is False
        assert (result.details["unsupported"], result.details["partial"]) == (1, 1)

    async def test_each_sentence_is_shown_with_the_sources_it_cites(self) -> None:
        model = judge_model(checks(True), checks(True))  # not verbatim: asked twice
        record = make_record("Dato combinado [1][2]. Sin cita.")
        await CitationSupportJudge(_call(model)).judge(record)

        user = model.structured_seen[0][1].text
        assert 'index="1"' in user
        assert "El plazo es de 24 meses." in user
        assert "La tasa es del 10%." in user
        assert "Sin cita" not in user  # sentences without markers are not judged here

    async def test_the_code_enumerates_the_sentences_and_the_model_must_answer_that_list(
        self,
    ) -> None:
        model = judge_model(checks(True, True))
        await CitationSupportJudge(_call(model)).judge(make_record())
        user = model.structured_seen[0][1].text
        assert user.index('index="1"') < user.index('index="2"')
        assert user.count("<oracion ") == 2

    async def test_no_citations_is_not_applicable(self) -> None:
        model = judge_model()
        result = await CitationSupportJudge(_call(model)).judge(make_record("Sin citas."))
        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.code == "no_citations"
        assert model.structured_seen == []

    async def test_an_agent_without_citations_is_not_applicable(self) -> None:
        model = judge_model()
        record = dataclasses.replace(make_record("El plazo es de 24 meses."), citations="none")
        result = await CitationSupportJudge(_call(model)).judge(record)
        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.code == "citations_disabled"
        assert model.structured_seen == []

    async def test_markers_of_an_agent_without_citations_are_judged_as_usual(self) -> None:
        # e.g. a custom synthesize prompt that cites despite `citations: none`
        model = judge_model(checks(True, False))
        record = dataclasses.replace(make_record(), citations="none")
        result = await CitationSupportJudge(_call(model)).judge(record)
        assert (result.status, result.verdict) == (JudgeStatus.OK, False)

        dangling = dataclasses.replace(make_record("Dato [7]."), citations="none")
        result = await CitationSupportJudge(_call(judge_model())).judge(dangling)
        assert (result.verdict, result.method) == (False, "deterministic")
        assert result.details["dangling"] == [7]

    async def test_a_marker_pointing_to_a_missing_source_fails_without_calling_the_model(
        self,
    ) -> None:
        model = judge_model()
        result = await CitationSupportJudge(_call(model)).judge(
            make_record("Dato [1]. Otro dato [7].")
        )

        assert result.status is JudgeStatus.OK
        assert result.verdict is False
        assert result.method == "deterministic"
        assert result.details["dangling"] == [7]
        assert model.structured_seen == []
        assert result.usage.calls == 0

    async def test_citations_without_any_source_fail_deterministically(self) -> None:
        result = await CitationSupportJudge(_call(judge_model())).judge(
            make_record("Dato [1].", sources=())
        )
        assert result.verdict is False
        assert result.details["dangling"] == [1]

    @pytest.mark.parametrize("indexes", [[1], [1, 1], [1, 3]])
    async def test_a_model_that_skips_or_repeats_a_sentence_is_an_error(
        self, indexes: list[int]
    ) -> None:
        out = checks(*([True] * len(indexes)), indexes=indexes)
        model = judge_model(out, out)  # an unknown index is also retried once
        result = await CitationSupportJudge(_call(model)).judge(make_record())
        assert result.status is JudgeStatus.ERROR

    async def test_a_sentence_quote_is_matched_against_its_own_sentence_not_the_answer(
        self,
    ) -> None:
        # sentence 1's quote is a verbatim span of the ANSWER (sentence 2's text): not accepted
        swapped = CitationSupportOutput(
            checks=[sentence_check(True, 1, sentence_quote="La tasa es del 10% [2]."),
                    sentence_check(True, 2)]
        )  # fmt: skip
        model = judge_model(swapped, swapped)
        result = await CitationSupportJudge(_call(model)).judge(make_record())

        assert result.verdict is False
        assert [s["label"] for s in result.details["sentences"]] == ["unverifiable", "supported"]
        assert result.code == "claim_not_verbatim"

    async def test_a_quote_covering_half_of_its_sentence_is_not_the_sentence(self) -> None:
        record = make_record("El plazo es de 24 meses y la tasa es del 10% [1]. Otra [2].")
        half = CitationSupportOutput(
            checks=[
                sentence_check(True, 1, sentence_quote="El plazo es de 24 meses"),
                sentence_check(True, 2, sentence_quote="Otra [2]."),
            ]
        )
        result = await CitationSupportJudge(_call(judge_model(half, half))).judge(record)
        assert result.details["sentences"][0]["label"] == "unverifiable"

    async def test_a_sentence_quote_without_its_marker_is_accepted(self) -> None:
        out = CitationSupportOutput(
            checks=[
                sentence_check(True, 1, sentence_quote="El plazo es de 24 meses."),
                sentence_check(True, 2, sentence_quote="La tasa es del 10%."),
            ]
        )
        result = await CitationSupportJudge(_call(judge_model(out))).judge(make_record())
        assert result.verdict is True

    async def test_a_missing_information_sentence_may_have_empty_evidence(self) -> None:
        record = make_record(
            "Las fuentes no mencionan esa información [1]. La tasa es del 10% [2]."
        )
        out = CitationSupportOutput(
            checks=[
                sentence_check(
                    True,
                    1,
                    sentence_quote="Las fuentes no mencionan esa información [1].",
                    cited=[1],
                    evidence_quote="",
                    states_missing_info=True,
                ),
                sentence_check(True, 2),
            ]
        )
        result = await CitationSupportJudge(_call(judge_model(out))).judge(record)

        assert result.verdict is True
        assert result.details["missing_info"] == 1
        assert result.details["sentences"][0]["states_missing_info"] is True

    async def test_the_missing_info_flag_on_a_sentence_that_asserts_a_fact_is_ignored(self) -> None:
        record = make_record("El plazo para reclamar es de treinta días hábiles [1].")
        out = CitationSupportOutput(
            checks=[
                sentence_check(
                    True,
                    1,
                    sentence_quote="El plazo para reclamar es de treinta días hábiles [1].",
                    evidence_quote="",
                    states_missing_info=True,
                )
            ]
        )
        result = await CitationSupportJudge(_call(judge_model(out))).judge(record)

        row = result.details["sentences"][0]
        assert (result.verdict, row["label"], row["reason"]) == (
            False,
            "not_supported",
            "evidence_not_verbatim",
        )
        assert (row["states_missing_info"], row["missing_info_flag_rejected"]) == (False, True)
        assert (result.details["missing_info"], result.details["missing_info_flag_rejected"]) == (
            0,
            1,
        )
        assert result.code == "evidence_not_verbatim"

    async def test_evidence_may_quote_the_title_and_locator_the_judge_saw(self) -> None:
        # source 1 is rendered as "[1] Norma A (p. 3)\nEl plazo es de 24 meses."
        evidence = "Norma A (p. 3) El plazo es de 24 meses"
        out = FaithfulnessOutput(claims=[claim(True, evidence_quote=evidence)], is_refusal=False)
        result = await FaithfulnessJudge(_call(judge_model(out))).judge(make_record())
        assert result.verdict is True
        assert result.details["claims"][0]["evidence_match"] == "exact"

    async def test_empty_evidence_is_still_downgraded_without_the_missing_info_flag(self) -> None:
        out = CitationSupportOutput(
            checks=[sentence_check(True, 1, evidence_quote=""), sentence_check(True, 2)]
        )
        result = await CitationSupportJudge(_call(judge_model(out))).judge(make_record())
        assert result.verdict is False
        assert result.details["sentences"][0]["reason"] == "evidence_not_verbatim"

    async def test_unparseable_output_is_retried_once(self) -> None:
        model = judge_model("garbage", checks(True, True))
        result = await CitationSupportJudge(_call(model)).judge(make_record())
        assert result.verdict is True
        assert result.details["retries"] == {"parse": 1, "verbatim": 0}
        assert result.usage.calls == 2

    async def test_a_sentence_quote_not_in_the_answer_is_retried_then_unverifiable(self) -> None:
        paraphrased = CitationSupportOutput(
            checks=[sentence_check(True, 1), sentence_check(True, 2, sentence_quote="La tasa: 10%")]
        )
        model = judge_model(paraphrased, paraphrased)
        result = await CitationSupportJudge(_call(model)).judge(make_record())

        assert len(model.structured_seen) == 2
        assert result.verdict is False
        assert [s["label"] for s in result.details["sentences"]] == ["supported", "unverifiable"]
        assert result.code == "claim_not_verbatim"
        assert result.details["retries"]["verbatim"] == 1

    async def test_a_sentence_quote_fixed_by_the_retry_is_judged_normally(self) -> None:
        paraphrased = CitationSupportOutput(
            checks=[
                sentence_check(True, 1, sentence_quote="plazo 24 meses"),
                sentence_check(True, 2),
            ]
        )
        result = await CitationSupportJudge(
            _call(judge_model(paraphrased, checks(True, True)))
        ).judge(make_record())
        assert result.verdict is True

    async def test_supported_with_evidence_not_in_the_cited_sources_becomes_not_supported(
        self,
    ) -> None:
        # span of source 2 for a sentence that cites source 1 only
        wrong = CitationSupportOutput(
            checks=[
                sentence_check(True, 1, evidence_quote="La tasa es del 10%."),
                sentence_check(True, 2),
            ]
        )
        result = await CitationSupportJudge(_call(judge_model(wrong))).judge(make_record())

        assert result.verdict is False
        row = result.details["sentences"][0]
        assert (row["label"], row["model_label"]) == ("not_supported", "supported")
        assert row["reason"] == "evidence_not_verbatim"
        assert result.code == "evidence_not_verbatim"
        assert result.details["guard"]["evidence_not_verbatim"] == 1

    async def test_evidence_spans_of_several_cited_sources_are_joined_with_a_bar(self) -> None:
        both = CitationSupportOutput(
            checks=[
                sentence_check(
                    True,
                    1,
                    sentence_quote="Dato combinado [1][2]",
                    cited=[1, 2],
                    evidence_quote="El plazo es de 24 meses. ... La tasa es del 10%.",
                )
            ]
        )
        record = make_record("Dato combinado [1][2]. Sin cita.")
        result = await CitationSupportJudge(_call(judge_model(both))).judge(record)
        assert result.verdict is True


# --- answer relevance ---------------------------------------------------------------------------


class TestAnswerRelevance:
    @pytest.mark.parametrize("verdict", [True, False])
    async def test_the_verdict_is_the_models(self, verdict: bool) -> None:
        result = await AnswerRelevanceJudge(_call(judge_model(relevance(verdict)))).judge(
            make_record()
        )
        assert result.verdict is verdict
        assert result.status is JudgeStatus.OK

    async def test_it_sees_question_and_answer_only(self) -> None:
        model = judge_model(relevance(True))
        await AnswerRelevanceJudge(_call(model)).judge(make_record())
        user = model.structured_seen[0][1].text
        assert "¿Cuál es el plazo?" in user
        assert "file:///a.pdf" not in user

    async def test_scope_out_of_an_expected_in_question_is_false_without_a_call(self) -> None:
        model = judge_model()
        record = make_record(
            "Mensaje fijo.", expected_scope="in", scope=ScopeVerdict("out", 0.1), sources=()
        )
        result = await AnswerRelevanceJudge(_call(model)).judge(record)

        assert (result.status, result.verdict) == (JudgeStatus.OK, False)
        assert result.code == "in_scope_refused"
        assert result.method == "deterministic"
        assert model.structured_seen == []

    @pytest.mark.parametrize("expected", ["out", None])
    async def test_scope_out_is_not_applicable_otherwise_without_a_call(
        self, expected: Any
    ) -> None:
        model = judge_model()
        record = make_record(
            "Mensaje fijo.", expected_scope=expected, scope=ScopeVerdict("out", 0.1), sources=()
        )
        result = await AnswerRelevanceJudge(_call(model)).judge(record)
        assert (result.status, result.code) == (JudgeStatus.NOT_APPLICABLE, "refusal")
        assert model.structured_seen == []

    async def test_a_refusal_reported_by_the_model_is_not_applicable_without_an_expected_scope(
        self,
    ) -> None:
        model = judge_model(relevance(False, is_refusal=True))
        record = make_record("No lo sé.", expected_scope=None)
        result = await AnswerRelevanceJudge(_call(model)).judge(record)
        assert (result.status, result.code) == (JudgeStatus.NOT_APPLICABLE, "model_refusal")

    async def test_an_out_of_scope_refusal_is_not_applicable_for_relevance(self) -> None:
        model = judge_model(relevance(False, is_refusal=True))
        record = make_record("No lo sé.", expected_scope="out")
        result = await AnswerRelevanceJudge(_call(model)).judge(record)
        assert result.status is JudgeStatus.NOT_APPLICABLE

    async def test_an_in_scope_refusal_the_heuristic_misses_fails_relevance(self) -> None:
        # "No tengo antecedentes" is only recognised by the model: the deterministic judge and
        # the heuristic would let an in-scope refusal pass everything; relevance closes the gap.
        model = judge_model(relevance(False, is_refusal=True))
        record = make_record("Eso escapa a lo que puedo orientar.", expected_scope="in")
        result = await AnswerRelevanceJudge(_call(model)).judge(record)

        assert result.status is JudgeStatus.OK
        assert result.verdict is False
        assert result.code == "in_scope_refused"
        assert result.details["model_is_refusal"] is True
        assert result.details["heuristic_refusal"] is False

    async def test_a_partial_answer_is_judged_not_treated_as_a_refusal(self) -> None:
        model = judge_model(relevance(True, is_refusal=False))
        record = make_record("No sé X, pero el plazo es de 24 meses [1].", expected_scope="in")
        assert (await AnswerRelevanceJudge(_call(model)).judge(record)).verdict is True

    async def test_an_empty_answer_fails_without_a_call(self) -> None:
        model = judge_model()
        result = await AnswerRelevanceJudge(_call(model)).judge(make_record(""))
        assert (result.verdict, result.method) == (False, "deterministic")
        assert model.structured_seen == []


# --- refusal correctness ------------------------------------------------------------------------


class TestRefusalCorrectness:
    judge = RefusalCorrectnessJudge()

    @pytest.mark.parametrize(
        ("expected", "answer", "scope", "verdict"),
        [
            ("out", "Mensaje fijo.", ScopeVerdict("out", 0.1), True),
            ("out", "No lo sé.", ScopeVerdict("in", 0.9), True),
            ("out", "No lo sé.", None, True),
            ("out", "La capital es Canberra.", ScopeVerdict("in", 0.9), False),
            ("in", "El plazo es de 24 meses [1].", ScopeVerdict("in", 0.9), True),
            ("in", "El plazo es de 24 meses [1].", None, True),
            ("in", "Mensaje fijo.", ScopeVerdict("out", 0.1), False),
            ("in", "No tengo información sobre eso.", None, False),
        ],
    )
    async def test_table(
        self, expected: Any, answer: str, scope: ScopeVerdict | None, verdict: bool
    ) -> None:
        result = await self.judge.judge(make_record(answer, expected_scope=expected, scope=scope))
        assert result.status is JudgeStatus.OK
        assert result.method == "deterministic"
        assert result.verdict is verdict
        assert result.usage.calls == 0

    async def test_no_expected_scope_is_not_applicable(self) -> None:
        result = await self.judge.judge(make_record(expected_scope=None))
        assert result.status is JudgeStatus.NOT_APPLICABLE
        assert result.code == "no_expected_scope"
        assert result.judge is JudgeName.REFUSAL_CORRECTNESS

    async def test_the_rationale_names_the_signal(self) -> None:
        by_scope = await self.judge.judge(
            make_record("x", expected_scope="out", scope=ScopeVerdict("out", 0.1))
        )
        by_text = await self.judge.judge(make_record("No lo sé.", expected_scope="out"))
        assert "refuse path" in by_scope.rationale
        assert "text heuristic" in by_text.rationale

    async def test_a_social_refusal_reason_counts_as_a_refusal(self) -> None:
        # a greeting reply carries no scope verdict and no refusal wording
        social = make_record(
            "¡Hola! Pregunta lo que necesites.", sources=(), refusal_reason="social"
        )
        kept = await self.judge.judge(dataclasses.replace(social, expected_scope="out"))
        misrouted = await self.judge.judge(social)  # an in-domain question sent to social

        assert kept.verdict is True
        assert misrouted.verdict is False
        assert "refuse path" in misrouted.rationale


class TestScriptMix:
    judge = ScriptMixJudge()

    async def test_plain_spanish_passes(self) -> None:
        result = await self.judge.judge(make_record("El plazo es de 24 meses [1]."))
        assert (result.status, result.verdict, result.code) == (JudgeStatus.OK, True, None)
        assert result.method == "deterministic"
        assert result.usage.calls == 0

    async def test_a_foreign_script_word_fails_and_is_reported(self) -> None:
        result = await self.judge.judge(make_record("El plazo es नमस्ते de 24 meses [1]."))
        assert (result.verdict, result.code) == (False, "foreign_script")
        assert result.details["tokens"] == ["नमस्ते"]
        assert "नमस्ते" in result.rationale

    async def test_a_character_of_the_sources_or_the_question_is_allowed(self) -> None:
        sources = (CitedSource(1, "Tasa", "uri", "La tasa β es del 2%."),)
        result = await self.judge.judge(make_record("La tasa β es del 2% [1].", sources=sources))
        assert result.verdict is True

    async def test_an_empty_answer_is_not_applicable(self) -> None:
        result = await self.judge.judge(make_record("  "))
        assert (result.status, result.code) == (JudgeStatus.NOT_APPLICABLE, "empty_answer")


# --- the call wrapper ---------------------------------------------------------------------------


class TestJudgeCall:
    async def test_runs_are_tagged_for_the_tracer(self) -> None:
        model = tracking_model(relevance(True))
        await AnswerRelevanceJudge(JudgeCall(model, LABEL)).judge(make_record(id="rec-9"))

        run = model.state["runs"][0]
        assert "agintis:judge" in run["tags"]
        assert "judge:answer_relevance" in run["tags"]
        assert run["metadata"]["agintis_component"] == "answer_judge"
        assert run["metadata"]["judge"] == "answer_relevance"
        assert run["metadata"]["record_id"] == "rec-9"
        assert run["metadata"]["judged_agent"] == "rag"
        assert run["metadata"]["judge_model"] == LABEL

    async def test_calls_in_flight_never_exceed_the_semaphore(self) -> None:
        model = tracking_model(relevance(True), delay=0.03)
        judge = AnswerRelevanceJudge(JudgeCall(model, LABEL, asyncio.Semaphore(2)))

        results = await asyncio.gather(*(judge.judge(make_record(id=f"r{i}")) for i in range(8)))

        assert all(r.verdict for r in results)
        assert model.state["peak"] == 2

    async def test_the_structured_binding_is_a_non_streamed_copy(self) -> None:
        model = judge_model(relevance(True))
        await AnswerRelevanceJudge(_call(model)).judge(make_record())
        assert model.disable_streaming is not True  # the original is never mutated
