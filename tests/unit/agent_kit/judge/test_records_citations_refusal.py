import hashlib
import json
import random
from dataclasses import replace

import pytest

from agintis.agent_kit.judge import (
    AnswerRecord,
    CitedSource,
    RecordFormatError,
    ScopeVerdict,
    citation_numbers,
    citations,
    cited_sentences,
    dangling_citations,
    is_refusal,
    looks_like_refusal,
    record_from_json,
    record_from_state,
    split_sentences,
)
from agintis.agent_kit.judge.render import citation_messages
from agintis.agent_kit.nodes import SourceLimits
from tests.unit.agent_kit.judge.support import make_record

# --- record schema ------------------------------------------------------------------------------


def test_record_roundtrips_through_json() -> None:
    record = make_record(scope=ScopeVerdict("low", 0.5123))
    again = record_from_json(json.loads(json.dumps(record.to_json(), ensure_ascii=False)))
    assert again == record
    assert record.to_json()["sources"][0] == {
        "n": 1,
        "title": "Norma A",
        "locator": "p. 3",
        "uri": "file:///a.pdf",
        "text": "El plazo es de 24 meses.",
    }


def test_record_without_scope_or_expected_scope_roundtrips() -> None:
    record = AnswerRecord(id="x", question="q", answer="a", agent="rag", citations="inline")
    assert record_from_json(record.to_json()) == record
    assert record.to_json()["scope"] is None
    assert record.to_json()["expected_scope"] is None


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"schema": 2}, "unsupported schema"),
        ({"expected_scope": "maybe"}, "expected_scope"),
        ({"answer": None}, "'answer'"),
        ({"scope": {"level": "wat", "score": 1.0}}, "scope level"),
        ({"scope": {"level": "in", "score": "high"}}, "scope score"),
        ({"models": {"a": 1}}, "'models'"),
        ({"meta": []}, "'meta'"),
        ({"sources": "none"}, "'sources'"),
        ({"citations": "footnotes"}, "citations must be one of"),
        ({"sources": [{"n": 2, "title": None, "uri": "u", "text": "t"}]}, "numbered 1..N"),
        ({"sources": [{"n": 1, "title": None, "uri": 3, "text": "t"}]}, "'uri'"),
    ],
)
def test_malformed_records_are_rejected(patch: dict[str, object], message: str) -> None:
    row = {**make_record().to_json(), **patch}
    with pytest.raises(RecordFormatError, match=message):
        record_from_json(row)


def test_a_record_without_citations_loads_as_inline_and_none_roundtrips() -> None:
    old = make_record().to_json()
    del old["citations"]  # a dump written before the option
    assert record_from_json(old).citations == "inline"

    record = replace(make_record("El plazo es de 24 meses."), citations="none")
    assert record_from_json(json.loads(json.dumps(record.to_json()))) == record
    # not part of what a label depends on
    assert record.answer_sha256 == replace(record, citations="inline").answer_sha256


def test_record_from_state_carries_the_citation_style() -> None:
    state = {"answer": "a", "sources": [], "messages": []}
    record = record_from_state(
        state, id="x", question="q", agent="a", limits=SourceLimits(), citations="none"
    )
    assert record.citations == "none"


def test_a_non_object_record_is_rejected() -> None:
    with pytest.raises(RecordFormatError, match="JSON object"):
        record_from_json([1, 2])


# --- citations ----------------------------------------------------------------------------------


def test_citation_numbers_are_distinct_and_ordered() -> None:
    assert citation_numbers("a [2] b [1][2] c [10]") == (2, 1, 10)
    assert citation_numbers("sin citas") == ()


def test_split_sentences_breaks_on_stops_and_lines() -> None:
    text = "Primera frase [1]. Segunda frase? Tercera!\n- item uno [2]\n- item dos"
    assert split_sentences(text) == [
        "Primera frase [1].",
        "Segunda frase?",
        "Tercera!",
        "- item uno [2]",
        "- item dos",
    ]


def test_a_marker_after_the_full_stop_is_glued_to_its_sentence() -> None:
    sentences = cited_sentences("El plazo es de 24 meses. [1] La tasa es del 10%. [2][3]")
    assert [(s.index, s.numbers) for s in sentences] == [(1, (1,)), (2, (2, 3))]
    assert sentences[0].text == "El plazo es de 24 meses. [1]"


def test_only_sentences_with_markers_are_cited_and_indexed_among_themselves() -> None:
    sentences = cited_sentences("Hola. Dato uno [1]. Sin cita. Dato dos [2].")
    assert [(s.index, s.text) for s in sentences] == [(1, "Dato uno [1]."), (2, "Dato dos [2].")]
    assert cited_sentences("No hay citas aquí.") == []


@pytest.mark.parametrize(
    ("answer", "count", "expected"),
    [
        ("a [1] b [2]", 2, ()),
        ("a [3]", 2, (3,)),
        ("a [0] b [1]", 1, (0,)),
        ("a [1]", 0, (1,)),
        ("sin citas", 0, ()),
    ],
)
def test_dangling_citations(answer: str, count: int, expected: tuple[int, ...]) -> None:
    assert dangling_citations(answer, count) == expected


# --- refusal detection --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "No lo sé.",
        "NO LO SE con la información disponible.",
        "No tengo información sobre eso en los documentos.",
        "Las fuentes no contienen la respuesta a tu pregunta.",
        "Solo puedo responder preguntas sobre la normativa.",
    ],
)
def test_text_heuristic_detects_refusals(answer: str) -> None:
    assert looks_like_refusal(answer, citations="inline")


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "El plazo es de 24 meses.",
        "No lo sé, pero el plazo es de 24 meses [1].",  # it cites: it is answering
        "x" * 700 + " no lo sé",  # too long to be a blanket refusal
    ],
)
def test_text_heuristic_does_not_flag_answers(answer: str) -> None:
    assert not looks_like_refusal(answer, citations="inline")


def test_scope_out_is_a_refusal_whatever_the_text_says() -> None:
    assert is_refusal(make_record("Texto cualquiera.", scope=ScopeVerdict("out", 0.1)))


def test_other_scopes_fall_back_to_the_text() -> None:
    assert not is_refusal(make_record("Respuesta [1].", scope=ScopeVerdict("low", 0.5)))
    assert is_refusal(make_record("No lo sé.", scope=ScopeVerdict("in", 0.9)))


def test_cited_source_is_a_plain_value() -> None:
    assert CitedSource(1, None, "u", "t") == CitedSource(1, None, "u", "t")


# --- review fixes: records ----------------------------------------------------------------------


def test_a_refusal_without_evidence_keeps_its_scope_with_a_null_score() -> None:
    record = make_record("Mensaje fijo.", scope=ScopeVerdict("out", None), sources=())
    row = json.loads(json.dumps(record.to_json()))
    assert row["scope"] == {"level": "out", "score": None}
    assert record_from_json(row) == record


def test_the_answer_hash_covers_the_answer_and_the_rendered_sources() -> None:
    base = make_record()
    other_answer = make_record("Otra respuesta [1].")
    other_source = make_record(
        sources=(
            CitedSource(1, "Norma A", "file:///a.pdf", "Otro texto.", "p. 3"),
            *base.sources[1:],
        )
    )
    other_locator = make_record(
        sources=(CitedSource(1, "Norma A", "file:///a.pdf", base.sources[0].text, "p. 4"),
                 *base.sources[1:])
    )  # fmt: skip
    hashes = {r.answer_sha256 for r in (base, other_answer, other_source, other_locator)}
    assert len(hashes) == 4
    assert base.to_json()["answer_sha256"] == base.answer_sha256


def test_an_edited_record_is_rejected_on_load() -> None:
    row = make_record().to_json()
    row["answer"] = "Texto editado a mano [1]."
    with pytest.raises(RecordFormatError, match="answer_sha256"):
        record_from_json(row)
    row = make_record().to_json()
    del row["answer_sha256"]
    with pytest.raises(RecordFormatError, match="answer_sha256"):
        record_from_json(row)


# --- review fixes: citations --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dato [1, 2]", (1, 2)),
        ("dato [1,2,3]", (1, 2, 3)),
        ("dato [1-3]", (1, 2, 3)),
        ("dato [1\u20133]", (1, 2, 3)),  # en dash
        ("dato [3-1]", (1, 2, 3)),
        ("dato [0001]", (1,)),
        ("dato [1, 3-4] y [2]", (1, 3, 4, 2)),
        ("dato [1-999]", (1, 999)),  # absurd range: only its ends
        ("dato [2024]", ()),  # a year is not a citation
        ("dato [1, 2024]", ()),
        ("dato [000012]", (12,)),
        ("dato [a]", ()),
        ("dato [12345]", ()),
    ],
)
def test_citation_marker_forms(text: str, expected: tuple[int, ...]) -> None:
    assert citation_numbers(text) == expected


def test_a_dangling_number_inside_a_list_or_range_is_caught() -> None:
    assert dangling_citations("dato [1, 7]", 3) == (7,)
    assert dangling_citations("dato [2-5]", 3) == (4, 5)
    assert dangling_citations("dato [0001]", 3) == ()


def test_cited_sentences_carry_every_number_of_a_multi_marker() -> None:
    sentences = cited_sentences("Dato combinado [1, 2]. Otro [3-4].")
    assert [s.numbers for s in sentences] == [(1, 2), (3, 4)]


@pytest.mark.parametrize(
    "text",
    [
        "Según el art. 5 del Código el plazo es de 24 meses [1].",
        "Conforme al D.L. 3.500 el plazo es de 24 meses [1].",
        "Lo dispone la Ley N° 20.255 y el inc. segundo del art. 12 [1].",
        "El Sr. Superintendente resolvió el plazo [1].",
        "Se aplica a los afiliados (art. 3 y sig.) en general [1].",
        "El plazo es de 24 meses, es decir, 2 años. y además queda sujeto a revisión [1].",
    ],
)
def test_abbreviations_do_not_cut_a_sentence(text: str) -> None:
    assert split_sentences(text) == [text]


def test_a_real_sentence_end_still_cuts() -> None:
    assert split_sentences("El plazo es corto. La tasa es alta. Ok? Sí.") == [
        "El plazo es corto.",
        "La tasa es alta.",
        "Ok?",
        "Sí.",
    ]


# --- review fixes: the refusal heuristic --------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "No sé cuál es el plazo.",
        "No sé si eso aplica.",
        "La información no está disponible en las fuentes proporcionadas.",
        "Las fuentes proporcionadas no contienen esa información.",
        "Los documentos consultados no mencionan esa información.",
        "No puedo confirmarlo con los documentos.",
        "Los documentos entregados no abordan el tema.",
        "No tengo antecedentes sobre eso.",
        "No hay referencias a ese plazo en las fuentes.",
        "Las fuentes no lo dicen.",
        "No se dispone de información.",
        "No encuentro esa información.",
        "No está en las fuentes.",
        "Desconozco la respuesta.",
        "No tengo la respuesta a eso.",
        "No cuento con información suficiente.",
        "Lo siento, no encontré nada relacionado con tu pregunta.",
    ],
)
def test_the_widened_markers_catch_the_reviewers_false_negatives(answer: str) -> None:
    assert looks_like_refusal(answer, citations="inline")


@pytest.mark.parametrize(
    "answer",
    [
        "El afiliado no se encuentra obligado a cotizar después de los 65 años.",
        "No aparece un límite, el plazo es de 30 días.",
        "El empleador no se encuentra exento de la obligación de pagar.",
        "La norma no se aplica a los trabajadores independientes.",
        "Eso no está permitido por la ley.",
        "Los documentos se encuentran disponibles en la sucursal.",
        "Mi opinión es que no sería sensato esperar.",
        "La normativa no permite retiros anticipados.",
        "El texto no indica un plazo distinto, por lo que rige el general.",
        "Los documentos no permiten el traspaso entre fondos distintos.",
        "Las fuentes no tienen carácter vinculante para las AFP.",
    ],
)
def test_the_narrowed_markers_do_not_flag_answers_that_merely_say_no(answer: str) -> None:
    assert not looks_like_refusal(answer, citations="inline")


def test_word_boundaries_stop_substring_matches() -> None:
    assert not looks_like_refusal(
        "Conozco bien el tema: el plazo es de un año.", citations="inline"
    )
    # "no lo se" inside a word
    assert not looks_like_refusal("Innovacion no lo sedimenta.", citations="inline")


# --- review fixes: the hash does not depend on the rendering code -------------------------------


def test_the_answer_hash_is_the_sha256_of_a_canonical_json_of_the_raw_fields() -> None:
    record = make_record("Respuesta [1].", sources=(CitedSource(1, "Título ñ", "u", "t", None),))
    canonical = (
        '{"answer":"Respuesta [1].","expected_scope":"in","question":"¿Cuál es el plazo?",'
        '"sources":[{"locator":null,"n":1,"text":"t","title":"Título ñ","uri":"u"}]}'
    )
    assert record.answer_sha256 == hashlib.sha256(canonical.encode()).hexdigest()


def test_changing_how_sources_are_rendered_never_changes_the_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agintis.agent_kit import source_text

    record = make_record()
    before = record.answer_sha256
    monkeypatch.setattr(source_text, "render_source", lambda n, item: f"SOMETHING ELSE {n}")
    assert record.answer_sha256 == before
    assert record_from_json(record.to_json()) == record  # and it still verifies on load


def test_the_hash_covers_the_question_and_the_expected_scope() -> None:
    base = make_record()
    relabelled = make_record(expected_scope="out")  # an audit relabel out -> in changes the label
    requested = replace(base, question="Otra pregunta")
    assert len({base.answer_sha256, relabelled.answer_sha256, requested.answer_sha256}) == 3

    row = base.to_json()
    row["expected_scope"] = "out"  # relabelled after the hash was taken: rejected on load
    with pytest.raises(RecordFormatError, match="answer_sha256"):
        record_from_json(row)


# --- review fixes: articles and contrastive continuations ----------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "El texto no indica la fecha exacta, pero el plazo es 30 días.",
        "El texto no indica la fecha exacta.",
        "Los documentos no mencionan las tasas vigentes, solo el procedimiento.",
        "No sé cuál es mejor, pero el fondo A tiene menor riesgo.",
        "No tengo la respuesta exacta, sin embargo el compendio fija 24 meses.",
    ],
)
def test_an_article_or_a_contrastive_continuation_that_asserts_something_is_not_a_refusal(
    answer: str,
) -> None:
    assert not looks_like_refusal(answer, citations="inline")


@pytest.mark.parametrize(
    "answer",
    [
        "Las fuentes no lo dicen.",
        "Los documentos no abordan eso.",
        "No lo sé, pero puedes reformular la pregunta.",
        "No tengo antecedentes sobre eso; consulta a tu AFP.",
        "No sé la respuesta. Puedes consultar el sitio de la Superintendencia.",
    ],
)
def test_pronoun_forms_and_offers_of_help_stay_refusals(answer: str) -> None:
    assert looks_like_refusal(answer, citations="inline")


@pytest.mark.parametrize(
    "answer",
    [
        "Los documentos no mencionan ese plazo.",
        "En los documentos no se menciona el plazo de preaviso.",
        "La información proporcionada no especifica el monto.",
        "Los extractos disponibles no tratan ese tema.",
    ],
)
def test_near_domain_refusals_stay_out_of_the_narrow_heuristic_on_purpose(answer: str) -> None:
    # Telling "the sources are silent about X" from "the text says something about X" needs the
    # judge model: the panel's hybrid refusal_correctness takes it from the LLM judges.
    assert not looks_like_refusal(answer, citations="inline")


@pytest.mark.parametrize(
    "answer",
    [
        "No encontré información sobre el plazo de preaviso en los documentos; el Código del "
        "Trabajo no está entre ellos.",
        # Indistinguishable from the case above without a model, so it stays a refusal here
        # (the judge models decide it in the hybrid refusal_correctness).
        "No puedo afirmar que exista un plazo; el general es de 30 días.",
    ],
)
def test_a_semicolon_continuation_is_not_a_contrast(answer: str) -> None:
    assert looks_like_refusal(answer, citations="inline")


@pytest.mark.parametrize(
    "answer",
    [
        "No tengo información sobre ese plazo, pero el afiliado puede solicitar el retiro "
        "en su AFP.",
        "No sé cuál es el monto, pero la pregunta se resuelve en el artículo 5.",
        "No lo sé, pero el trabajador puede consultar la norma vigente.",
    ],
)
def test_a_bare_puede_or_consulta_is_a_fact_not_an_offer_of_help(answer: str) -> None:
    assert not looks_like_refusal(answer, citations="inline")


@pytest.mark.parametrize(
    "answer",
    [
        "No lo sé, pero puedes reformular la pregunta.",
        "No tengo información sobre eso, pero te recomiendo consultar a tu AFP.",
        "No lo sé, pero puedes consultar en el sitio de la Superintendencia.",
        "No encontré nada, pero intenta con otras palabras.",
        "No lo sé, pero contacta a tu AFP.",
    ],
)
def test_offers_directed_at_the_user_keep_the_refusal(answer: str) -> None:
    assert looks_like_refusal(answer, citations="inline")


def test_a_lone_surrogate_in_model_output_is_scrubbed_and_never_crashes_the_hash() -> None:
    bad = make_record("Respuesta rota \ud83d [1].")
    assert len(bad.answer_sha256) == 64  # surrogatepass: no UnicodeEncodeError

    state = {"answer": "Respuesta rota \ud83d [1].", "sources": [], "messages": []}
    record = record_from_state(
        state, id="x", question="q", agent="a", limits=SourceLimits(), citations="inline"
    )
    assert "\ufffd" in record.answer
    assert "\ud83d" not in record.answer
    record.answer.encode("utf-8")  # writable
    assert record_from_json(json.loads(json.dumps(record.to_json()))) == record


_CODE = "```python\nfrom fastapi import FastAPI\napp = FastAPI()\n```"


def test_fenced_block_joins_its_intro_and_trailing_marker_on_the_next_line() -> None:
    answer = f"Por ejemplo:\n{_CODE}\n[1]"
    assert split_sentences(answer) == [f"Por ejemplo:\n{_CODE} [1]"]
    assert [s.numbers for s in cited_sentences(answer)] == [(1,)]


def test_fenced_block_with_the_marker_on_the_closing_line() -> None:
    answer = "Por ejemplo: [1]\n```python\nx = 1\n\ny = 2\n``` [2]"
    assert split_sentences(answer) == ["Por ejemplo: [1]\n```python\nx = 1\n\ny = 2\n``` [2]"]
    assert cited_sentences(answer)[0].numbers == (1, 2)


def test_tilde_fence_and_longer_closing_fence() -> None:
    assert split_sentences("Mira:\n~~~\na. b.\n~~~\n[1]") == ["Mira:\n~~~\na. b.\n~~~ [1]"]
    # a shorter or different fence inside does not close the block
    answer = "Mira:\n````\n```\nx. y.\n````\n[1] Fin."
    assert split_sentences(answer) == ["Mira:\n````\n```\nx. y.\n```` [1]", "Fin."]
    assert split_sentences("A:\n~~~\n```\nz. w.\n~~~") == ["A:\n~~~\n```\nz. w.\n~~~"]


def test_unclosed_fence_runs_to_the_end() -> None:
    assert split_sentences("Ver [1]:\n```\na. b.\nc.") == ["Ver [1]:\n```\na. b.\nc."]
    # a shorter fence does not close it either
    assert split_sentences("Ver:\n````\na.\n```\nb. c.") == ["Ver:\n````\na.\n```\nb. c."]


def test_closing_fence_followed_by_text_closes_and_the_text_continues_as_prose() -> None:
    answer = "Mira:\n```\nx = 1\n``` Esto asigna x [2].\nFin."
    assert split_sentences(answer) == ["Mira:\n```\nx = 1\n```", "Esto asigna x [2].", "Fin."]
    assert split_sentences("Mira:\n```\nx = 1\n```.\nFin.") == ["Mira:\n```\nx = 1\n```", "Fin."]


def test_fence_with_an_info_string_inside_a_block_does_not_close_it() -> None:
    answer = "Mira:\n```\n```js\nx\n```\nFin."
    assert split_sentences(answer) == ["Mira:\n```\n```js\nx\n```", "Fin."]


def test_fences_indented_differently_still_pair() -> None:
    assert split_sentences("Mira:\n   ```\nx\n```\n[1]") == ["Mira:\n   ```\nx\n``` [1]"]
    assert split_sentences("Mira:\n```\nx\n  ```\n[1]") == ["Mira:\n```\nx\n  ``` [1]"]


def test_block_after_a_closed_sentence_is_its_own_unit() -> None:
    answer = "A [1].\n```py\nx\n```"
    assert split_sentences(answer) == ["A [1].", "```py\nx\n```"]
    assert [s.text for s in cited_sentences(answer)] == ["A [1]."]
    cited = split_sentences("A [1].\n```py\nx\n```\n[2]")
    assert cited == ["A [1].", "```py\nx\n``` [2]"]


def test_block_glues_to_an_intro_with_markers_or_without_punctuation() -> None:
    assert split_sentences("Ejemplo: [1]\n```\nx\n```") == ["Ejemplo: [1]\n```\nx\n```"]
    assert split_sentences("Ejemplo:\n```\nx\n```\n[1]") == ["Ejemplo:\n```\nx\n``` [1]"]
    assert split_sentences("Ejemplo\n```\nx\n```") == ["Ejemplo\n```\nx\n```"]


def test_consecutive_blocks_are_separate_units() -> None:
    answer = "Dos:\n```\na\n```\n[1]\n```\nb\n```\n[2]"
    assert split_sentences(answer) == ["Dos:\n```\na\n``` [1]", "```\nb\n``` [2]"]


def test_marker_at_the_start_of_the_line_after_a_fence_belongs_to_the_block() -> None:
    answer = "Mira:\n```\nx\n```\n[2] Además B."
    assert split_sentences(answer) == ["Mira:\n```\nx\n``` [2]", "Además B."]


def test_block_without_markers_stays_uncited_and_following_prose_is_apart() -> None:
    answer = f"Intro:\n{_CODE}\nOtra frase [2]."
    assert split_sentences(answer) == [f"Intro:\n{_CODE}", "Otra frase [2]."]
    assert [s.text for s in cited_sentences(answer)] == ["Otra frase [2]."]


def test_block_at_the_start_is_its_own_unit() -> None:
    assert split_sentences(f"{_CODE}\n[1]") == [f"{_CODE} [1]"]


def test_inline_code_does_not_open_a_fence() -> None:
    assert split_sentences("Usa ```x``` aquí [1]. Luego `y`. Fin.") == [
        "Usa ```x``` aquí [1].",
        "Luego `y`.",
        "Fin.",
    ]


def test_multiline_unit_matches_a_quote_copied_with_collapsed_whitespace() -> None:
    from agintis.agent_kit.judge import guard

    sentence = cited_sentences(f"Por ejemplo:\n{_CODE}\n[1]")[0].text
    quote = "Por ejemplo: ```python from fastapi import FastAPI app = FastAPI() ``` [1]"
    assert guard.match_sentence(quote, sentence) == "exact"


def test_a_long_run_of_spaces_before_a_marker_is_split_in_linear_time() -> None:
    import time

    answer = "Por ejemplo" + " " * 20_000 + "[1]"
    start = time.perf_counter()
    split_sentences(answer)
    assert time.perf_counter() - start < 5  # generous: catches quadratic, not a CI hiccup


# --- markers inside code are not citations (judge v11) ------------------------------------------

_ARRAY_BLOCK = "```python\nconst a = [5, 2, 9];\n```"


def test_markers_inside_a_fenced_block_are_not_citations() -> None:
    answer = f"Por ejemplo:\n{_ARRAY_BLOCK}\nEs un arreglo."
    assert citation_numbers(answer) == ()
    assert dangling_citations(answer, 2) == ()
    assert cited_sentences(answer) == []


def test_a_marker_right_after_the_block_still_counts() -> None:
    answer = f"Por ejemplo:\n{_ARRAY_BLOCK}\n[1]"
    assert citation_numbers(answer) == (1,)
    assert [s.numbers for s in cited_sentences(answer)] == [(1,)]
    assert dangling_citations(answer, 2) == ()


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Usa `x[1]` para indexar.", ()),
        ("Usa ``x[1] y `z` `` aquí [2].", (2,)),
        ("Usa `x[1]` y luego `y[3]`. [2]", (2,)),
        ("Un backtick suelto ` y [1] después.", (1,)),  # unclosed run is literal
        ("Usa ``a[1]` b`` y [2].", (2,)),  # a run only closes with the same length
    ],
)
def test_inline_code_spans_are_not_citations(answer: str, expected: tuple[int, ...]) -> None:
    assert citation_numbers(answer) == expected
    assert dangling_citations(answer, 2) == ()


def test_a_sentence_with_markers_only_in_code_is_uncited() -> None:
    assert [s.text for s in cited_sentences("Usa `x[1]` ahí. Otra frase [2].")] == [
        "Otra frase [2]."
    ]


def test_pure_prose_is_unchanged_by_the_code_rules() -> None:
    answer = "El plazo es de 24 meses [1]. La tasa es del 10% [2, 3]. Sin cita."
    assert [(s.text, s.numbers) for s in cited_sentences(answer)] == [
        ("El plazo es de 24 meses [1].", (1,)),
        ("La tasa es del 10% [2, 3].", (2, 3)),
    ]
    assert dangling_citations(answer, 2) == (3,)


def test_a_sentence_boundary_inside_a_code_span_keeps_the_marker_in_code() -> None:
    answer = "Usa `x = a. B[7]` para eso [1]."
    assert dangling_citations(answer, 1) == ()
    sentences = cited_sentences(answer)
    assert [(s.text, s.numbers) for s in sentences] == [(answer, (1,))]
    record = make_record(answer=answer, sources=(CitedSource(1, "A", "file:///a", "texto"),))
    citation_messages("p", record, sentences)  # no KeyError


def test_citation_messages_assert_every_number_has_a_source() -> None:
    record = make_record(answer="Algo [1].", sources=(CitedSource(1, "A", "file:///a", "texto"),))
    sentence = citations.CitedSentence(1, "Algo [1, 9].", (1, 9))
    with pytest.raises(AssertionError):
        citation_messages("p", record, [sentence])


def test_an_escaped_backtick_does_not_open_a_code_span() -> None:
    assert citation_numbers(r"Mira \`a [1] b` fin") == (1,)
    assert citation_numbers("Mira `a [1] b` fin") == ()


def test_a_refusal_with_a_code_bracket_is_still_a_refusal() -> None:
    answer = "No encontré información sobre `items[0]` en las fuentes."
    assert looks_like_refusal(answer, citations="inline")
    assert not looks_like_refusal(answer + " [1]", citations="inline")


def test_an_agent_without_citations_refuses_only_when_the_answer_opens_with_it() -> None:
    partial = "Se renueva en el mostrador con tu carnet; no hay información sobre el costo."
    assert looks_like_refusal(partial, citations="inline")  # inline: an uncited answer is suspect
    assert not looks_like_refusal(partial, citations="none")
    refusal = "  No encontré información sobre eso en las fuentes."
    assert looks_like_refusal(refusal, citations="none")
    assert is_refusal(replace(make_record(refusal, scope=None), citations="none"))
    # the citation gate still applies
    assert not looks_like_refusal(refusal + " [1]", citations="none")


@pytest.mark.parametrize(
    "answer",
    [
        "Lo siento, no tengo información sobre el costo.",
        "Lamentablemente no lo sé.",
        "En las fuentes no hay información sobre eso.",
        "Lo siento, pero en los documentos proporcionados no encontré nada sobre eso.",
    ],
)
def test_an_uncited_refusal_may_open_with_a_courtesy_or_a_place(answer: str) -> None:
    assert looks_like_refusal(answer, citations="none")


def test_a_lead_in_does_not_turn_a_partial_answer_into_a_refusal() -> None:
    partial = "Lo siento, se renueva en el mostrador; no hay información sobre el costo."
    assert not looks_like_refusal(partial, citations="none")


def test_precheck_and_enumeration_see_the_same_numbers() -> None:
    # a fence line parses differently alone than inside the longer text
    answer = "Abre el bloque. ``` y luego cierra [1]."
    seen = {n for s in cited_sentences(answer) for n in s.numbers}
    assert set(citation_numbers(answer)) == seen

    rng = random.Random(1314)
    alphabet = ["a", "b", ".", "\n", "`", "~", "```", "[1]", "[2]", " "]
    for _ in range(400):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 25)))
        enumerated = {n for s in cited_sentences(text) for n in s.numbers}
        assert set(citation_numbers(text)) == enumerated, text
        dangling = set(dangling_citations(text, 1))
        assert dangling | (set(citation_numbers(text)) & {1}) == enumerated, text
