import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from agintis.agent_kit.judge import AnswerRecord, citations, guard, prompts, refusal
from agintis.agent_kit.judge.guard import (
    is_tolerated,
    is_verbatim,
    is_watched,
    match_evidence,
    match_quote,
    match_sentence,
    normalize,
    strip_markers,
    uncovered_protected,
)
from agintis.agent_kit.judge.llm import judge_meta
from agintis.agent_kit.judge.render import faithfulness_messages
from tests.unit.agent_kit.judge.support import make_record


def is_exact(quote: str, sources: Iterable[str]) -> bool:
    return match_evidence(quote, sources) == "exact"


RSQUO = chr(0x2019)
COMBINING_ACUTE = chr(0x301)


class TestGuard:
    @pytest.mark.parametrize(
        ("quote", "text"),
        [
            ("el plazo es", "Dijo que el plazo es de 24 meses."),
            ("plazo   es\nde", "el plazo es de 24 meses"),  # whitespace collapsed
            ("“hola”", 'dijo "hola"'),  # curly vs straight, both directions
            ('"hola"', "dijo “hola”"),
            (f"it{RSQUO}s", "it's"),
            (f"cafe{COMBINING_ACUTE}", "un caf\u00e9"),  # NFD vs NFC
            ("a &lt; b", "si a < b"),  # copied from the neutralized rendering
            ("El plazo", "el plazo es de 24 meses"),  # only the FIRST word ignores case
            ("Podrá retirar", "Si cotiza, podrá retirar sus fondos"),
        ],
    )
    def test_found(self, quote: str, text: str) -> None:
        assert is_verbatim(quote, text)

    @pytest.mark.parametrize(
        ("quote", "text"),
        [
            ("el Plazo", "el plazo es de 24 meses"),  # no case folding after the first word
            ("cafe", "un café"),  # no accent stripping
            ("el plazo es de 36 meses", "el plazo es de 24 meses"),
            ("cuando ya se ha traspasado", "si aún no se ha traspasado"),
            ("", "anything"),
            ("   ", "anything"),
            ("compatible con la cuenta", "es incompatible con la cuenta de ahorro"),
            ("cuenta de aho", "es compatible con la cuenta de ahorro"),  # ends inside a word
        ],
    )
    def test_not_found(self, quote: str, text: str) -> None:
        assert not is_verbatim(quote, text)

    def test_any_of_several_texts_matches(self) -> None:
        assert is_verbatim("la tasa", ["el plazo", "la tasa es"])
        assert not is_verbatim("la tasa", [])

    SOURCES = ("El plazo es de 24 meses. Otra cosa.", "La tasa es del 10% anual.")

    @pytest.mark.parametrize("sep", [" ... ", "...", " \u2026 ", " [...] ", " | "])
    def test_fragments_split_on_ellipses_and_bars_each_must_be_found(self, sep: str) -> None:
        quote = f"El plazo es de 24 meses{sep}La tasa es del 10% anual"
        assert is_exact(quote, self.SOURCES)

    def test_a_fragment_that_is_not_in_any_source_fails_the_quote(self) -> None:
        quote = "El plazo es de 24 meses ... La tasa es del 20% anual"
        assert not is_exact(quote, self.SOURCES)

    def test_fragments_shorter_than_the_minimum_are_ignored_but_one_long_one_must_match(
        self,
    ) -> None:
        # "xx yy" (5 chars) is ignored, the long fragment matches
        assert is_exact("xx yy ... El plazo es de 24 meses", self.SOURCES)
        # only short fragments: nothing to prove
        assert not is_exact("xx yy ... zz ww", self.SOURCES)
        # the long one decides even next to a matching short one
        assert not is_exact("24 meses ... Un plazo inexistente aqui", self.SOURCES)

    def test_a_short_quote_that_is_a_contiguous_span_still_passes(self) -> None:
        assert is_exact("24 meses", self.SOURCES)
        assert not is_exact("", self.SOURCES)

    def test_inline_footnote_digits_in_the_source_are_tolerated(self) -> None:
        sources = ["La tasa de interés equivalente3 deriva del vector. Fin."]
        assert is_exact("La tasa de interés equivalente deriva", sources)
        assert is_exact("La tasa de interés equivalente3 deriva", sources)
        # conservative: a digit after a space or a longer number is not a footnote mark
        assert not is_exact("tasa de 24 meses", ["tasa de 2 4 meses"])
        assert not is_exact("el plazo es de 2 meses", ["el plazo es de 24 meses"])
        assert not is_exact("la norma 350 vigente", ["la norma3500 vigente"])

    def test_a_source_containing_a_bar_matches_as_a_whole(self) -> None:
        assert is_exact("El plazo es | de 24 meses", ["x El plazo es | de 24 meses y"])

    def test_normalize_is_light(self) -> None:
        assert normalize(f"  A{RSQUO}B \n C ") == "A'B C"


S1 = (
    "El Ente Gestor informa a la Tesorería para que efectúe el depósito dentro del plazo "
    "señalado por el Reglamento [2]."
)
S2 = "El plazo es de treinta días hábiles contados desde la notificación del acta al titular [1]."
S3 = "La licencia es personal y puede cederse previa autorización escrita de la Dirección [1]."
S4 = "El informe se presenta dentro del tercer día siguiente a la recepción del expediente [1]."
ANSWER = f"Primero se presenta la solicitud. {S1} {S2} {S3} {S4}"


def grams(text: str) -> set[tuple[str, ...]]:
    """The lowercase word 5-grams of `text` (the leak check of the prompts' invented examples)."""
    words = [w.lower() for w in re.findall(r"\w+", text)]
    return {tuple(words[i : i + 5]) for i in range(len(words) - 4)}


def plain(sentence: str) -> str:
    return strip_markers(sentence)


class TestTolerantQuote:
    def test_the_exact_quote_is_exact_with_or_without_markers(self) -> None:
        assert match_quote(S1, ANSWER) == "exact"
        assert match_quote(plain(S1), ANSWER) == "exact"

    def test_a_marker_in_the_middle_of_the_answer_sentence_does_not_break_the_match(self) -> None:
        answer = "El Ente Gestor informa a la Tesorería [2] para que efectúe el depósito [1]."
        assert match_quote("El Ente Gestor informa a la Tesorería para que efectúe", answer) == (
            "exact"
        )
        assert match_quote(answer, answer) == "exact"
        assert strip_markers("el plazo [1]. Otro [2][3], fin") == "el plazo. Otro, fin"

    def test_an_inserted_function_word_is_a_fuzzy_match(self) -> None:
        quote = plain(S1).replace("para que efectúe", "para que éste efectúe")
        assert match_quote(quote, ANSWER) == "fuzzy"

    def test_a_substitution_inside_an_equivalence_class_is_a_fuzzy_match(self) -> None:
        quote = plain(S1).replace("a la Tesorería", "a el Tesorería")
        assert match_quote(quote, ANSWER) == "fuzzy"

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("a la Tesorería", "de la Tesorería"),  # preposition for preposition
            ("para que", "por que"),
            ("a la Tesorería", "a una Tesorería"),  # definite -> indefinite
            ("a la Tesorería", "a unas Tesorería"),
            ("el depósito", "un depósito"),
        ],
    )
    def test_a_substitution_outside_a_class_fails(self, old: str, new: str) -> None:
        assert match_quote(plain(S1).replace(old, new), ANSWER) is None

    def test_the_probes_of_the_review_fail(self) -> None:
        answer = "Los clientes pueden transferir sus fondos a la caja que elijan, según rige."
        assert match_quote("transferir sus fondos a la caja que elijan", answer) == "exact"
        assert match_quote("transferir sus fondos de la caja que elijan", answer) is None
        assert match_quote("Los clientes pueden transferir sus fondos a la caja", answer) == "exact"
        assert match_quote("Unos clientes pueden transferir sus fondos a la caja", answer) is None
        by_employer = "El pago lo realiza el empleador por cuenta del cliente y en su nombre ok."
        assert match_quote("lo realiza el empleador para cuenta del cliente", by_employer) is None

    def test_a_demonstrative_pair_is_interchangeable(self) -> None:
        answer = "Dicho plazo vence cuando el cliente cumple la edad exigida por ley vigente."
        assert match_quote(answer.replace("Dicho", "Ese"), answer) == "fuzzy"

    @pytest.mark.parametrize(
        ("sentence", "old", "new"),
        [
            (S2, "treinta", "sesenta"),  # a number word
            (S2, "hábiles", "corridos"),  # a unit
            (S2, "titular", "empleador"),  # an actor
            (S2, "contados", "descontados"),
            (S1, "informa", "oculta"),
            (S1, "Tesorería", "Contraloría"),  # an entity
            (S1, "dentro", "fuera"),  # an antonym
            (S3, "y", "o"),  # a conjunction flips a condition
            (S3, "personal", "transferible"),
            (S4, "tercer", "décimo"),  # an ordinal
        ],
    )
    def test_swapping_a_content_word_fails_even_inside_the_edit_budget(
        self, sentence: str, old: str, new: str
    ) -> None:
        assert f" {old} " in f" {sentence} "
        quote = re.sub(rf"\b{old}\b", new, plain(sentence), count=1)
        assert quote != plain(sentence)
        assert match_quote(quote, ANSWER) is None

    @pytest.mark.parametrize("word", ["no", "todas", "ya", "aún", "solo", "nunca", "puede", "si"])
    def test_inserting_any_non_function_word_fails(self, word: str) -> None:
        quote = plain(S1).replace("para que efectúe", f"para que {word} efectúe")
        assert match_quote(quote, ANSWER) is None

    def test_deleting_a_negation_or_quantifier_of_the_answer_fails(self) -> None:
        negated = ANSWER.replace("para que efectúe", "para que no efectúe")
        assert match_quote(plain(S1), negated) is None
        quantified = ANSWER.replace("informa a la", "informa a todas la")
        assert match_quote(plain(S1), quantified) is None

    def test_inserting_or_changing_a_number_or_a_marker_fails(self) -> None:
        assert match_quote(plain(S1).replace("el depósito", "el depósito 3"), ANSWER) is None
        assert match_quote(plain(S2).replace("treinta", "30"), ANSWER) is None

    def test_the_function_word_list_is_closed_and_has_no_conjunctions(self) -> None:
        for word in ("el", "éste", "dicha", "se", "de", "del", "a", "al", "para", "que", "lo"):
            assert is_tolerated(word)
        for word in ("y", "e", "o", "u", "no", "sin", "todo", "sesenta", "titular"):
            assert not is_tolerated(word)
        assert is_tolerated("Éste")  # lowercased for the membership test
        assert not is_tolerated("este-")  # nothing else is folded
        assert not is_tolerated("estè")  # no accent folding

    def test_three_tolerated_edits_exceed_the_budget(self) -> None:
        quote = plain(S1).replace("para que efectúe", "para que éste los efectúe una")
        assert match_quote(quote, ANSWER) is None

    def test_the_ratio_limits_a_medium_quote(self) -> None:
        # 13 tokens: 10% allows ONE edit; two tolerated edits fail
        quote = plain(S3).replace("La licencia", "El licencia").replace("puede", "éste puede")
        assert match_quote(quote, ANSWER) is None
        assert match_quote(plain(S3).replace("La licencia", "El licencia"), ANSWER) == "fuzzy"

    def test_a_short_quote_has_no_budget(self) -> None:
        assert match_quote("Primero se presenta la solicitud", ANSWER) == "exact"
        assert match_quote("Primero se presenta una solicitud", ANSWER) is None

    def test_unrelated_text_does_not_match(self) -> None:
        unrelated = "Nada que ver con lo que dice la respuesta, en absoluto " * 2
        assert match_quote(unrelated, ANSWER) is None

    def test_the_watch_list_flags_negations_quantifiers_conditions_and_digits(self) -> None:
        assert all(is_watched(t) for t in ("no", "todas", "solo", "sólo", "dentro", "aún"))
        assert all(is_watched(t) for t in ("No", "10", "art5"))
        assert not any(is_watched(t) for t in ("éste", "Tesorería"))


class TestMatchSentence:
    def test_the_whole_sentence_matches_with_or_without_its_marker(self) -> None:
        assert match_sentence(plain(S1), S1) == "exact"
        assert match_sentence(S1, S1) == "exact"

    def test_a_quote_of_another_sentence_fails(self) -> None:
        assert match_sentence(plain(S2), S1) is None

    def test_a_fragment_covering_less_than_90_percent_fails(self) -> None:
        half = "El Ente Gestor informa a la Tesorería para que efectúe"
        assert match_quote(half, S1) == "exact"  # a span of the sentence...
        assert match_sentence(half, S1) is None  # ...but not the sentence

    def test_almost_the_whole_sentence_passes(self) -> None:
        assert match_sentence(plain(S1).removeprefix("El "), S1) == "exact"

    def test_a_function_word_edit_inside_the_sentence_is_fuzzy(self) -> None:
        quote = plain(S1).replace("para que efectúe", "para que éste efectúe")
        assert match_sentence(quote, S1) == "fuzzy"


class TestUncoveredProtected:
    ANSWER = "El socio no podrá transferir su puesto antes de los 60 días [1]."

    def test_a_truncating_quote_leaves_the_negation_number_and_condition_uncovered(self) -> None:
        quote = "podrá transferir su puesto"
        assert is_verbatim(quote, self.ANSWER)  # it matches exactly: that is the hole
        assert uncovered_protected(self.ANSWER, [quote]) == ["no", "antes", "60"]

    def test_a_full_quote_leaves_nothing_uncovered(self) -> None:
        assert uncovered_protected(self.ANSWER, [self.ANSWER]) == []

    def test_several_quotes_cover_the_union_and_markers_are_not_tokens(self) -> None:
        quotes = ["El socio no podrá transferir", "su puesto antes de los 60 días"]
        assert uncovered_protected(self.ANSWER, quotes) == []
        assert "1" not in uncovered_protected(self.ANSWER, [])


FOOTNOTED = (
    "Se estimó la estructura temporal de tasas de interés 1 Sólo se consideró la información "
    "correspondiente a días hábiles bancarios para calcular el exceso de retorno promedio del "
    "período muestral completo en 2019. Los resultados se presentan en el cuadro siguiente para "
    "cada uno de los portafolios evaluados durante el período indicado y se comparan con el "
    "índice de referencia del mercado local sin ajustes adicionales por riesgo."
)
WORDS = FOOTNOTED.split()
SKIPPING = " ".join([*WORDS[3:9], *WORDS[10:56]])  # the quote that jumps over the footnote "1"
CONTIGUOUS = WORDS[10:56]  # an exact span, 46 words


class TestFuzzyEvidence:
    def test_a_quote_that_skips_an_injected_footnote_is_a_fuzzy_match(self) -> None:
        assert match_evidence(SKIPPING, [FOOTNOTED]) == "fuzzy"
        assert not is_exact(SKIPPING, [FOOTNOTED])  # the exact rules alone refuse it

    def test_an_inserted_negation_fails(self) -> None:
        words = [*CONTIGUOUS[:20], "no", *CONTIGUOUS[20:]]
        assert match_evidence(" ".join(CONTIGUOUS), [FOOTNOTED]) == "exact"
        assert match_evidence(" ".join(words), [FOOTNOTED]) is None

    def test_a_prefixed_nunca_fails(self) -> None:
        assert match_evidence(" ".join(["nunca", *CONTIGUOUS]), [FOOTNOTED]) is None

    @pytest.mark.parametrize("edge", [0, -1])
    def test_swapping_an_edge_word_fails(self, edge: int) -> None:
        words = list(CONTIGUOUS)
        words[edge] = "otro"
        assert match_evidence(" ".join(words), [FOOTNOTED]) is None

    def test_an_inserted_function_word_is_tolerated(self) -> None:
        words = [*CONTIGUOUS[:20], "la", *CONTIGUOUS[20:]]
        assert match_evidence(" ".join(words), [FOOTNOTED]) == "fuzzy"

    def test_real_words_recombined_in_a_new_order_fail(self) -> None:
        quote = (
            "el exceso de retorno promedio se consideró correspondiente a la estructura "
            "temporal de días hábiles bancarios para calcular la información del período"
        )
        assert match_evidence(quote, [FOOTNOTED]) is None

    def test_a_changed_number_fails_even_when_the_shingles_cover_most(self) -> None:
        words = [w.replace("2019.", "2018.") for w in CONTIGUOUS]
        assert "2018." in words
        assert match_evidence(" ".join(words), [FOOTNOTED]) is None

    def test_a_number_that_exists_elsewhere_in_the_sources_does_not_help(self) -> None:
        sources = [FOOTNOTED, "Otro dato: 2018."]
        words = [w.replace("2019.", "2018.") for w in CONTIGUOUS]
        assert match_evidence(" ".join(words), sources) is None

    def test_a_fragment_under_five_words_keeps_the_exact_rule(self) -> None:
        assert match_evidence("estructura temporal tasas interés", [FOOTNOTED]) is None
        assert match_evidence("estructura temporal de tasas", [FOOTNOTED]) == "exact"

    def test_each_fragment_is_judged_and_exact_ones_stay_exact(self) -> None:
        quote = (
            "Se estimó la estructura temporal ... días hábiles bancarios para calcular el "
            "exceso de retorno promedio del período muestral"
        )
        assert match_evidence(quote, [FOOTNOTED]) == "exact"

    def test_a_four_dot_ellipsis_leaves_no_stray_dot(self) -> None:
        quote = "El plazo es de 24 meses.... La tasa es del 10% anual"
        sources = ("El plazo es de 24 meses. Otra cosa.", "La tasa es del 10% anual.")
        assert match_evidence(quote, sources) == "exact"
        quote = "[...] El plazo es de 24 meses [...] .La tasa es del 10% anual."
        assert match_evidence(quote, sources) == "exact"

    def test_a_short_fragment_with_a_number_or_negation_is_never_ignored(self) -> None:
        sources = ("El plazo es de 24 meses. Otra cosa.", "La tasa es del 10% anual.")
        assert match_evidence("xx yy ... El plazo es de 24 meses", sources) == "exact"  # ignored
        assert match_evidence("25 días ... El plazo es de 24 meses", sources) is None
        assert match_evidence("no ... El plazo es de 24 meses", sources) is None
        assert match_evidence("10% ... El plazo es de 24 meses", sources) == "exact"

    def test_a_short_watched_fragment_matches_whole_words_only(self) -> None:
        sources = ("Ninguno de los plazos cambia en la práctica.",)
        assert match_evidence("no ... Ninguno de los plazos cambia", sources) is None


class TestQuoteAlignment:
    def test_a_short_quote_differing_only_in_markup_is_fuzzy_not_refused(self) -> None:
        answer = "Requisito: tener 65 años cumplidos [1]. Otra cosa distinta."
        assert match_quote("**Requisito**: tener 65 años", answer) == "fuzzy"
        assert match_quote("**Requisito**: tener 66 años", answer) is None

    def test_only_the_first_word_of_a_quote_ignores_case(self) -> None:
        tail = "sus fondos en una sola cuota dentro del plazo señalado por la ley"
        answer = f"Si cumple los requisitos, podrá retirar {tail}."
        assert match_quote(f"Podrá retirar {tail}", answer) == "exact"
        assert match_quote(f"Podrá retirar {tail.upper()}", answer) is None
        # the fuzzy path too: the first word's case, plus one tolerated insertion
        assert match_quote(f"Podrá retirar éste {tail}", answer) == "fuzzy"

    def test_the_verdict_does_not_depend_on_an_earlier_sentence_with_the_same_article(self) -> None:
        sentence = "El Ente Gestor informa a la Tesorería para que efectúe el depósito debido."
        quote = sentence.replace("para que efectúe", "para que éste efectúe")
        earlier = "El Ente cierra el trámite. El comité revisa el caso del interesado. "
        assert match_quote(quote, sentence) == "fuzzy"
        assert match_quote(quote, earlier + sentence) == "fuzzy"
        assert match_quote(quote, sentence + " " + earlier) == "fuzzy"

    def test_uncovered_protected_counts_only_quotes_that_matched(self) -> None:
        answer = "El socio no podrá transferir su puesto antes de los 60 días [1]."
        assert uncovered_protected(answer, ["El socio no podrá transferir su inventado"]) == [
            "no",
            "antes",
            "60",
        ]

    def test_number_words_and_ordinals_are_watched(self) -> None:
        for word in ("treinta", "Quince", "cien", "mil", "millones", "mitad", "doble", "tercio"):
            assert is_watched(word)
        for word in ("dieciséis", "veintidós", "doscientos", "trescientas", "primer", "tercera"):
            assert is_watched(word)
        for word in ("décimo", "undécimo", "vigésima", "último", "penúltima", "anterior"):
            assert is_watched(word)
        assert not any(is_watched(w) for w in ("ciencia", "pacientes", "un", "una", "plazo"))
        answer = "Debe pagar la mitad en el segundo mes y treinta días después."
        assert uncovered_protected(answer, ["Debe pagar"]) == [
            "mitad",
            "segundo",
            "treinta",
            "después",
        ]

    def test_a_short_evidence_fragment_with_a_number_word_must_be_exact(self) -> None:
        sources = ("El plazo es de treinta días hábiles. La tasa es del 10% anual.",)
        assert match_evidence("treinta ... La tasa es del 10% anual", sources) == "exact"
        assert match_evidence("sesenta ... La tasa es del 10% anual", sources) is None
        assert match_evidence("segundo ... La tasa es del 10% anual", sources) is None


class TestEvidenceShortcut:
    SOURCES = ("El plazo de notificación es de 24 meses.", "No se admiten prórrogas.")

    @pytest.mark.parametrize("quote", [".", "de", "plazo", "notif"])
    def test_a_short_quote_without_a_watched_token_proves_nothing(self, quote: str) -> None:
        assert match_evidence(quote, self.SOURCES) is None

    def test_a_short_watched_quote_must_be_a_whole_word_span(self) -> None:
        assert match_evidence("24 meses", self.SOURCES) == "exact"
        assert match_evidence("No se", self.SOURCES) == "exact"
        sources = ("La notificación llegó.",)
        assert match_evidence("no", sources) is None

    def test_a_long_whole_quote_is_exact(self) -> None:
        assert match_evidence("plazo de notificación", self.SOURCES) == "exact"

    def test_double_dots_split_fragments(self) -> None:
        quote = "El plazo de notificación es..de 24 meses"
        assert match_evidence(quote, self.SOURCES) == "exact"


class TestEdgeAndMarkup:
    def test_underscore_is_a_word_character_not_an_edge(self) -> None:
        assert "_" not in guard._EDGE
        source = ['Se ejecuta si el nombre es "__main__" al iniciar.']
        assert is_exact('si el nombre es "__main__".', source)
        assert not is_exact('si el nombre es "__main".', source)

    def test_emphasis_pattern_is_part_of_the_hashed_parameters(self) -> None:
        assert guard._EMPHASIS.pattern in guard.parameters()["patterns"]
        assert "_" not in guard.parameters()["edge_chars"]

    def test_html_tags_are_not_stripped_from_sources(self) -> None:
        source = ["Un parámetro <strong>REQUERIDO</strong> debe enviarse siempre en la petición."]
        assert match_evidence("Un parámetro **REQUERIDO** debe enviarse siempre", source) is None

    @pytest.mark.parametrize("word", ["sesenta", "no", "nunca"])
    def test_emphasized_watched_fragment_must_be_in_the_source(self, word: str) -> None:
        source = ["El cliente puede inscribirse por internet al cumplir la edad legal."]
        quote = f"El cliente puede inscribirse por internet … _{word}_"
        assert match_evidence(quote, source) is None

    def test_emphasized_watched_fragment_present_in_the_source_passes(self) -> None:
        source = ["El cliente puede inscribirse por internet al cumplir sesenta años."]
        quote = "El cliente puede inscribirse por internet … _sesenta_"
        assert match_evidence(quote, source) == "exact"

    def test_emphasis_with_underscores_inside_is_not_unwrapped(self) -> None:
        quote = "__init__ y __main__ … se usan en el arranque del módulo"
        source = ["Los nombres __init__ y __main__ se usan en el arranque del módulo."]
        assert match_evidence(quote, source) == "exact"
        assert guard._trim("_el plazo es de 30 días_") == "el plazo es de 30 días"
        assert guard._trim("__init__ y __main__") == "__init__ y __main__"

    def test_markdown_italics_quote_is_exact(self) -> None:
        source = "Para el trámite, el plazo es de 30 días hábiles."
        assert match_evidence("_el plazo es de 30 días hábiles_", [source]) == "exact"
        assert match_evidence("__el plazo es de 30 días hábiles__.", [source]) == "exact"

    def test_dunder_name_quote_still_passes(self) -> None:
        assert guard._trim('"__main__".') == "__main__"
        assert match_evidence('"__main__": run()', ['if __name__ == "__main__": run()']) == "exact"


class TestPrompts:
    def test_faithfulness_prompt_has_the_contrastive_examples_and_the_supported_ones(self) -> None:
        prompt = prompts.FAITHFULNESS_PROMPT
        assert "label: contradicted" in prompt
        assert "label: supported" in prompt
        for axis in ("polaridad", "condición", "cuantificador", "números", "actor"):
            assert axis in prompt
        assert "TEXTUALMENTE" in prompt
        assert "aunque el resto" in prompt  # one changed axis is enough
        assert prompt.index("claim_quote:") < prompt.index("evidence_quote:")

    def test_the_prompts_state_the_principles(self) -> None:
        prompt = prompts.FAITHFULNESS_PROMPT
        assert "MENOS específica" in prompt
        assert "omitir un elemento de una enumeración" in prompt  # harmless omission
        assert "omitir una condición, excepción o límite" in prompt  # harmful omission
        assert "Ejemplo 4 (condición eliminada)" in prompt
        assert "Ejemplo 6 (omisión de un elemento de una enumeración" in prompt
        assert "NO son afirmaciones" in prompt
        assert "source_ns" in prompt
        assert 'unidos con "..."' in prompt
        assert "en conjunto" in prompt
        assert "contexto de TODA la respuesta" in prompt
        citation = prompts.CITATION_SUPPORT_PROMPT
        assert "omitir un elemento de una" in citation
        assert "states_missing_info" in citation
        assert 'únelos con "..."' in citation

    def test_the_examples_use_a_fictitious_domain_not_the_evaluated_one(self) -> None:
        text = prompts.FAITHFULNESS_PROMPT + prompts.CITATION_SUPPORT_PROMPT
        # Domain terms of a former example case: the judge prompts must never contain them.
        for term in ("afiliado", "AFP", "cotizaci", "pensi", "Superintendencia", "días hábiles"):
            assert term not in text

    def test_no_prompt_shares_a_5_gram_with_the_case_calibration_or_agent_files(self) -> None:
        root = Path(__file__).resolve().parents[4] / "case"
        files = [
            f
            for pattern in ("*/calibration/**/*", "*/agents/**/*")
            for f in root.glob(pattern)
            if f.is_file()
            and f.suffix.lower() in {".jsonl", ".json", ".md", ".yaml", ".yml", ".txt", ".csv"}
        ]
        if not files:
            pytest.skip("no case calibration or agent files in this checkout")

        own = grams(
            "\n".join(v for k, v in vars(prompts).items() if k.isupper() and isinstance(v, str))
        )
        shared: dict[tuple[str, ...], str] = {}
        for f in files:
            for gram in own & grams(f.read_text(encoding="utf-8", errors="ignore")):
                shared.setdefault(gram, f.name)
        assert not shared, f"prompts overlap case files: {list(shared.items())[:5]}"

    @pytest.mark.parametrize(
        "prompt",
        [prompts.FAITHFULNESS_PROMPT, prompts.ANSWER_RELEVANCE_PROMPT],
    )
    def test_the_refusal_definition_is_v8_topic_based_with_two_examples(self, prompt: str) -> None:
        assert "NINGUNA información sobre el tema preguntado" in prompt
        assert "mismo objeto, otro aspecto" in prompt  # same object, other aspect: partial
        assert "De nada" in prompt  # courtesy is not a refusal
        assert "consejos de salud" in prompt  # off-domain content is not a refusal
        assert "No encuentro esa información" in prompt  # an actual refusal
        assert "is_refusal true" in prompt
        assert "is_refusal false" in prompt

    def test_citation_prompt_makes_partial_a_non_pass_and_fixes_the_list(self) -> None:
        prompt = prompts.CITATION_SUPPORT_PROMPT
        assert "partial" in prompt
        assert "not_supported" in prompt
        assert "exactamente una entrada por oración" in prompt

    def test_the_sources_come_first_in_the_user_message(self) -> None:
        text = faithfulness_messages("sys", make_record())[1].text
        assert text.index("<fuentes>") < text.index("<respuesta>") < text.index("<pregunta>")


class TestJudgeMeta:
    def test_it_carries_the_version_and_a_stable_hash(self) -> None:
        meta = judge_meta()
        assert meta["judge_version"] == prompts.JUDGE_VERSION
        assert len(meta["judge_prompts_sha256"]) == 64
        assert judge_meta() == meta

    def test_the_hash_follows_the_prompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = judge_meta()["judge_prompts_sha256"]
        monkeypatch.setattr(prompts, "FAITHFULNESS_PROMPT", prompts.FAITHFULNESS_PROMPT + " x")
        assert judge_meta()["judge_prompts_sha256"] != before

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("MIN_FRAGMENT_CHARS", 13),
            ("MAX_QUOTE_EDITS", 3),
            ("MAX_QUOTE_EDIT_RATIO", 0.2),
            ("MIN_SENTENCE_COVERAGE", 0.8),
            ("SHINGLE_WORDS", 4),
            ("MIN_SHINGLE_COVERAGE", 0.7),
            ("TOLERATED_TOKENS", frozenset({"el"})),
            ("WATCH_TOKENS", frozenset({"no"})),
        ],
    )
    def test_the_hash_follows_every_guard_parameter(
        self, monkeypatch: pytest.MonkeyPatch, name: str, value: object
    ) -> None:
        before = judge_meta()["judge_prompts_sha256"]
        monkeypatch.setattr(guard, name, value)
        assert judge_meta()["judge_prompts_sha256"] != before

    @pytest.mark.parametrize(
        ("module", "name", "value"),
        [
            (citations, "ABBREVIATIONS", frozenset({"art"})),
            (refusal, "REFUSAL_PATTERNS", (r"\bno lo se\b",)),
            (refusal, "MAX_REFUSAL_CHARS", 601),
        ],
    )
    def test_the_hash_follows_the_sentence_splitter_and_the_refusal_heuristic(
        self, monkeypatch: pytest.MonkeyPatch, module: Any, name: str, value: object
    ) -> None:
        before = judge_meta()["judge_prompts_sha256"]
        monkeypatch.setattr(module, name, value)
        assert judge_meta()["judge_prompts_sha256"] != before

    def test_the_hash_follows_the_rendered_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from langchain_core.messages import HumanMessage

        from agintis.agent_kit.judge import llm, render

        before = judge_meta()["judge_prompts_sha256"]
        original = render.faithfulness_messages

        def changed(prompt: str, record: AnswerRecord) -> list[Any]:
            return [*original(prompt, record), HumanMessage("extra")]

        monkeypatch.setattr(llm, "faithfulness_messages", changed)
        assert judge_meta()["judge_prompts_sha256"] != before
