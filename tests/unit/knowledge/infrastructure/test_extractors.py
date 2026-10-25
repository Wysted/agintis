"""Metadata extractors and value transforms (registered strategies, params validated)."""

from collections.abc import Mapping
from datetime import date
from typing import Any, ClassVar

import pytest

from agintis.modules.knowledge.api import (
    ChunkDraft,
    ExtractionContext,
    Extractor,
    InvalidProfileError,
    ParsedDocument,
    ParsedSection,
)
from agintis.modules.knowledge.application import RawValue
from agintis.modules.knowledge.infrastructure.extractors import (
    Classify,
    DateAfterHeading,
    DateFloor,
    LayeredRegex,
    PathSegment,
    PrefixMap,
    RegexAll,
    RegexLabeled,
    RejectAfter,
    Template,
    ValueMap,
)
from agintis.modules.knowledge.infrastructure.extractors.base import NoParams, scan_windows

TODAY = date(2026, 10, 15)


def _context(
    body: str = "",
    *,
    document: str = "",
    groups: dict[str, tuple[str, ...]] | None = None,
    document_metadata: dict[str, Any] | None = None,
    resolved: dict[str, Any] | None = None,
    selector: str | None = None,
    title: str = "Parsed title",
    memo: dict[str, object] | None = None,
) -> ExtractionContext:
    parsed = ParsedDocument(
        title, (ParsedSection(document or body),), document_metadata=document_metadata or {}
    )
    return ExtractionContext(
        draft=ChunkDraft(body, groups=groups or {}),
        parsed=parsed,
        document_text=document or body,
        resolved=resolved or {},
        today=TODAY,
        selector=selector,
        memo={} if memo is None else memo,
    )


def _run(extractor: Extractor, ctx: ExtractionContext, params: Mapping[str, Any]) -> RawValue:
    return extractor.extract(ctx, params)


class TestRegexAll:
    def test_values_of_a_group_in_order(self) -> None:
        ctx = _context("Circular N° 12 y Circular N° 7 y Circular N° 12")
        params = {"pattern": r"Circular N° (\d+)", "group": 1}
        assert _run(RegexAll(), ctx, params) == ("12", "7", "12")
        assert _run(RegexAll(), ctx, {**params, "unique": True}) == ("12", "7")

    def test_the_pattern_runs_over_the_multi_line_chunk_text(self) -> None:
        ctx = _context("Norma de\nCarácter General N° 31 y\nNorma de Carácter General N° 5")
        out = _run(
            RegexAll(), ctx, {"pattern": r"Norma de\s+Carácter\s+General\s+N°\s*(\d+)", "group": 1}
        )
        assert out == ("31", "5")

    def test_flags_and_normalization(self) -> None:
        ctx = _context("MODIFICA la  Circular   N° 9")
        params = {
            "pattern": r"modifica\s+la\s+circular\s+n°\s*\d+",
            "flags": "i",
            "normalize": "compact_spaces",
        }
        assert _run(RegexAll(), ctx, params) == ("MODIFICA la Circular N° 9",)

    def test_document_scope_reads_the_whole_text_once(self) -> None:
        memo: dict[str, object] = {}
        params = {"pattern": r"Circular (\d+)", "group": 1, "scope": "document"}
        first = _context("chunk text", document="Circular 4 and Circular 5", memo=memo)
        assert _run(RegexAll(), first, params) == ("4", "5")
        memo_size = len(memo)
        second = _context("other chunk", document="DIFFERENT TEXT", memo=memo)
        assert _run(RegexAll(), second, params) == ("4", "5")  # cached for the document
        assert len(memo) == memo_size

    def test_no_match_is_empty(self) -> None:
        assert _run(RegexAll(), _context("nothing"), {"pattern": r"\d+"}) == ()

    def test_params_are_validated(self) -> None:
        for params, message in (
            ({}, "pattern"),
            ({"pattern": "(a+)+"}, "nested quantifier"),
            ({"pattern": "a", "group": 1}, "does not exist"),
            ({"pattern": "a", "flags": "x"}, "flags"),
            ({"pattern": "a", "scope": "world"}, "scope"),
        ):
            with pytest.raises(InvalidProfileError, match=message):
                _run(RegexAll(), _context("a"), params)


class TestScanWindows:
    def test_short_text_is_one_window(self) -> None:
        assert list(scan_windows("a\n\nb", 100)) == ["a\n\nb"]
        assert list(scan_windows("", 100)) == []

    def test_windows_cover_the_text_exactly_and_prefer_paragraph_breaks(self) -> None:
        text = "\n\n".join(f"paragraph {n} " + "x" * 30 for n in range(20))
        windows = list(scan_windows(text, 120))
        assert "".join(windows) == text
        assert all(len(w) <= 120 for w in windows)
        assert all(w.endswith("\n\n") for w in windows[:-1])

    def test_a_text_without_breaks_is_cut_at_the_limit(self) -> None:
        windows = list(scan_windows("x" * 250, 100))
        assert [len(w) for w in windows] == [100, 100, 50]

    def test_a_reference_wrapped_over_a_line_still_matches_inside_a_window(self) -> None:
        body = "filler " * 40 + "Norma de\nCarácter General N° 31"
        out = _run(
            RegexAll(),
            _context(body),
            {"pattern": r"Norma de\s+Carácter\s+General\s+N°\s*(\d+)", "group": 1},
        )
        assert out == ("31",)

    def test_regex_all_scans_documents_bigger_than_the_regex_cap(self) -> None:
        document = ("filler line\n" * 20_000) + "Circular 42\n" + ("more\n" * 1_000)
        assert len(document) > 100_000
        params = {"pattern": r"Circular (\d+)", "group": 1, "scope": "document"}
        assert _run(RegexAll(), _context("chunk", document=document), params) == ("42",)


class TestClassify:
    RULES: ClassVar[dict[str, Any]] = {
        "rules": [
            {"field": "section", "pattern": "Lobby", "label": "lobby"},
            {"field": "section", "pattern": "Normativa del", "label": "normativa"},
        ],
        "default": "institucional",
    }

    def test_first_matching_rule_wins(self) -> None:
        ctx = _context(document_metadata={"section": ("A > Normativa del Sistema", "Lobby")})
        assert _run(Classify(), ctx, self.RULES) == "lobby"

    def test_default_when_nothing_matches_or_the_field_is_missing(self) -> None:
        assert (
            _run(Classify(), _context(document_metadata={"section": ("X",)}), self.RULES)
            == "institucional"
        )
        assert _run(Classify(), _context(), self.RULES) == "institucional"
        assert _run(Classify(), _context(), {"rules": self.RULES["rules"]}) is None

    def test_head_of_the_document_text(self) -> None:
        params = {
            "rules": [{"head_chars": 10, "pattern": "MODIFICA", "flags": "i", "label": "amendment"}]
        }
        assert _run(Classify(), _context(document="modifica el libro"), params) == "amendment"
        assert _run(Classify(), _context(document="x" * 20 + "modifica"), params) is None

    def test_the_title_falls_back_to_the_parsed_one(self) -> None:
        params = {
            "rules": [{"field": "title", "pattern": r"\?\s*$", "label": "yes"}],
            "default": "no",
        }
        assert _run(Classify(), _context(title="Why?"), params) == "yes"
        assert _run(Classify(), _context(document_metadata={"title": "Why not."}), params) == "no"

    def test_a_rule_needs_exactly_one_source(self) -> None:
        with pytest.raises(InvalidProfileError, match="exactly one"):
            _run(Classify(), _context(), {"rules": [{"pattern": "a", "label": "x"}]})


class TestTemplate:
    PARAMS: ClassVar[dict[str, Any]] = {
        "template": "{system}/L{libro}/T{titulo}{letra?/{letra}}{capitulo?/C{capitulo}}",
        "required": ["system", "libro", "titulo"],
    }

    def test_renders_resolved_fields_and_optional_parts(self) -> None:
        resolved = {"system": "SP", "libro": "I", "titulo": "III", "letra": "A", "capitulo": "V"}
        assert _run(Template(), _context(resolved=resolved), self.PARAMS) == "SP/LI/TIII/A/CV"
        del resolved["letra"]
        assert _run(Template(), _context(resolved=resolved), self.PARAMS) == "SP/LI/TIII/CV"

    def test_missing_required_values_give_nothing(self) -> None:
        assert (
            _run(Template(), _context(resolved={"system": "SP", "libro": "I"}), self.PARAMS) is None
        )

    def test_groups_are_available_and_resolved_fields_win(self) -> None:
        ctx = _context(
            groups={"titulo": ("IV",), "libro": ("X",)}, resolved={"libro": "I", "system": "SP"}
        )
        assert _run(Template(), ctx, self.PARAMS) == "SP/LI/TIV"

    def test_lists_and_numbers_are_rendered_as_text(self) -> None:
        ctx = _context(resolved={"n": 5, "tags": ("a", "b")})
        assert _run(Template(), ctx, {"template": "{n}:{tags}"}) == "5:a | b"


class TestLayeredRegex:
    PARAMS: ClassVar[dict[str, Any]] = {
        "layers": [{"field": "title"}, {"group": "parte_head"}, {"group": "enmienda_head"}],
        "strip_quoted": {"open": "“", "close": "”"},
        "max_chars": 260,
        "require": ["libro"],
        "keys": {
            "libro": {"pattern": r"\bLibro\s+([IVX]+)\b", "flags": "i", "upper": True},
            "titulo": {"pattern": r"\bT[ií]tulo\s+([IVXL]+)\b", "flags": "i", "upper": True},
            "letra": {"pattern": r"\bLetra\s+([A-Z](?:\.\d)?)(?=[\s.,])"},
            "capitulo": {"pattern": r"\bCap[ií]tulo\s+([IVXL]+)\b", "flags": "i", "upper": True},
            "numero": {"pattern": r"\bn[uú]mero\s+(\d{1,2}(?:\.\d{1,2})*)(?!\.\d)"},
        },
        "reset": [{"when": "letra", "unless": "capitulo", "clear": ["capitulo", "numero"]}],
    }

    def _get(self, key: str, **kwargs: Any) -> object:
        return _run(LayeredRegex(), _context(selector=key, **kwargs), self.PARAMS)

    def test_later_layers_override_earlier_ones(self) -> None:
        kwargs = {
            "document_metadata": {"title": "MODIFICA EL TÍTULO I DEL LIBRO III DEL MANUAL"},
            "groups": {
                "enmienda_head": (
                    "Reemplácese, en el número 6.1 del Capítulo II de la Letra D.1 siguiente",
                )
            },
        }
        assert self._get("libro", **kwargs) == "III"
        assert self._get("titulo", **kwargs) == "I"
        assert self._get("letra", **kwargs) == "D.1"
        assert self._get("capitulo", **kwargs) == "II"
        assert self._get("numero", **kwargs) == "6.1"

    def test_a_letra_without_chapter_in_the_same_layer_resets_the_inherited_chapter(self) -> None:
        kwargs = {
            "document_metadata": {"title": "Libro III Capítulo V"},
            "groups": {"parte_head": ("Letra B de este título",)},
        }
        assert self._get("capitulo", **kwargs) is None
        assert self._get("letra", **kwargs) == "B"

    def test_quoted_text_is_ignored_and_an_open_quote_hides_the_rest(self) -> None:
        kwargs = {
            "document_metadata": {"title": "Libro II"},
            "groups": {"enmienda_head": ("Agrégase “Capítulo IX” inserted, Capítulo III",)},
        }
        assert self._get("capitulo", **kwargs) == "III"
        unclosed = {
            "document_metadata": {"title": "Libro II"},
            "groups": {"enmienda_head": ("Capítulo I “Capítulo IX",)},
        }
        assert self._get("capitulo", **unclosed) == "I"

    def test_required_keys_missing_give_nothing_at_all(self) -> None:
        assert self._get("titulo", document_metadata={"title": "Título I only"}) is None

    def test_the_title_falls_back_to_the_parsed_one(self) -> None:
        assert self._get("libro", title="Libro IV") == "IV"

    def test_text_is_cut_to_max_chars(self) -> None:
        params = {**self.PARAMS, "max_chars": 10, "require": []}
        ctx = _context(
            selector="titulo", document_metadata={"title": "Libro I " + "x" * 20 + " Título II"}
        )
        assert _run(LayeredRegex(), ctx, params) is None

    def test_selectors_are_the_declared_keys(self) -> None:
        assert LayeredRegex().selectors_for(self.PARAMS) == {
            "libro",
            "titulo",
            "letra",
            "capitulo",
            "numero",
        }

    def test_params_are_validated(self) -> None:
        for overrides, message in (
            ({"require": ["nope"]}, "undeclared keys"),
            ({"keys": {"a": {"pattern": "no group"}}}, "capture group"),
            ({"layers": [{"field": "t", "group": "g"}]}, "exactly one"),
            ({"layers": []}, "at least 1 item"),
        ):
            with pytest.raises(InvalidProfileError, match=message):
                _run(LayeredRegex(), _context(selector="a"), {**self.PARAMS, **overrides})


LEGAL_REFS: dict[str, Any] = {
    "rules": [
        {"pattern": r"\bD\.\s?L\.\s*N[°º]\s*([\d.]+)", "label": "DL", "strip_chars": "."},
        {"pattern": r"\bDecreto\s+Ley\s+N[°º]\s*([\d.]+)", "label": "DL", "strip_chars": "."},
        {"pattern": r"\bD\.\s?F\.\s?L\.\s*N[°º]\s*([\d.]+)", "label": "DFL", "strip_chars": "."},
        {"pattern": r"\bD\.\s?S\.(?:\s+conjunto)?\s+N[°º]\s*(\d+)", "label": "DS"},
        {
            "pattern": r"(?<!Decreto )\b[Ll]ey\s+N[°º]\s*([\d.]+)",
            "label": "Ley",
            "strip_chars": ".",
        },
    ]
}
MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}
VIGENCIA: dict[str, Any] = {
    "heading_pattern": r"^(?:[IVX]{1,4}\.\s*)?VIGENCIA\s*:?\s*$",
    "window_chars": 700,
    "phrase_pattern": r"(?:regir[áa]|comenzar[áa]n? a regir)[^.]{0,160}",
    "phrase_flags": "i",
    "date": {
        "pattern": r"(\d{1,2})[°º]?\s+de\s+(" + "|".join(MONTHS) + r")\s+(?:de|del)\s+(\d{4})",
        "flags": "i",
        "day_group": 1,
        "month_group": 2,
        "year_group": 3,
        "months": MONTHS,
    },
}


class TestRegexLabeled:
    def test_references_are_labeled_normalized_and_unique(self) -> None:
        body = (
            "El D.L. N° 3.500, de 1980, la ley N° 21.735, el D.S. conjunto N° 47 y el "
            "Decreto Ley N° 3.500; DFL N° 1; D.F.L. N° 29. Ley N° 21.735."
        )
        assert _run(RegexLabeled(), _context(body), LEGAL_REFS) == (
            "DL 3500",
            "DFL 29",
            "DS 47",
            "Ley 21735",
        )

    def test_separator_and_duplicates_are_configurable(self) -> None:
        params = {
            "rules": [{"pattern": r"Law (\d+)", "label": "L"}],
            "separator": ":",
            "unique": False,
        }
        assert _run(RegexLabeled(), _context("Law 1, Law 1"), params) == ("L:1", "L:1")

    def test_a_group_that_does_not_exist_is_rejected(self) -> None:
        with pytest.raises(InvalidProfileError, match="does not exist"):
            _run(RegexLabeled(), _context("x"), {"rules": [{"pattern": "a", "label": "x"}]})

    def test_text_beyond_one_window_is_still_scanned(self) -> None:
        body = "x" * 150_000 + "\nLey N° 21.735"
        assert _run(RegexLabeled(), _context(body), LEGAL_REFS) == ("Ley 21735",)


class TestDateAfterHeading:
    def _date(self, text: str, **overrides: Any) -> RawValue:
        return _run(DateAfterHeading(), _context(document=text), {**VIGENCIA, **overrides})

    def test_effective_date_under_the_heading(self) -> None:
        text = (
            "1. algo\nII. VIGENCIA\n\n"
            "La presente norma rige a partir del 1 de septiembre de 2026.\nfin"
        )
        assert self._date(text) == "2026-09-01"

    def test_dates_use_the_configured_month_names(self) -> None:
        text = "VIGENCIA\na contar del 1° de Octubre de 2026 y el 5 de setiembre del 2011"
        assert self._date(text) == "2026-10-01"
        assert self._date("VIGENCIA\n5 de Setiembre del 2011") == "2011-09-05"
        english = {"pattern": r"(\d{1,2}) (\w+) (\d{4})", "day_group": 1, "month_group": 2}
        english |= {"year_group": 3, "months": {"march": 3}}
        assert self._date("VIGENCIA\n4 March 2020", date=english) == "2020-03-04"

    def test_found_from_a_phrase_without_heading(self) -> None:
        text = "La circular comenzará a regir el 3 de enero de 2020. Otra fecha 4 de mayo de 2021."
        assert self._date(text) == "2020-01-03"
        assert self._date(text, phrase_pattern=None) is None

    def test_a_heading_must_be_a_line_of_its_own(self) -> None:
        assert self._date("La VIGENCIA de este texto es incierta, 4 de mayo de 2021.") is None

    def test_an_impossible_date_gives_no_value(self) -> None:
        assert self._date("VIGENCIA\n31 de febrero de 2026") is None
        assert self._date("VIGENCIA\n0 de enero de 2026") is None
        assert self._date("VIGENCIA\n1 de enero de 0000") is None

    def test_a_month_missing_from_the_map_gives_no_value(self) -> None:
        text = "VIGENCIA\n1 de enero de 2026"
        months = {"febrero": 2}
        assert self._date(text, date={**VIGENCIA["date"], "months": months}) is None

    def test_the_date_is_computed_once_per_document(self) -> None:
        memo: dict[str, object] = {}
        text = "VIGENCIA\n1 de marzo de 2026"
        first = _context(document=text, memo=memo)
        assert _run(DateAfterHeading(), first, VIGENCIA) == "2026-03-01"
        other = _context(document="changed", memo=memo)
        assert _run(DateAfterHeading(), other, VIGENCIA) == "2026-03-01"

    def test_params_are_validated(self) -> None:
        bad_month = {**VIGENCIA["date"], "months": {"enero": 13}}
        bad_group = {**VIGENCIA["date"], "year_group": 9}
        for overrides, message in (
            ({"date": bad_month}, "from 1 to 12"),
            ({"date": bad_group}, "does not exist"),
            ({"window_chars": 0}, "greater than or equal to 1"),
            ({"heading_pattern": "(a+)+"}, "nested quantifier"),
        ):
            with pytest.raises(InvalidProfileError, match=message):
                self._date("x", **overrides)

    def test_no_params_model_still_rejects_unknown_keys(self) -> None:
        with pytest.raises(InvalidProfileError, match="Extra inputs"):
            _run(Classify(), _context("x"), {"rules": [], "nope": 1})
        assert NoParams().model_dump() == {}


class TestClassifyIsMemoized:
    def test_the_label_is_computed_once_per_document(self) -> None:
        memo: dict[str, object] = {}
        params = {"rules": [{"head_chars": 20, "pattern": "MODIFICA", "label": "a"}]}
        first = _context(document="MODIFICA algo", memo=memo)
        assert _run(Classify(), first, params) == "a"
        assert _run(Classify(), _context(document="otro", memo=memo), params) == "a"


class TestTransforms:
    def test_value_map_replaces_known_values_only(self) -> None:
        params = {"map": {"billing": "BL", "refunds": "RF"}}
        assert ValueMap().apply("billing", params, TODAY) == "BL"
        assert ValueMap().apply("otro", params, TODAY) == "otro"
        assert ValueMap().apply(("billing", "x"), params, TODAY) == ("BL", "x")
        assert ValueMap().apply(5, params, TODAY) == 5

    def test_date_floor_completes_partial_dates(self) -> None:
        floor = DateFloor()
        assert floor.apply("2021", {}, TODAY) == "2021-01-01"
        assert floor.apply("2021-03", {}, TODAY) == "2021-03-01"
        assert floor.apply("2021-03-09", {}, TODAY) == "2021-03-09"
        assert floor.apply("March", {}, TODAY) == "March"

    def test_reject_after_drops_future_dates(self) -> None:
        reject = RejectAfter()
        assert reject.apply("2026-10-15", {}, TODAY) == "2026-10-15"
        assert reject.apply("2026-10-16", {}, TODAY) is None
        assert reject.apply("2030-01-01", {"limit": "ingest_date"}, TODAY) is None
        with pytest.raises(InvalidProfileError):
            reject.apply("2020-01-01", {"limit": "tomorrow"}, TODAY)

    def test_path_segment_keeps_the_last_segment_of_each_path(self) -> None:
        paths = ("Atención > Empleadores", "Educación > Seguro > Empleadores", "Raíz")
        assert PathSegment().apply(paths, {}, TODAY) == ("Empleadores", "Raíz")
        assert PathSegment().apply("A > B", {"take": "first"}, TODAY) == "A"
        assert PathSegment().apply("A / B", {"separator": "/"}, TODAY) == "B"

    def test_path_segment_never_raises_on_mixed_values(self) -> None:
        assert PathSegment().apply(("b > 2", 7, "a"), {}, TODAY) == (7, "2", "a")
        assert PathSegment().apply((None, True, "x"), {}, TODAY) == (True, "x")

    def test_date_floor_drops_impossible_dates(self) -> None:
        floor = DateFloor()
        assert floor.apply("2021-13", {}, TODAY) is None
        assert floor.apply("2021-02-30", {}, TODAY) is None
        assert floor.apply(("2021-02-30", "2021"), {}, TODAY) == ("2021-01-01",)

    def test_prefix_map_matches_stems_ignoring_accents_and_case(self) -> None:
        params = {"map": {"reempl": "replace", "agrég": "add", "modif": "modify", "re": "other"}}
        lemma = PrefixMap()
        assert lemma.apply("Reemplácese", params, TODAY) == "replace"
        assert lemma.apply("Agrégase", params, TODAY) == "add"
        assert lemma.apply("MODIFÍQUESE", params, TODAY) == "modify"
        assert (
            lemma.apply("Rectifíquese", params, TODAY) == "other"
        )  # the longest stem wins elsewhere
        assert lemma.apply("Elimínase", params, TODAY) is None
        assert lemma.apply(("Reemplácese", "zzz"), params, TODAY) == ("replace",)

    def test_params_are_validated(self) -> None:
        with pytest.raises(InvalidProfileError, match="map"):
            ValueMap().apply("a", {}, TODAY)
        with pytest.raises(InvalidProfileError, match="Extra inputs"):
            DateFloor().apply("2021", {"x": 1}, TODAY)
