import time
from typing import Any

import pytest

from agintis.modules.knowledge.application import (
    InvalidProfileError,
    ParsedDocument,
    ParsedSection,
)
from agintis.modules.knowledge.infrastructure.cleaners import (
    CLEANER_CLASSES,
    DedupeHeadingEcho,
    DropPagesBefore,
    DropTocPages,
    JoinContinuationLines,
    NormalizeQuotes,
    NormalizeWhitespace,
    StripLines,
    StripRepeatedPageLines,
    StripRunningHeaders,
)
from agintis.modules.knowledge.infrastructure.registry import (
    build_cleaners,
    cleaner_param_schemas,
)
from agintis.modules.knowledge.profiles import MAX_REGEX_INPUT_CHARS, ParamsModel


def _pages(*texts: str) -> ParsedDocument:
    sections = tuple(ParsedSection(text=t, page=i + 1) for i, t in enumerate(texts))
    return ParsedDocument(title="t", sections=sections)


def _plain(*texts: str) -> ParsedDocument:
    return ParsedDocument(title="t", sections=tuple(ParsedSection(text=t) for t in texts))


def _texts(parsed: ParsedDocument) -> list[str]:
    return [s.text for s in parsed.sections]


async def _run(cleaner: Any, doc: ParsedDocument, **params: Any) -> ParsedDocument:
    result: ParsedDocument = await cleaner().clean(doc, params)
    return result


class TestRegistry:
    def test_every_cleaner_is_registered_once_with_a_schema(self) -> None:
        cleaners, schemas = build_cleaners(), cleaner_param_schemas()
        keys = [c.key for c in CLEANER_CLASSES]
        assert len(set(keys)) == len(keys) == len(cleaners) == len(schemas)
        assert set(keys) == {
            "strip_running_headers",
            "strip_repeated_page_lines",
            "strip_lines",
            "drop_toc_pages",
            "drop_pages_before",
            "dedupe_heading_echo",
            "normalize_quotes",
            "normalize_whitespace",
            "join_continuation_lines",
            "html_main_content",
            "html_collect",
        }
        assert all(issubclass(s, ParamsModel) for s in schemas.values())


class TestParamsValidation:
    async def test_unknown_param_is_rejected_naming_the_cleaner(self) -> None:
        with pytest.raises(InvalidProfileError, match=r"strip_lines.*bogus"):
            await _run(StripLines, _plain("x"), patterns=["a"], bogus=1)

    async def test_invalid_regex_is_rejected(self) -> None:
        with pytest.raises(InvalidProfileError, match="invalid regular expression"):
            await _run(StripLines, _plain("x"), patterns=["("])

    async def test_missing_required_param_is_rejected(self) -> None:
        with pytest.raises(InvalidProfileError, match="patterns"):
            await _run(StripLines, _plain("x"))

    async def test_wrong_type_is_rejected(self) -> None:
        with pytest.raises(InvalidProfileError, match="max_blank_lines"):
            await _run(NormalizeWhitespace, _plain("x"), max_blank_lines="2")

    async def test_dedupe_pattern_needs_the_key_group(self) -> None:
        with pytest.raises(InvalidProfileError, match="key"):
            await _run(DedupeHeadingEcho, _plain("x"), pattern=r"^\d+$")


class TestStripLines:
    async def test_removes_matching_lines_and_drops_emptied_sections(self) -> None:
        doc = _pages("keep\n12\nVerifique documento en https://x.cl/a", "7")
        out = await _run(
            StripLines, doc, patterns=[r"^\s*\d{1,3}\s*$", r"^Verifique documento en \S+$"]
        )
        assert _texts(out) == ["keep"]
        assert [s.page for s in out.sections] == [1]

    async def test_matches_line_by_line_not_across_lines(self) -> None:
        out = await _run(StripLines, _plain("a\n\nb"), patterns=[r"^$"])
        assert _texts(out) == ["a\nb"]


class TestNormalizeWhitespace:
    async def test_collapses_strips_and_caps_blank_lines(self) -> None:
        doc = _plain("  a   b\u00a0\u00a0c  \n\n\n\n d\t\te ")
        out = await _run(NormalizeWhitespace, doc, max_blank_lines=1)
        assert _texts(out) == ["a b c\n\nd e"]

    async def test_options_can_be_turned_off(self) -> None:
        out = await _run(
            NormalizeWhitespace,
            _plain("a  b\n\n\nc"),
            collapse_inner_spaces=False,
            max_blank_lines=2,
        )
        assert _texts(out) == ["a  b\n\n\nc"]


class TestNormalizeQuotes:
    async def test_maps_characters(self) -> None:
        out = await _run(NormalizeQuotes, _plain("«hola»"), map={"«": "“", "»": "”"})
        assert _texts(out) == ["“hola”"]

    async def test_alternates_ascii_quotes(self) -> None:
        doc = _plain('di "a" y "b"', '"c"')
        out = await _run(NormalizeQuotes, doc, ascii_double="alternate")
        assert _texts(out) == ["di “a” y “b”", "“c”"]

    async def test_pageless_sections_share_one_alternation_state(self) -> None:
        out = await _run(
            NormalizeQuotes, _plain('abre "una', 'sigue" y cierra'), ascii_double="alternate"
        )
        assert _texts(out) == ["abre “una", "sigue” y cierra"]

    async def test_each_pdf_page_starts_with_an_opening_quote(self) -> None:
        out = await _run(
            NormalizeQuotes, _pages('rota "una', '"dos" bien'), ascii_double="alternate"
        )
        assert _texts(out) == ["rota “una", "“dos” bien"]

    async def test_sections_of_the_same_page_share_the_state(self) -> None:
        doc = ParsedDocument(
            title="t",
            sections=(ParsedSection(text='a "b', page=3), ParsedSection(text='c" d', page=3)),
        )
        out = await _run(NormalizeQuotes, doc, ascii_double="alternate")
        assert _texts(out) == ["a “b", "c” d"]

    async def test_map_is_a_simultaneous_replacement_independent_of_entry_order(self) -> None:
        forward = {"a": "b", "b": "c", "<<": "a"}
        backward = dict(reversed(forward.items()))
        one = await _run(NormalizeQuotes, _plain("ab<<"), map=forward)
        two = await _run(NormalizeQuotes, _plain("ab<<"), map=backward)
        assert _texts(one) == _texts(two) == ["bca"]  # sequential replacement would give "ccc"

    async def test_longest_key_wins_over_its_prefix(self) -> None:
        out = await _run(NormalizeQuotes, _plain("<<x<"), map={"<": "(", "<<": "«"})
        assert _texts(out) == ["«x("]

    async def test_map_values_are_not_interpreted_as_regex_replacements(self) -> None:
        out = await _run(NormalizeQuotes, _plain("a.b"), map={".": r"\1"})
        assert _texts(out) == [r"a\1b"]

    async def test_keeps_ascii_quotes_by_default(self) -> None:
        assert _texts(await _run(NormalizeQuotes, _plain('"a"'))) == ['"a"']


class TestJoinContinuationLines:
    CONTINUATION = r"^(?!\d+\.\s)\S"

    async def test_joins_wrapped_lines_and_drops_blanks(self) -> None:
        doc = _plain("1. tema largo\ncontinua aqui\n\n2. otro\nOf. Ord. 5")
        out = await _run(JoinContinuationLines, doc, continuation=self.CONTINUATION)
        assert _texts(out) == ["1. tema largo continua aqui\n2. otro Of. Ord. 5"]

    async def test_record_start_always_begins_a_line(self) -> None:
        doc = _plain("1. a\nB. nuevo\nsigue")
        out = await _run(
            JoinContinuationLines, doc, continuation=r"^\S", record_start=r"^[A-Z]\.\s"
        )
        assert _texts(out) == ["1. a\nB. nuevo sigue"]


class TestDedupeHeadingEcho:
    PATTERN = r"^(?P<key>\d+\.\s+\S.*)$"

    async def test_drops_the_short_marker_and_keeps_the_full_title(self) -> None:
        doc = _plain("text\n1. Intro\n1. Intro\nbody")
        out = await _run(DedupeHeadingEcho, doc, pattern=self.PATTERN)
        assert _texts(out) == ["text\n1. Intro\nbody"]

    async def test_drops_lines_between_a_wrapped_marker_and_the_echo(self) -> None:
        doc = _plain(
            "Cap\u00edtulo. I\nTitulo largo\nque se parte\nCap\u00edtulo I. Titulo largo\nbody"
        )
        out = await _run(
            DedupeHeadingEcho,
            doc,
            pattern=r"^(?P<key>Cap[i\u00ed]tulo\.?\s*[IVXL]+\b)",
            key_normalize="strip_dots_spaces",
        )
        assert _texts(out) == ["Cap\u00edtulo I. Titulo largo\nbody"]

    async def test_echo_outside_the_window_is_kept(self) -> None:
        doc = _plain("1. A\nx\ny\nz\n1. A")
        out = await _run(DedupeHeadingEcho, doc, pattern=self.PATTERN, window_lines=2)
        assert _texts(out) == ["1. A\nx\ny\nz\n1. A"]


class TestStripRunningHeaders:
    PATTERNS = (r"^Handbook of Rules .* Libro I \d+$",)

    async def test_removes_matching_lines_in_the_zone_only(self) -> None:
        header = "Handbook of Rules - Libro I 12"
        doc = _pages(f"{header}\nbody", f"body\n{header}")
        out = await _run(
            StripRunningHeaders, doc, patterns=list(self.PATTERNS), zone="top", max_lines=1
        )
        assert _texts(out) == ["body", f"body\n{header}"]

    async def test_bottom_and_any_zones(self) -> None:
        header = "Handbook of Rules - Libro I 12"
        doc = _pages(f"{header}\nbody\n{header}")
        bottom = await _run(
            StripRunningHeaders, doc, patterns=list(self.PATTERNS), zone="bottom", max_lines=1
        )
        anywhere = await _run(
            StripRunningHeaders, doc, patterns=list(self.PATTERNS), zone="any", max_lines=1
        )
        assert _texts(bottom) == [f"{header}\nbody"]
        assert _texts(anywhere) == ["body"]

    async def test_documents_without_pages_are_untouched(self) -> None:
        doc = _plain("Handbook of Rules - Libro I 12\nbody")
        assert await _run(StripRunningHeaders, doc, patterns=list(self.PATTERNS)) == doc


def _page_body(index: int, lines: int = 8) -> str:
    # Letters only: with digit normalization "p1 line 2" and "p3 line 4" would look alike.
    return "\n".join(f"page{chr(97 + index)} line{chr(97 + n)}" for n in range(lines))


class TestStripRepeatedPageLines:
    @staticmethod
    def _pages_with_furniture(count: int) -> ParsedDocument:
        return _pages(
            *(f"LETTERHEAD S.A.\n{_page_body(i)}\nPage {i} of {count}" for i in range(count))
        )

    async def test_removes_furniture_repeated_across_pages(self) -> None:
        out = await _run(
            StripRepeatedPageLines, self._pages_with_furniture(4), normalize_digits=True
        )
        assert _texts(out) == [_page_body(i) for i in range(4)]

    async def test_digits_are_not_normalized_by_default(self) -> None:
        out = await _run(StripRepeatedPageLines, self._pages_with_furniture(4))
        assert _texts(out) == [f"{_page_body(i)}\nPage {i} of 4" for i in range(4)]

    async def test_short_documents_are_left_alone_by_default(self) -> None:
        doc = _pages("Title line\nbody one", "Title line\nbody two")
        assert await _run(StripRepeatedPageLines, doc) == doc

    async def test_two_pages_need_the_line_on_both_pages(self) -> None:
        doc = _pages("Shared header\nunique one\nreal content A", "Shared header\nunique two")
        out = await _run(StripRepeatedPageLines, doc, min_pages=2)
        assert _texts(out) == ["unique one\nreal content A", "unique two"]

    async def test_a_line_found_on_a_single_page_is_never_furniture(self) -> None:
        doc = _pages("only here\nbody one", "body two")
        out = await _run(StripRepeatedPageLines, doc, min_pages=2, min_page_ratio=0.1)
        assert _texts(out) == ["only here\nbody one", "body two"]

    async def test_bare_page_numbers_are_not_treated_as_furniture(self) -> None:
        doc = _pages("a\n1", "b\n2", "c\n3")
        assert await _run(StripRepeatedPageLines, doc, normalize_digits=True) == doc

    async def test_content_in_the_middle_of_the_page_is_never_removed(self) -> None:
        middle = "\n".join(f"repeated middle {n}" for n in range(4))
        doc = _pages(*(f"H\nt{i}a\nt{i}b\n{middle}\nb{i}a\nb{i}b\nF" for i in range(3)))
        out = await _run(StripRepeatedPageLines, doc)
        assert all(middle in text for text in _texts(out))
        assert all(not text.startswith("H") and not text.endswith("F") for text in _texts(out))


class TestDropTocPages:
    LEADER = r"\.{4,}\s*\d+$"

    async def test_drops_only_the_leading_toc_run(self) -> None:
        doc = _pages(
            "Cap 1 ........ 3\nCap 2 ........ 9",
            "Cap 3 ........ 12\nnota",
            "Real text\nmore text",
            "Cap 9 ........ 99",
        )
        out = await _run(DropTocPages, doc, leader_pattern=self.LEADER, min_leader_ratio=0.5)
        assert [s.page for s in out.sections] == [3, 4]

    async def test_never_drops_every_page(self) -> None:
        doc = _pages("a ........ 1", "b ........ 2")
        assert await _run(DropTocPages, doc, leader_pattern=self.LEADER) == doc

    async def test_respects_max_scan_pages(self) -> None:
        doc = _pages("a ........ 1", "b ........ 2", "c ........ 3", "text")
        out = await _run(DropTocPages, doc, leader_pattern=self.LEADER, max_scan_pages=2)
        assert [s.page for s in out.sections] == [3, 4]


class TestDropPagesBefore:
    PATTERN = r"^\s*INDICE\s*$"

    async def test_drops_cover_and_keeps_the_matching_page(self) -> None:
        doc = _pages("cover", "prologue", "INDICE\nrecords", "more")
        out = await _run(DropPagesBefore, doc, pattern=self.PATTERN)
        assert [s.page for s in out.sections] == [3, 4]

    async def test_can_drop_the_matching_page_too(self) -> None:
        doc = _pages("cover", "INDICE\nx", "more")
        out = await _run(DropPagesBefore, doc, pattern=self.PATTERN, keep_matching_page=False)
        assert [s.page for s in out.sections] == [3]

    async def test_pattern_not_found_keeps_everything(self) -> None:
        doc = _pages("a", "b")
        assert await _run(DropPagesBefore, doc, pattern=self.PATTERN) == doc

    async def test_matches_per_line_not_inside_a_line(self) -> None:
        doc = _pages("see the INDICE of this", "INDICE")
        out = await _run(DropPagesBefore, doc, pattern=self.PATTERN)
        assert [s.page for s in out.sections] == [2]


class TestRegexSafety:
    """W3: every profile regex is validated and bounded."""

    @pytest.mark.parametrize(
        "pattern",
        [r"(a+)+$", r"(\w+\s?)*", r"(.*a)*", r"(?:\s*\w+)+", r"((a+)b)+", r"(a|b+)*", r"(x+){20}"],
    )
    async def test_nested_unbounded_quantifiers_are_rejected_with_a_clear_error(
        self, pattern: str
    ) -> None:
        with pytest.raises(InvalidProfileError, match="nested quantifier"):
            await _run(StripLines, _plain("x"), patterns=[pattern])

    @pytest.mark.parametrize(
        "pattern",
        [
            r"^\d+$",
            r"(\d+\.){2}\d+",
            r"(?>a+)+",
            r"(a++)+",
            r"(\d{1,3}\.)+",
            r"^(?:[A-Z][a-z]+ ){1,3}",
        ],
    )
    async def test_safe_shapes_and_atomic_alternatives_are_accepted(self, pattern: str) -> None:
        await _run(StripLines, _plain("x"), patterns=[pattern])

    @pytest.mark.parametrize(
        ("cleaner", "params"),
        [
            (StripLines, {"patterns": [r"(a+)+"]}),
            (StripRunningHeaders, {"patterns": [r"(a+)+"]}),
            (DropTocPages, {"leader_pattern": r"(a+)+"}),
            (DropPagesBefore, {"pattern": r"(a+)+"}),
            (JoinContinuationLines, {"continuation": r"(a+)+"}),
            (JoinContinuationLines, {"continuation": "x", "record_start": r"(a+)+"}),
            (DedupeHeadingEcho, {"pattern": r"(?P<key>a+)+"}),
        ],
    )
    async def test_every_regex_parameter_goes_through_the_check(
        self, cleaner: Any, params: dict[str, Any]
    ) -> None:
        with pytest.raises(InvalidProfileError, match="nested quantifier"):
            await _run(cleaner, _plain("x"), **params)

    async def test_an_evil_looking_line_cannot_hang_a_safe_pattern(self) -> None:
        started = time.perf_counter()
        out = await _run(StripLines, _plain("a" * 100_000 + "!"), patterns=[r"^a+$"])
        assert time.perf_counter() - started < 0.5
        assert len(_texts(out)) == 1  # over the cap: no match, the line stays

    async def test_a_line_over_the_cap_never_matches_and_is_reported_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        long_line = "x" * (MAX_REGEX_INPUT_CHARS + 1)
        text = "\n".join([long_line, "keep", long_line])
        out = await _run(StripLines, _plain(text), patterns=["x"])
        assert _texts(out) == [text]
        assert caplog.text.count("skipped an input") == 1

    async def test_a_line_at_the_cap_still_matches(self) -> None:
        line = "x" * MAX_REGEX_INPUT_CHARS
        assert _texts(await _run(StripLines, _plain(f"{line}\nkeep"), patterns=["^x+$"])) == [
            "keep"
        ]

    async def test_the_cap_applies_to_page_cleaners_too(self) -> None:
        long_line = "Titulo " + "x" * MAX_REGEX_INPUT_CHARS
        doc = _pages(f"{long_line}\nbody", "other")
        out = await _run(StripRunningHeaders, doc, patterns=["^Titulo"])
        assert _texts(out)[0].startswith("Titulo")
