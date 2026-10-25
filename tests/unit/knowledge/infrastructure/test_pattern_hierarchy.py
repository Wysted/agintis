"""Pattern-hierarchy chunker: structure, context levels, quote masking, merging, oversize."""

import itertools
import re
import time
from typing import Any, ClassVar

import pytest

from agintis.modules.knowledge.api import (
    ChunkDraft,
    InvalidProfileError,
    ParsedDocument,
    ParsedSection,
    RagProfile,
)
from agintis.modules.knowledge.infrastructure.chunkers import (
    PatternHierarchyChunker,
    PatternHierarchyParams,
)
from agintis.modules.knowledge.infrastructure.chunkers.pattern_hierarchy import (
    QuoteSpanParams,
    _inside,
    _quote_spans,
)
from agintis.modules.knowledge.infrastructure.chunkers.splitting import SPECIAL_TOKENS_RESERVE
from agintis.modules.knowledge.profiles import validate_params
from tests.unit.knowledge.infrastructure.test_chunker import WordCounter

CHAPTER = {
    "name": "capitulo",
    "pattern": r"^Cap[ií]tulo (?P<capitulo>[IVX]+)\.\s+(?P<capitulo_nombre>\S.*)$",
    "heading_template": "Capítulo {capitulo}. {capitulo_nombre}",
}
ITEM = {
    "name": "numero",
    "pattern": r"^(?P<numero>\d{1,2})\.\s+\S",
    "heading_template": "{numero}.",
}
LETTER = {
    "name": "literal",
    "pattern": r"^(?P<literal>[a-z])\)\s+\S",
    "heading_template": "{literal})",
}


def _params(levels: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "levels": levels if levels is not None else [CHAPTER, ITEM],
        "split_level": "numero",
        "heading_prefix_template": "{path}",
        "preamble_heading": "Encabezado",
    }
    params.update(overrides)
    return params


def _profile(budget: int = 60, **params: Any) -> RagProfile:
    return RagProfile(
        "p",
        3,
        budget + SPECIAL_TOKENS_RESERVE,
        "simple",
        "pattern_hierarchy",
        chunker_params=_params(**params),
    )


def _doc(*pages: str, metadata: dict[str, Any] | None = None) -> ParsedDocument:
    sections = tuple(ParsedSection(text=t, page=i) for i, t in enumerate(pages, start=1))
    return ParsedDocument(title="Doc", sections=sections, document_metadata=metadata or {})


async def _chunk(doc: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
    return await PatternHierarchyChunker(WordCounter()).chunk(doc, profile)


def _page(draft: ChunkDraft, key: str) -> int:
    value = draft.metadata[key]
    assert isinstance(value, int)
    return value


def _group(draft: ChunkDraft, name: str) -> tuple[str, ...]:
    return draft.groups.get(name, ())


LAW = """\
Preface of the document.
Capítulo I. General rules
1. First item of the first chapter.
2. Second item of the first chapter.
Capítulo II. Other rules
1. First item of the second chapter."""


class TestStructure:
    async def test_units_are_cut_at_the_split_level_with_the_open_path_as_heading(self) -> None:
        drafts = await _chunk(_doc(LAW), _profile())
        assert [d.text.split("\n")[0] for d in drafts] == [
            "Encabezado",
            "Capítulo I. General rules",
            "Capítulo I. General rules > 1.",
            "Capítulo I. General rules > 2.",
            "Capítulo II. Other rules",
            "Capítulo II. Other rules > 1.",
        ]
        item = drafts[2]
        assert item.body == "1. First item of the first chapter."
        assert item.text == "Capítulo I. General rules > 1.\n1. First item of the first chapter."
        assert item.metadata["heading"] == "Capítulo I. General rules > 1."

    async def test_named_groups_of_every_open_level_are_recorded(self) -> None:
        drafts = await _chunk(_doc(LAW), _profile())
        item = drafts[3]
        assert _group(item, "capitulo") == ("I",)
        assert _group(item, "capitulo_nombre") == ("General rules",)
        assert _group(item, "numero") == ("2",)
        assert _group(drafts[0], "capitulo") == ()  # the preamble has no open level

    async def test_preamble_can_be_dropped(self) -> None:
        drafts = await _chunk(_doc(LAW), _profile(preamble="drop"))
        assert "Encabezado" not in [d.text.split("\n")[0] for d in drafts]
        assert len(drafts) == 5

    async def test_preamble_heading_is_configurable(self) -> None:
        drafts = await _chunk(_doc(LAW), _profile(preamble_heading="Header"))
        assert drafts[0].text.startswith("Header\n")

    async def test_the_heading_prefix_is_optional(self) -> None:
        drafts = await _chunk(_doc(LAW), _profile(include_heading=False))
        assert drafts[2].text == "1. First item of the first chapter."
        assert drafts[2].metadata["heading"] == "Capítulo I. General rules > 1."

    async def test_prefix_template_reads_document_metadata_and_groups(self) -> None:
        profile = _profile(heading_prefix_template="REF {number} ({date}) {title:12} > {path}")
        doc = _doc(
            LAW,
            metadata={"number": 372, "date": "2026-08-18", "title": "A very long document title"},
        )
        drafts = await _chunk(doc, profile)
        assert drafts[2].text.startswith("REF 372 (2026-08-18) A very long… > Capítulo I.")

    async def test_a_document_without_headings_is_one_preamble_chunk(self) -> None:
        drafts = await _chunk(_doc("just text\nwith two lines"), _profile())
        assert [d.text for d in drafts] == ["Encabezado\njust text\nwith two lines"]

    async def test_a_deeper_split_level_cuts_inside_chapters_and_items(self) -> None:
        text = "Capítulo I. A\n1. Item\na) one thing\nb) another thing\n2. Next"
        drafts = await _chunk(
            _doc(text), _profile(levels=[CHAPTER, ITEM, LETTER], split_level="literal")
        )
        assert [d.text.split("\n")[0] for d in drafts] == [
            "Capítulo I. A",
            "Capítulo I. A > 1.",
            "Capítulo I. A > 1. > a)",
            "Capítulo I. A > 1. > b)",
            "Capítulo I. A > 2.",
        ]

    async def test_depth_makes_sibling_levels(self) -> None:
        annex = {
            **CHAPTER,
            "name": "anexo",
            "pattern": r"^Anexo (?P<anexo>\d+)$",
            "depth": 0,
            "heading_template": "Anexo {anexo}",
        }
        chapter = {**CHAPTER, "depth": 0}
        item = {**ITEM, "depth": 1}
        text = "Capítulo I. A\n1. x\nAnexo 1\n1. y"
        drafts = await _chunk(
            _doc(text), _profile(levels=[chapter, annex, item], split_level="numero")
        )
        # The annex closes the chapter: its item does not inherit it.
        assert drafts[-1].text.split("\n")[0] == "Anexo 1 > 1."
        assert _group(drafts[-1], "capitulo") == ()

    async def test_patterns_are_matched_one_line_at_a_time(self) -> None:
        """`[^.;]*` and `\\s*$` must not swallow the next line (re.M over the text would)."""
        annex = {
            "name": "anexo",
            "pattern": r"^Anexo\s*(?P<anexo>\d+)?\s*(?P<nombre>[^.;]*[^.;\s])?\s*$",
            "heading_template": "Anexo {anexo} {nombre}",
        }
        text = "Anexo 1\n\n\n1. Heading of the next item that must survive"
        drafts = await _chunk(_doc(text), _profile(levels=[annex, ITEM], split_level="numero"))
        assert _group(drafts[0], "nombre") == ()
        assert any(d.body == "1. Heading of the next item that must survive" for d in drafts)

    async def test_a_lookahead_guard_rejects_cross_references_at_line_start(self) -> None:
        guarded = {
            **CHAPTER,
            "pattern": (
                r"^(?!.*\b(?:de la presente|podrá)\b)Cap[ií]tulo (?P<capitulo>[IVX]+)\."
                r"\s+(?P<capitulo_nombre>\S.*)$"
            ),
        }
        text = (
            "Capítulo I. Real\n1. item that cites\n"
            "Capítulo V. Recaudación, de la presente Letra\n2. more"
        )
        drafts = await _chunk(_doc(text), _profile(levels=[guarded, ITEM]))
        assert [_group(d, "capitulo") for d in drafts] == [("I",), ("I",), ("I",)]


class TestPages:
    async def test_units_may_cross_pages_and_record_both(self) -> None:
        drafts = await _chunk(
            _doc("Capítulo I. A\n1. starts here and", "continues on page two\n2. next"),
            _profile(),
        )
        crossing = next(d for d in drafts if d.body and d.body.startswith("1."))
        assert crossing.metadata["page"] == 1
        assert crossing.metadata["page_end"] == 2
        # Pages are joined by a blank line; the unit text is continuous across them.
        assert crossing.body == "1. starts here and\n\ncontinues on page two"

    async def test_pageless_documents_have_no_page_metadata(self) -> None:
        doc = ParsedDocument(title="t", sections=(ParsedSection("Capítulo I. A\n1. x"),))
        drafts = await _chunk(doc, _profile())
        assert all("page" not in d.metadata for d in drafts)


BREADCRUMB = {
    "name": "seccion",
    "depth": 0,
    "pattern": r"^Libro I, T[ií]tulo (?P<titulo>[IVX]+)(?:, Letra (?P<letra>[A-Z]))?$",
    "heading_template": "Título {titulo}{letra?, Letra {letra}}",
    "role": "context",
    "reset_on_change": ["titulo", "letra"],
    "keep_heading_line": False,
}


class TestContextLevels:
    LEVELS: ClassVar[list[dict[str, Any]]] = [
        BREADCRUMB,
        {**CHAPTER, "depth": 1},
        {**ITEM, "depth": 2},
    ]

    async def test_a_repeated_breadcrumb_does_not_close_the_open_chapter(self) -> None:
        text = (
            "Libro I, Título I, Letra A\nCapítulo I. A\n1. first item\n"
            "Libro I, Título I, Letra A\n2. second item on a new page\n"
            "Libro I, Título I, Letra B\n3. other letter"
        )
        drafts = await _chunk(_doc(text), _profile(levels=self.LEVELS))
        by_item = {_group(d, "numero"): d for d in drafts if _group(d, "numero")}
        assert _group(by_item[("2",)], "capitulo") == ("I",)  # still inside the chapter
        assert _group(by_item[("2",)], "letra") == ("A",)
        assert _group(by_item[("3",)], "capitulo") == ()  # the Letra changed: chapter closed
        assert _group(by_item[("3",)], "letra") == ("B",)

    async def test_the_breadcrumb_line_is_cut_out_of_the_text(self) -> None:
        text = (
            "Libro I, Título I, Letra A\nCapítulo I. A\n1. first item\n"
            "Libro I, Título I, Letra A\n2. second"
        )
        drafts = await _chunk(_doc(text), _profile(levels=self.LEVELS))
        assert all("Libro I" not in (d.body or "") for d in drafts)
        assert drafts[-1].body == "2. second"
        assert drafts[-1].text.startswith("Título I, Letra A > Capítulo I. A > 2.\n")


class TestQuotedText:
    QUOTE: ClassVar[dict[str, Any]] = {"open": "“", "close": "”", "nesting": True}

    async def test_markers_inside_quotes_are_not_headings(self) -> None:
        text = (
            "Capítulo I. A\n1. Replace the text by:\n“Capítulo X. Inserted\n1. inserted item\n"
            "2. another inserted item”\n2. real second item"
        )
        drafts = await _chunk(_doc(text), _profile(ignore_inside=[self.QUOTE]))
        assert [_group(d, "numero") for d in drafts] == [(), ("1",), ("2",)]
        assert "inserted item" in (drafts[1].body or "")

    async def test_quotes_nest(self) -> None:
        text = (
            "Capítulo I. A\n1. Insert:\n“text with “inner” quote\n"
            "2. still inside the outer quote”\n3. outside"
        )
        drafts = await _chunk(_doc(text), _profile(ignore_inside=[self.QUOTE]))
        assert [_group(d, "numero") for d in drafts] == [(), ("1",), ("3",)]

    async def test_without_masking_the_quoted_markers_split_the_unit(self) -> None:
        text = "Capítulo I. A\n1. Insert:\n“\n2. quoted\n”\n3. outside"
        drafts = await _chunk(_doc(text), _profile())
        assert [_group(d, "numero") for d in drafts] == [(), ("1",), ("2",), ("3",)]

    async def test_an_unclosed_quote_masks_nothing(self) -> None:
        text = "Capítulo I. A\n1. open “quote never closes\n2. still a heading"
        drafts = await _chunk(_doc(text), _profile(ignore_inside=[self.QUOTE]))
        assert [_group(d, "numero") for d in drafts] == [(), ("1",), ("2",)]

    async def test_a_span_longer_than_the_limit_is_not_a_quote(self) -> None:
        text = "Capítulo I. A\n1. open “" + "x" * 50 + "\n2. inside?”\n3. after"
        quote = {**self.QUOTE, "max_span_chars": 20}
        drafts = await _chunk(_doc(text), _profile(ignore_inside=[quote]))
        assert [_group(d, "numero") for d in drafts] == [(), ("1",), ("2",), ("3",)]

    async def test_flat_spans_pair_open_and_close(self) -> None:
        text = 'Capítulo I. A\n1. say "\n2. hidden"\n3. visible'
        flat = {"open": '"', "close": '"'}
        drafts = await _chunk(_doc(text), _profile(ignore_inside=[flat]))
        assert [_group(d, "numero") for d in drafts] == [(), ("1",), ("3",)]


class TestQuoteSpans:
    """Regression scenarios of the review probes: overlap, stray quotes, linearity."""

    BOTH: ClassVar[list[QuoteSpanParams]] = [
        QuoteSpanParams(open="«", close="»"),
        QuoteSpanParams(open='"', close='"'),
    ]

    def test_spans_of_several_specs_are_merged_into_disjoint_intervals(self) -> None:
        text = '«aaa "x\n1. hidden heading\n" bbb»\n2. real'
        spans = _quote_spans(text, self.BOTH)
        assert spans == [(0, text.index("»") + 1)]  # one interval, not two overlapping ones
        hidden = text.index("1. hidden")
        assert _inside(hidden, spans, [start for start, _ in spans])

    def test_interleaved_spans_still_hide_everything_between(self) -> None:
        text = '«aaa "x" y\n1. hidden heading\nbbb»\n2. real'
        spans = _quote_spans(text, self.BOTH)
        assert len(spans) == 1
        assert _inside(text.index("1. hidden"), spans, [s for s, _ in spans])
        assert not _inside(text.index("2. real"), spans, [s for s, _ in spans])

    def test_spans_are_disjoint_and_sorted(self) -> None:
        text = '«a» "b" «c "d" e» f'
        spans = _quote_spans(text, self.BOTH)
        assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))

    def test_a_stray_open_quote_does_not_disable_masking_for_the_rest(self) -> None:
        text = "“stray open\n1. a\n“\n5. quoted heading inside\n”\n2. b"
        spans = _quote_spans(text, [QuoteSpanParams(open="“", close="”", nesting=True)])
        quoted = text.index("5. quoted")
        assert any(start < quoted < end for start, end in spans)
        assert not any(start < text.index("2. b") < end for start, end in spans)

    def test_an_opener_without_a_closer_in_range_is_abandoned(self) -> None:
        quote = QuoteSpanParams(open="“", close="”", nesting=True, max_span_chars=40)
        text = "“never closed " + "x" * 100 + " “ok” tail"
        spans = _quote_spans(text, [quote])
        assert spans == [(text.index("“ok”"), text.index("”") + 1)] or len(spans) == 1
        assert text[spans[0][0] : spans[0][1]] == "“ok”"

    def test_many_stray_openers_are_linear(self) -> None:
        started = time.perf_counter()
        for opens in (2_000, 20_000):
            big = ("“x " * opens) + "\n" + ("y " * 200_000)
            _quote_spans(big, [QuoteSpanParams(open="“", close="”")])
            _quote_spans(big, [QuoteSpanParams(open="“", close="”", nesting=True)])
        assert time.perf_counter() - started < 5

    def test_flat_openers_before_one_distant_closer_scan_once(self) -> None:
        text = "“" * 5_000 + "x" * 200_000 + "”"
        quote = QuoteSpanParams(open="“", close="”", max_span_chars=100)
        started = time.perf_counter()
        spans = _quote_spans(text, [quote])
        assert time.perf_counter() - started < 2
        assert spans == []  # every opener is too far from the closer

    async def test_masking_survives_a_stray_quote_in_the_chunker(self) -> None:
        text = "Capítulo I. A\n1. stray “ quote\n2. next\n“\n3. inserted heading\n”\n4. after"
        quote = {"open": "“", "close": "”", "nesting": True}
        drafts = await _chunk(_doc(text), _profile(ignore_inside=[quote]))
        numbers = [_group(d, "numero") for d in drafts]
        assert ("3",) not in numbers
        assert ("4",) in numbers


class TestFoldAndMerge:
    async def test_a_short_chapter_intro_is_prepended_to_its_first_item(self) -> None:
        drafts = await _chunk(_doc(LAW), _profile(preamble="drop", fold_intro_below_tokens=10))
        assert drafts[0].body == "Capítulo I. General rules\n1. First item of the first chapter."
        assert drafts[0].text.startswith("Capítulo I. General rules > 1.\n")
        assert _group(drafts[0], "numero") == ("1",)
        assert len(drafts) == 3

    async def test_small_siblings_merge_forward_keeping_every_group(self) -> None:
        text = "Capítulo I. A\n1. aa bb\n2. cc dd\n3. " + " ".join(["w"] * 12)
        drafts = await _chunk(_doc(text), _profile(preamble="drop", merge_below_tokens=5))
        assert drafts[0].body == "Capítulo I. A"  # a different level: never merged
        merged = drafts[1]
        assert merged.body == "1. aa bb\n2. cc dd"
        assert _group(merged, "numero") == ("1", "2")
        assert merged.text.startswith("Capítulo I. A > 1.\n")  # the first unit's heading
        assert _group(drafts[2], "numero") == ("3",)

    async def test_units_with_different_parents_never_merge(self) -> None:
        text = "Capítulo I. A\n1. aa\nCapítulo II. B\n1. bb"
        drafts = await _chunk(
            _doc(text), _profile(merge_below_tokens=50, fold_intro_below_tokens=0)
        )
        assert [_group(d, "numero") for d in drafts] == [(), ("1",), (), ("1",)]

    async def test_a_merge_that_would_exceed_the_budget_is_skipped(self) -> None:
        big = " ".join(["w"] * 25)
        text = f"Capítulo I. A\n1. {big}\n2. {big}"
        drafts = await _chunk(
            _doc(text), _profile(budget=40, preamble="drop", merge_below_tokens=100)
        )
        assert [_group(d, "numero") for d in drafts] == [(), ("1",), ("2",)]


class TestOversize:
    def _text(self, words: int) -> str:
        return "Capítulo I. A\n1. " + " ".join(f"w{i}" for i in range(words))

    async def test_every_chunk_fits_the_budget_prefix_included(self) -> None:
        profile = _profile(budget=30, preamble="drop")
        drafts = await _chunk(_doc(self._text(200)), profile)
        assert len(drafts) > 3
        assert all(WordCounter().count(d.text) <= 30 for d in drafts)
        assert all(_group(d, "numero") == ("1",) for d in drafts[1:])

    async def test_it_cuts_at_the_configured_deeper_levels_first(self) -> None:
        a, b = " ".join(["a"] * 12), " ".join(["b"] * 12)
        text = f"Capítulo I. A\n1. intro\na) {a}\nb) {b}"
        profile = _profile(
            levels=[CHAPTER, ITEM, LETTER],
            budget=24,
            preamble="drop",
            oversize={"levels": ["literal"]},
        )
        drafts = await _chunk(_doc(text), profile)
        bodies = [d.body for d in drafts[1:]]
        assert bodies == [f"1. intro\na) {a}", f"b) {b}"]

    async def test_extra_split_patterns_cut_even_inside_quotes(self) -> None:
        a, b = " ".join(["a"] * 12), " ".join(["b"] * 12)
        text = f"Capítulo I. A\n1. replace by:\n“1. {a}\n2. {b}”"
        profile = _profile(
            budget=26,
            preamble="drop",
            ignore_inside=[{"open": "“", "close": "”"}],
            oversize={"extra_split_patterns": [r"^\d{1,2}\.\s+\S"]},
        )
        drafts = await _chunk(_doc(text), profile)
        assert [d.body for d in drafts[1:]] == [f"1. replace by:\n“1. {a}", f"2. {b}”"]

    async def test_a_word_longer_than_the_budget_is_still_cut(self) -> None:
        # WordCounter cannot split inside a word: the unit stays in one piece per whitespace run.
        drafts = await _chunk(_doc(self._text(60)), _profile(budget=10, preamble="drop"))
        assert all(WordCounter().count(d.text) <= 10 for d in drafts)


class TestOversizePieces:
    """Every piece of an oversize unit carries its own pages."""

    def _paged(self) -> ParsedDocument:
        sections = tuple(
            ParsedSection(
                text=("1. start\n" if page == 1 else "")
                + " ".join(f"p{page}w{i}" for i in range(40)),
                page=page,
            )
            for page in range(1, 6)
        )
        return ParsedDocument("T", sections)

    async def test_each_piece_records_the_pages_it_was_read_from(self) -> None:
        drafts = await _chunk(self._paged(), _profile(budget=50, preamble="drop"))
        assert len(drafts) > 4
        for draft in drafts:
            words = [w for w in (draft.body or "").split() if re.fullmatch(r"p\dw\d+", w)]
            if words:
                assert draft.metadata["page"] == int(words[0][1])
                assert draft.metadata["page_end"] == int(words[-1][1])
        assert len({d.metadata["page"] for d in drafts}) > 1  # not the whole unit's hull

    async def test_oversize_pieces_after_excised_lines_keep_exact_pages(self) -> None:
        context = {
            "name": "bc",
            "depth": 0,
            "pattern": r"^BC (?P<bc>\w+)$",
            "role": "context",
            "keep_heading_line": False,
        }
        text1 = "BC a\n1. " + " ".join(f"a{i}" for i in range(30))
        text2 = "BC a\n\n\n\n" + " ".join(f"b{i}" for i in range(30))
        doc = ParsedDocument("T", (ParsedSection(text1, page=1), ParsedSection(text2, page=2)))
        profile = _profile(levels=[context, ITEM], budget=20, preamble="drop")
        drafts = await _chunk(doc, profile)
        pages = [(_page(d, "page"), _page(d, "page_end")) for d in drafts]
        assert pages[0][0] == 1
        assert pages[-1] == (2, 2)
        assert all(a <= b for a, b in pages)

    async def test_pieces_of_a_pageless_document_have_no_pages(self) -> None:
        doc = ParsedDocument("T", (ParsedSection("1. " + " ".join(["w"] * 80)),))
        drafts = await _chunk(doc, _profile(budget=20))
        assert len(drafts) > 3
        assert all("page" not in d.metadata for d in drafts)


class TestMergeBudget:
    ITEMS = (
        "1. " + " ".join(f"a{i}" for i in range(9)) + "\n2. " + " ".join(f"b{i}" for i in range(8))
    )

    async def test_a_merge_leaves_room_for_the_first_units_heading_prefix(self) -> None:
        profile = _profile(
            budget=20, merge_below_tokens=15, heading_prefix_template="{title} {path}"
        )
        doc = ParsedDocument("A B C D E", (ParsedSection(self.ITEMS),))
        drafts = await PatternHierarchyChunker(WordCounter()).chunk(doc, profile)
        assert all(WordCounter().count(d.text) <= 20 for d in drafts)
        # The merged body would not fit next to a 6-token prefix: no merge, no re-split with
        # the groups of both items on every piece.
        assert [_group(d, "numero") for d in drafts] == [("1",), ("2",)]

    async def test_without_a_prefix_the_same_units_do_merge(self) -> None:
        profile = _profile(budget=20, merge_below_tokens=15, include_heading=False)
        drafts = await _chunk(_doc(self.ITEMS), profile)
        assert [_group(d, "numero") for d in drafts] == [("1", "2")]


class TestPreambleHeading:
    async def test_there_is_no_default_preamble_heading(self) -> None:
        params = _params()
        del params["preamble_heading"]
        profile = RagProfile("p", 3, 68, "simple", "pattern_hierarchy", chunker_params=params)
        drafts = await _chunk(_doc(LAW), profile)
        assert drafts[0].text == "Preface of the document."
        assert "heading" not in drafts[0].metadata

    def test_the_default_is_none(self) -> None:
        assert PatternHierarchyParams.model_fields["preamble_heading"].default is None


class TestTemplatesAtBoot:
    def test_a_heading_template_may_only_read_its_own_groups(self) -> None:
        level = {**CHAPTER, "heading_template": "{capitulo} {ghost}"}
        with pytest.raises(InvalidProfileError, match="ghost"):
            validate_params(PatternHierarchyParams, _params([level, ITEM]), where="p")

    def test_malformed_templates_are_rejected(self) -> None:
        level = {**CHAPTER, "heading_template": "{capitulo:0}"}
        with pytest.raises(InvalidProfileError, match="malformed placeholder"):
            validate_params(PatternHierarchyParams, _params([level, ITEM]), where="p")
        with pytest.raises(InvalidProfileError, match="malformed placeholder"):
            validate_params(
                PatternHierarchyParams, _params(heading_prefix_template="{path"), where="p"
            )

    def test_the_prefix_template_may_read_document_metadata(self) -> None:
        params = _params(heading_prefix_template="{number} {title:10} {path}")
        assert validate_params(PatternHierarchyParams, params, where="p")


class TestParentLevel:
    async def test_parent_heading_is_recorded_as_plain_metadata(self) -> None:
        drafts = await _chunk(_doc(LAW), _profile(parent_level="capitulo"))
        assert drafts[2].metadata["parent_heading"] == "Capítulo I. General rules"
        assert "parent_heading" not in drafts[0].metadata  # the preamble has no parent


class TestParams:
    def _validate(self, **overrides: Any) -> PatternHierarchyParams:
        return validate_params(
            PatternHierarchyParams, _params(**overrides), where="pattern_hierarchy"
        )

    def test_defaults(self) -> None:
        params = self._validate()
        assert params.depths() == {"capitulo": 0, "numero": 1}
        assert params.group_names() == {"capitulo", "capitulo_nombre", "numero"}
        assert params.preamble == "keep"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"split_level": "nope"}, "split_level"),
            ({"levels": []}, "at least 1 item"),
            ({"levels": [CHAPTER, CHAPTER]}, "unique"),
            ({"oversize": {"levels": ["capitulo"]}}, "deeper than the split level"),
            ({"oversize": {"levels": ["nope"]}}, "unknown level"),
            ({"parent_level": "numero"}, "shallower"),
            ({"parent_level": "nope"}, "parent_level"),
            ({"levels": [{**CHAPTER, "pattern": r"^(a+)+$"}, ITEM]}, "nested quantifier"),
            ({"levels": [{**CHAPTER, "pattern": "("}, ITEM]}, "invalid regular expression"),
            ({"levels": [{**CHAPTER, "name": "Bad Name"}, ITEM]}, "level names"),
            ({"levels": [{**CHAPTER, "reset_on_change": ["capitulo"]}, ITEM]}, "role: context"),
            (
                {"levels": [{**CHAPTER, "role": "context", "reset_on_change": ["x"]}, ITEM]},
                "unknown groups",
            ),
            ({"ignore_inside": [{"open": "ab", "close": "b"}]}, "at most 1 character"),
            ({"ignore_inside": [{"open": "a", "close": "a", "nesting": True}]}, "nesting"),
            ({"unknown": 1}, "Extra inputs"),
            ({"merge_below_tokens": -1}, "greater than or equal to 0"),
        ],
    )
    def test_invalid_params_are_reported(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(InvalidProfileError, match=message):
            self._validate(**overrides)

    async def test_the_budget_must_exceed_the_special_tokens_reserve(self) -> None:
        profile = RagProfile(
            "p",
            1,
            SPECIAL_TOKENS_RESERVE,
            "simple",
            "pattern_hierarchy",
            chunker_params=_params(),
        )
        with pytest.raises(ValueError, match="special tokens reserve"):
            await _chunk(_doc(LAW), profile)
