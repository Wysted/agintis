import time
from typing import Any, ClassVar

import pytest

from agintis.modules.knowledge.application import (
    InvalidProfileError,
    ParsedDocument,
    SourceFile,
    UnparseableDocumentError,
    UnsupportedDocumentTypeError,
)
from agintis.modules.knowledge.infrastructure.cleaners import HtmlCollect, HtmlMainContent
from agintis.modules.knowledge.infrastructure.html import (
    BlockOptions,
    collect_blocks,
    normalize_space,
    parse_html_tree,
    parse_selector,
    select_all,
    text_content,
)
from agintis.modules.knowledge.infrastructure.parsers import SimpleParser
from agintis.modules.knowledge.profiles import MAX_REGEX_INPUT_CHARS

PAGE = """<!doctype html>
<html><head><title>Site - Pregunta</title><style>.x{}</style></head>
<body>
<nav><a href="/">Home</a></nav>
<div id="titulo"><h1>Que es el plan?</h1></div>
<div id="cuerpo">
  <p>Primer parrafo con <a href="/x">enlace</a>
  y salto<br>de linea.</p>
  <p><strong>Requisitos</strong></p>
  <ul><li>Tener 65 anos<li>Estar afiliado</ul>
  <h3>Detalle</h3>
  <p>Texto del detalle.</p>
  <div class="botones-texto">Aumentar letra</div>
  <span style="display: none">tag1, tag2 ,tag3</span>
  <table><tr><td>a</td><td>b</td></tr></table>
</div>
<div id="otras"><a href="/w3-article-101.html">uno</a><a href="/w3-article-202.html">dos</a></div>
<footer>Pie de pagina</footer>
<script>var x = "<p>no</p>";</script>
</body></html>
"""


async def _parse(html: str = PAGE, uri: str = "https://x.cl/a.html") -> ParsedDocument:
    source = SourceFile(uri=uri, mime_type="text/html; charset=utf-8", content=html.encode())
    return await SimpleParser().parse(source)


async def _clean(cleaner: Any, parsed: ParsedDocument, **params: Any) -> ParsedDocument:
    result: ParsedDocument = await cleaner().clean(parsed, params)
    return result


class TestSelectors:
    def _select(self, selector: str) -> list[str]:
        root = parse_html_tree(PAGE)
        return [text_content(n).strip() for n in select_all(root, [parse_selector(selector)])]

    def test_id_class_tag_and_descendant(self) -> None:
        assert self._select("#titulo h1") == ["Que es el plan?"]
        assert self._select(".botones-texto") == ["Aumentar letra"]
        assert self._select("#cuerpo li")[0].startswith("Tener 65")

    def test_attribute_operators_and_whitespace_insensitive_style(self) -> None:
        assert self._select('[style*="display:none"]') == ["tag1, tag2 ,tag3"]
        assert self._select("#otras a[href$='101.html']") == ["uno"]
        assert self._select("a[href^='/w3']") == ["uno", "dos"]
        assert self._select("[href]")[0] == "Home"

    def test_child_combinator_and_lists(self) -> None:
        assert self._select("#titulo > h1") == ["Que es el plan?"]
        assert self._select("#cuerpo > h1") == []
        assert self._select("h1, footer") == ["Que es el plan?", "Pie de pagina"]

    @pytest.mark.parametrize("selector", ["", "div >", "a:hover", "div[", "##x"])
    def test_unsupported_syntax_is_a_value_error(self, selector: str) -> None:
        with pytest.raises(ValueError, match=r"."):
            parse_selector(selector)

    def test_hostile_nesting_does_not_blow_the_stack(self) -> None:
        for depth in (150, 5000):
            root = parse_html_tree("<div>" * depth + "deep" + "</div>" * depth)
            text_content(root)  # iterative: no RecursionError whatever the depth libxml2 kept
            collect_blocks([root], BlockOptions())
        shallow = parse_html_tree("<div>" * 150 + "deep" + "</div>" * 150)
        assert "deep" in text_content(shallow)


class TestHtmlRoute:
    async def test_generic_extraction_drops_scripts_nav_and_footer(self) -> None:
        parsed = await _parse()
        text = "\n".join(s.text for s in parsed.sections)
        assert parsed.title == "Site - Pregunta"
        assert "Home" not in text
        assert "Pie de pagina" not in text
        assert "var x" not in text
        assert "Primer parrafo con enlace y salto de linea." in text
        assert parsed.source_html is not None
        assert parsed.sections[0].headings == ("Que es el plan?",)

    async def test_extension_routes_when_the_mime_is_generic(self) -> None:
        source = SourceFile("https://x.cl/p.htm", "application/octet-stream", b"<p>hola</p>")
        parsed = await SimpleParser().parse(source)
        assert [s.text for s in parsed.sections] == ["hola"]

    async def test_unclosed_list_items_do_not_nest(self) -> None:
        parsed = await _parse("<ul><li>uno<li>dos<li>tres</ul>")
        assert [s.text for s in parsed.sections] == ["- uno\n\n- dos\n\n- tres"]

    async def test_text_with_markdown_lookalikes_stays_text(self) -> None:
        parsed = await _parse("<p># not a heading</p><p>- not a list</p>")
        assert [s.headings for s in parsed.sections] == [()]
        assert parsed.sections[0].text == "# not a heading\n\n- not a list"

    async def test_nul_in_the_markup_is_sanitized(self) -> None:
        parsed = await _parse("<p>a\x00b</p>")
        assert parsed.sections[0].text == "ab"
        assert parsed.source_html == "<p>ab</p>"


class TestHtmlMainContent:
    PARAMS: ClassVar[dict[str, Any]] = {
        "title_selector": "#titulo h1",
        "include_selectors": ["#cuerpo"],
        "exclude_selectors": ['[style*="display:none"]', ".botones-texto"],
        "bold_paragraph_as_heading": True,
        "heading_selectors": ["h2", "h3", "h4"],
    }

    async def test_keeps_only_the_selected_content_as_sections(self) -> None:
        out = await _clean(HtmlMainContent, await _parse(), **self.PARAMS)
        assert out.title == "Que es el plan?"
        by_heading = {s.headings: s.text for s in out.sections}
        assert by_heading[("Que es el plan?",)] == ("Primer parrafo con enlace y salto de linea.")
        assert (
            by_heading[("Que es el plan?", "Requisitos")] == "- Tener 65 anos\n\n- Estar afiliado"
        )
        assert (
            by_heading[("Que es el plan?", "Requisitos", "Detalle")]
            == "Texto del detalle.\n\na | b"
        )
        flat = " ".join(by_heading.values())
        for gone in ("Aumentar", "tag1", "Pie de pagina", "Home"):
            assert gone not in flat

    async def test_nested_include_matches_are_not_duplicated(self) -> None:
        params = {**self.PARAMS, "include_selectors": ["#cuerpo", "#cuerpo p"]}
        out = await _clean(HtmlMainContent, await _parse(), **params)
        assert sum("Texto del detalle." in s.text for s in out.sections) == 1

    async def test_defaults_include_the_whole_body(self) -> None:
        out = await _clean(HtmlMainContent, await _parse())
        assert any("Aumentar letra" in s.text for s in out.sections)

    async def test_non_html_source_is_rejected(self) -> None:
        plain = ParsedDocument(title="t", sections=())
        with pytest.raises(UnsupportedDocumentTypeError, match="html_main_content"):
            await _clean(HtmlMainContent, plain)

    @pytest.mark.parametrize(
        "params",
        [
            {"include_selectors": ["div["]},
            {"include_selectors": []},
            {"heading_selectors": ["strong"]},
            {"title_selector": "a:hover"},
            {"output": "html"},
            {"unknown": 1},
        ],
    )
    async def test_invalid_params_are_rejected(self, params: dict[str, Any]) -> None:
        with pytest.raises(InvalidProfileError, match="html_main_content"):
            await _clean(HtmlMainContent, await _parse(), **params)


class TestHtmlCollect:
    FIELDS: ClassVar[dict[str, Any]] = {
        "keywords": {"selector": '#cuerpo [style*="display:none"]', "split": ","},
        "related_aids": {
            "selector": "#otras a",
            "attr": "href",
            "regex": r"w3-article-(\d+)\.html",
        },
        "headline": {"selector": "#titulo h1"},
    }

    async def test_collects_named_values_from_the_whole_dom(self) -> None:
        main = await _clean(HtmlMainContent, await _parse(), include_selectors=["#cuerpo"])
        out = await _clean(HtmlCollect, main, fields=self.FIELDS)
        assert dict(out.collected) == {
            "keywords": ("tag1", "tag2", "tag3"),
            "related_aids": (101, 202),
            "headline": ("Que es el plan?",),
        }
        assert out.sections == main.sections

    async def test_collected_values_are_read_only(self) -> None:
        out = await _clean(HtmlCollect, await _parse(), fields=self.FIELDS)
        with pytest.raises(TypeError):
            out.collected["x"] = ()  # type: ignore[index]

    @pytest.mark.parametrize(
        "fields",
        [
            {"k": {"selector": "p", "regex": r"\d+"}},  # no capture group
            {"k": {"selector": "p", "regex": r"(\d)", "split": ","}},
            {"Bad-Name": {"selector": "p"}},
            {"k": {"selector": "p["}},
            {},
        ],
    )
    async def test_invalid_fields_are_rejected(self, fields: dict[str, Any]) -> None:
        with pytest.raises(InvalidProfileError, match="html_collect"):
            await _clean(HtmlCollect, await _parse(), fields=fields)


def _texts(parsed: ParsedDocument) -> list[str]:
    return [s.text for s in parsed.sections]


def _structure(parsed: ParsedDocument) -> list[tuple[tuple[str, ...], str]]:
    return [(s.headings, s.text) for s in parsed.sections]


class TestRealWorldMarkup:
    """Implied end tags and omitted wrappers (what the hand-written DOM got wrong)."""

    @pytest.mark.parametrize(
        "markup",
        [
            "<html><head><title>Titulo</title><meta charset=utf-8><body><p>Contenido</p>",
            "<html><head><title>Titulo</title><p>Contenido</p>",  # </head> and <body> omitted
            "<title>Titulo</title><p>Contenido</p>",  # no <html>, <head> or <body> at all
            "<head><title>Titulo</title></head><p>Contenido",
        ],
    )
    async def test_omitted_head_and_body_keep_the_content_and_not_the_title(
        self, markup: str
    ) -> None:
        parsed = await _parse(markup)
        assert parsed.title == "Titulo"
        assert _texts(parsed) == ["Contenido"]

    async def test_a_paragraph_is_closed_by_a_following_block(self) -> None:
        markup = "<div id=m><p>Intro text<h2>Requisitos</h2><p>Body</div>"
        out = await _clean(HtmlMainContent, await _parse(markup), include_selectors=["#m"])
        assert _structure(out)[-2:] == [
            ((out.title,), "Intro text"),
            ((out.title, "Requisitos"), "Body"),
        ]

    async def test_a_paragraph_closed_by_a_div_does_not_swallow_it(self) -> None:
        markup = "<div id=a><p>one<div id=b>two</div>three</div>"
        out = await _clean(HtmlMainContent, await _parse(markup), include_selectors=["#a"])
        assert "|".join(_texts(out)).replace("\n\n", "|") == "one|two|three"

    async def test_unclosed_cells_and_rows_make_one_row_each(self) -> None:
        markup = "<table><tr><td>a<td>b<tr><td>c<td>d</table><p>after</p>"
        out = await _clean(HtmlMainContent, await _parse(markup))
        assert _texts(out) == ["a | b\n\nc | d\n\nafter"]

    async def test_a_selector_sees_unclosed_list_items_as_siblings(self) -> None:
        root = parse_html_tree("<ul id=o><li>a<ul id=i><li>b<li>c</ul><li>d</ul>")
        outer = select_all(root, [parse_selector("#o > li")])
        inner = select_all(root, [parse_selector("#i > li")])
        assert [normalize_space(text_content(e)) for e in outer] == ["a b c", "d"]
        assert [normalize_space(text_content(e)) for e in inner] == ["b", "c"]

    async def test_entities_comments_and_rcdata(self) -> None:
        parsed = await _parse("<body><!-- <p>hidden</p> --><p>a&nbsp;b &amp; &aacute; &#x41;</p>")
        assert _texts(parsed) == ["a b & á A"]

    async def test_xml_declaration_and_blank_documents_do_not_raise(self) -> None:
        xhtml = '<?xml version="1.0" encoding="utf-8"?><html><body><p>x</p></body></html>'
        assert _texts(await _parse(xhtml)) == ["x"]
        for blank in ("", "   ", "<!-- only a comment -->"):
            assert (await _parse(blank)).sections == ()

    async def test_attribute_names_and_tags_are_case_insensitive(self) -> None:
        root = parse_html_tree('<DIV ID="X" CLASS="A b" class="c">t</DIV>')
        assert len(select_all(root, [parse_selector("div#X.A")])) == 1
        assert select_all(root, [parse_selector("#x")]) == []  # ids are case-sensitive

    async def test_a_start_tag_flooded_with_attributes_is_rejected_fast(self) -> None:
        flood = "<div " + " ".join(f"a{i}=v" for i in range(5000)) + ">x</div>"
        with pytest.raises(UnparseableDocumentError, match="attributes"):
            await _parse(flood)


class TestSelectorEngine:
    def _count(self, markup: str, selector: str) -> int:
        return len(select_all(parse_html_tree(markup), [parse_selector(selector)]))

    def test_non_matching_deep_selectors_finish_within_the_time_budget(self) -> None:
        markup = "<body>" + "<div>" * 25 + "<p>x</p>" * 2000 + "</div>" * 25 + "</body>"
        root = parse_html_tree(markup)
        for selector in (
            "section div div div p",
            "div div div div section",
            "div > div > div > section p",
            "p p p p p",
        ):
            compiled = parse_selector(selector)
            started = time.perf_counter()
            assert select_all(root, [compiled]) == []
            assert time.perf_counter() - started < 0.5, selector

    def test_matching_deep_selector_is_also_fast(self) -> None:
        markup = "<body>" + "<div>" * 25 + "<p>x</p>" * 2000 + "</div>" * 25 + "</body>"
        root = parse_html_tree(markup)
        started = time.perf_counter()
        assert len(select_all(root, [parse_selector("div div div div p")])) == 2000
        assert time.perf_counter() - started < 0.5

    @pytest.mark.parametrize("operator", ["*=", "^=", "$="])
    def test_an_empty_value_matches_nothing_as_in_css(self, operator: str) -> None:
        assert self._count('<p title="abc">x</p>', f'[title{operator}""]') == 0
        assert self._count('<p title="abc">x</p>', f"[title{operator}'']") == 0

    def test_empty_equality_matches_an_empty_attribute(self) -> None:
        assert self._count('<p title="">x</p><p title="a">y</p>', '[title=""]') == 1

    def test_values_with_quotes_cannot_break_the_expression(self) -> None:
        markup = """<p title="it's">a</p><p title='say "hi"'>b</p>"""
        assert self._count(markup, '[title="it\'s"]') == 1
        assert self._count(markup, "[title='say \"hi\"']") == 1
        assert self._count(markup, "[title*=\"' or '1\"]") == 0

    def test_suffix_longer_than_the_value_does_not_match(self) -> None:
        assert self._count('<a href="x.html">a</a>', '[href$="abcdex.html"]') == 0
        assert self._count('<a href="x.html">a</a>', '[href$=".html"]') == 1

    def test_namespaced_attribute_names_and_class_lists(self) -> None:
        markup = '<svg-like xlink:href="#a" class="one  two\tthree">t</svg-like>'
        assert self._count(markup, "[xlink:href]") == 1
        assert self._count(markup, ".two.three") == 1
        assert self._count(markup, ".tw") == 0

    def test_results_are_document_ordered_and_deduplicated(self) -> None:
        root = parse_html_tree("<body><p class=a>1</p><p class=a id=x>2</p><i>3</i></body>")
        found = select_all(root, [parse_selector("#x, .a, p"), parse_selector("p.a")])
        assert [normalize_space(text_content(e)) for e in found] == ["1", "2"]


class TestExcludedSubtrees:
    HIDDEN = (
        '<body><div id=main><div style="display: none"><p class=t>SECRET</p>'
        "<h2>Hidden heading</h2></div><p class=t>visible</p></div></body>"
    )

    async def test_nodes_inside_an_excluded_subtree_cannot_be_included(self) -> None:
        out = await _clean(
            HtmlMainContent,
            await _parse(self.HIDDEN),
            include_selectors=["p.t"],
            exclude_selectors=['[style*="display:none"]'],
        )
        assert _texts(out) == ["visible"]

    async def test_an_excluded_subtree_adds_no_text_or_heading_to_a_container(self) -> None:
        out = await _clean(
            HtmlMainContent,
            await _parse(self.HIDDEN),
            include_selectors=["#main"],
            exclude_selectors=['[style*="display:none"]'],
        )
        flat = " ".join(_texts(out)) + " ".join(h for s in out.sections for h in s.headings)
        assert "SECRET" not in flat
        assert "Hidden heading" not in flat
        assert "visible" in flat

    async def test_the_text_after_an_excluded_element_is_kept(self) -> None:
        markup = "<body><p>one <span class=x>gone</span> two</p></body>"
        out = await _clean(HtmlMainContent, await _parse(markup), exclude_selectors=[".x"])
        assert _texts(out) == ["one  two".replace("  ", " ")]

    async def test_html_collect_still_reads_what_main_content_drops(self) -> None:
        out = await _clean(
            HtmlCollect,
            await _parse(self.HIDDEN),
            fields={"secret": {"selector": '[style*="display:none"] p'}},
        )
        assert out.collected["secret"] == ("SECRET",)


class TestBlockBoundaries:
    """Text of a block that contains other blocks never glues words together."""

    async def _blocks(self, markup: str, **params: Any) -> list[str]:
        out = await _clean(HtmlMainContent, await _parse(markup), **params)
        return "\n\n".join(_texts(out)).split("\n\n")

    async def test_cells_with_paragraphs_and_line_breaks(self) -> None:
        markup = "<table><tr><td><p>uno</p><p>dos</p></td><td>tres<br>cuatro</td></tr></table>"
        assert await self._blocks(markup) == ["uno dos | tres cuatro"]

    async def test_list_items_with_block_children_are_not_glued(self) -> None:
        markup = "<ul><li><div>Primero</div><div>segundo</div></li><li><p>uno</p></li></ul>"
        assert await self._blocks(markup) == ["- Primero segundo", "- uno"]

    async def test_nested_lists_become_separate_items(self) -> None:
        markup = "<ul><li>Requisitos<ul><li>Tener 65<li>Afiliado</ul></li><li>Plazo</li></ul>"
        assert await self._blocks(markup) == ["- Requisitos", "- Tener 65", "- Afiliado", "- Plazo"]

    async def test_text_after_a_nested_list_keeps_its_own_item(self) -> None:
        markup = "<ul><li>Uno<ul><li>sub</li></ul>cierre</li></ul>"
        assert await self._blocks(markup) == ["- Uno", "- sub", "- cierre"]

    async def test_definition_with_a_nested_list(self) -> None:
        markup = "<dl><dt>Termino</dt><dd>Definicion<ul><li>a</li><li>b</li></ul></dd></dl>"
        assert await self._blocks(markup) == ["Termino", "Definicion", "- a", "- b"]

    async def test_inline_markup_stays_glued(self) -> None:
        assert await self._blocks("<p>pre<b>fix</b>ed <i>word</i>s</p>") == ["prefixed words"]

    async def test_a_block_inside_an_inline_wrapper_is_separated(self) -> None:
        assert await self._blocks("<span><div>a</div><div>b</div></span>") == ["a b"]


class TestHeadingsAndTitle:
    async def test_a_heading_equal_to_the_title_is_skipped(self) -> None:
        markup = (
            "<body><div id=t><h1>Que es el plan?</h1></div>"
            "<div id=c><h2>Que es el plan?</h2><p>Es un beneficio.</p>"
            "<h2>Otro</h2><p>x</p></div>"
            "</body>"
        )
        out = await _clean(
            HtmlMainContent,
            await _parse(markup),
            title_selector="#t h1",
            include_selectors=["#c"],
        )
        assert _structure(out) == [
            (("Que es el plan?",), "Es un beneficio."),
            (("Que es el plan?", "Otro"), "x"),
        ]

    async def test_headings_keep_their_natural_level_below_the_title(self) -> None:
        markup = "<body><h2>A</h2><p>a</p><h3>B</h3><p>b</p><h2>C</h2><p>c</p></body>"
        out = await _clean(HtmlMainContent, await _parse(markup))
        assert [s.headings for s in out.sections] == [
            (out.title, "A"),
            (out.title, "A", "B"),
            (out.title, "C"),
        ]


class TestCharsetSniffing:
    async def test_meta_charset_is_honoured_before_the_cp1252_fallback(self) -> None:
        html = (
            '<html><head><meta charset="iso-8859-2"><title>Zażółć</title></head>'
            "<body><p>Zażółć gęślą</p></body></html>"
        )
        parsed = await _parse_bytes(html.encode("iso-8859-2"))
        assert parsed.title == "Zażółć"
        assert _texts(parsed) == ["Zażółć gęślą"]

    async def test_http_equiv_content_type_form(self) -> None:
        html = (
            '<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-2"><p>gęślą</p>'
        )
        assert _texts(await _parse_bytes(html.encode("iso-8859-2"))) == ["gęślą"]

    async def test_without_a_declaration_it_falls_back_to_cp1252(self) -> None:
        assert _texts(await _parse_bytes("<p>año</p>".encode("cp1252"))) == ["año"]

    async def test_a_declaration_after_the_first_kilobyte_is_ignored(self) -> None:
        padding = "<!--" + "x" * 1100 + "-->"
        html = padding + '<meta charset="iso-8859-2"><p>ą</p>'
        parsed = await _parse_bytes(html.encode("iso-8859-2"))
        assert _texts(parsed) != ["ą"]

    async def test_an_unknown_declared_charset_falls_back(self) -> None:
        html = '<meta charset="no-such-charset"><p>año</p>'
        assert _texts(await _parse_bytes(html.encode("cp1252"))) == ["año"]


async def _parse_bytes(content: bytes, uri: str = "https://x.cl/a.html") -> ParsedDocument:
    return await SimpleParser().parse(SourceFile(uri=uri, mime_type="text/html", content=content))


class TestCollectedValues:
    async def test_long_or_non_ascii_digit_captures_stay_strings(self) -> None:
        markup = (
            '<body><a href="w3-article-123.html">a</a>'
            '<a href="w3-article-1234567890123456789.html">b</a>'
            '<a href="w3-article-١٢٣.html">c</a>'
            '<a href="w3-article-007.html">d</a></body>'
        )
        out = await _clean(
            HtmlCollect,
            await _parse(markup),
            fields={"ids": {"selector": "a", "attr": "href", "regex": r"w3-article-(\d+)\.html"}},
        )
        assert out.collected["ids"] == (123, "1234567890123456789", "١٢٣", 7)

    async def test_eighteen_digits_are_still_an_int(self) -> None:
        value = "9" * 18
        out = await _clean(
            HtmlCollect,
            await _parse(f"<body><b>{value}</b></body>"),
            fields={"n": {"selector": "b", "regex": r"(\d+)"}},
        )
        assert out.collected["n"] == (int(value),)

    async def test_an_oversized_value_is_not_matched_by_the_regex(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        big = "x" * (MAX_REGEX_INPUT_CHARS + 1) + " 42"
        out = await _clean(
            HtmlCollect,
            await _parse(f"<body><b>{big}</b><b>7</b></body>"),
            fields={"n": {"selector": "b", "regex": r"(\d+)"}},
        )
        assert out.collected["n"] == (7,)
        assert "skipped an input" in caplog.text
