import io
import logging
import time
import zipfile

import pytest
from pypdf import PdfReader

from agintis.modules.knowledge.api import (
    DocumentTooLargeError,
    ScannedDocumentError,
    SourceFile,
    UnsupportedDocumentTypeError,
)
from agintis.modules.knowledge.application import UnparseableDocumentError
from agintis.modules.knowledge.infrastructure.parsers import ParseLimits, SimpleParser
from agintis.modules.knowledge.infrastructure.parsers import pdf as pdf_module
from agintis.modules.knowledge.infrastructure.parsers._title import title_from_uri
from agintis.modules.knowledge.infrastructure.text import (
    decode_text,
    mime_charset,
    strip_unstorable,
)
from tests.support.documents import (
    make_blank_pdf,
    make_docx,
    make_docx_with_merged_cells,
    make_encrypted_pdf,
    make_pdf,
    with_extra_zip_member,
)

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SPANISH = "Camión de señalización: ¿qué tal? Ñandú, acción y corazón - 5 € «ok»"


def _src(content: bytes, mime: str, uri: str = "file:///docs/guide.bin") -> SourceFile:
    return SourceFile(uri=uri, mime_type=mime, content=content)


class TestPdf:
    async def test_one_section_per_page_with_page_numbers(self) -> None:
        pdf = make_pdf(["First page has enough text.", "Second page has more text."])
        parsed = await SimpleParser().parse(_src(pdf, PDF, "file:///docs/manual.pdf"))
        assert parsed.title == "manual"
        assert [(s.page, s.text) for s in parsed.sections] == [
            (1, "First page has enough text."),
            (2, "Second page has more text."),
        ]

    async def test_blank_page_in_text_pdf_is_skipped_but_pages_keep_numbering(self) -> None:
        pdf = make_pdf(["Page one carries the content.", "", "Page three also has content."])
        parsed = await SimpleParser().parse(_src(pdf, PDF))
        assert [s.page for s in parsed.sections] == [1, 3]

    async def test_scanned_pdf_suggests_the_ocr_parser(self) -> None:
        with pytest.raises(ScannedDocumentError, match="ocr_index_card"):
            await SimpleParser().parse(_src(make_blank_pdf(3), PDF))

    async def test_minority_of_textless_pages_is_dropped_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        text = "This page has a real text layer."
        pdf = make_pdf([text, "", text, "", text])  # 2 of 5 textless: below the 50% threshold
        with caplog.at_level(logging.WARNING):
            parsed = await SimpleParser().parse(_src(pdf, PDF, "file:///docs/mixed.pdf"))
        assert [s.page for s in parsed.sections] == [1, 3, 5]
        assert "Dropping 2 textless page(s)" in caplog.text
        assert "[2, 4]" in caplog.text

    async def test_majority_of_textless_pages_means_scanned(self) -> None:
        text = "This page has a real text layer."
        pdf = make_pdf([text, "", "", "", text])  # 3 of 5 textless
        with pytest.raises(ScannedDocumentError, match="3 of 5"):
            await SimpleParser().parse(_src(pdf, PDF))

    async def test_corrupt_pdf(self) -> None:
        with pytest.raises(UnparseableDocumentError):
            await SimpleParser().parse(_src(b"not a pdf at all", PDF))

    async def test_truncated_pdf(self) -> None:
        with pytest.raises(UnparseableDocumentError):
            await SimpleParser().parse(_src(make_pdf(["Some page text here."])[:80], PDF))

    async def test_any_pypdf_failure_becomes_unparseable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: object, **kwargs: object) -> PdfReader:
            raise RecursionError("pathological structure")

        monkeypatch.setattr(pdf_module, "PdfReader", explode)
        with pytest.raises(UnparseableDocumentError, match="pathological"):
            await SimpleParser().parse(_src(make_pdf(["Some page text here."]), PDF))

    async def test_zero_page_pdf_is_unparseable_not_scanned(self) -> None:
        with pytest.raises(UnparseableDocumentError, match="no pages"):
            await SimpleParser().parse(_src(make_pdf([]), PDF))

    @pytest.mark.parametrize("algorithm", ["AES-128", "AES-256"])
    async def test_empty_user_password_pdf_is_decrypted(self, algorithm: str) -> None:
        pdf = make_encrypted_pdf(["Owner restricted but readable."], "", algorithm=algorithm)
        assert PdfReader(io.BytesIO(pdf)).is_encrypted
        parsed = await SimpleParser().parse(_src(pdf, PDF))
        assert [s.text for s in parsed.sections] == ["Owner restricted but readable."]

    async def test_pdf_with_a_user_password_is_rejected_clearly(self) -> None:
        pdf = make_encrypted_pdf(["Secret page content here."], "hunter2")
        with pytest.raises(UnparseableDocumentError, match="password protected"):
            await SimpleParser().parse(_src(pdf, PDF))


class TestDocx:
    async def test_sections_follow_headings(self) -> None:
        content = make_docx(
            [
                ("title", "Handbook"),
                ("p", "Preface text."),
                ("h1", "Setup"),
                ("p", "Install it."),
                ("h2", "Linux"),
                ("p", "Use apt."),
                ("h1", "Usage"),
                ("p", "Run it."),
            ],
            table=[["a", "b"], ["c", "d"]],
        )
        parsed = await SimpleParser().parse(_src(content, DOCX))
        assert parsed.title == "Handbook"
        assert [(s.headings, s.text) for s in parsed.sections] == [
            (("Handbook",), "Preface text."),
            (("Setup",), "Install it."),
            (("Setup", "Linux"), "Use apt."),
            (("Usage",), "Run it.\n\na | b\nc | d"),
        ]
        assert all(s.page is None for s in parsed.sections)

    async def test_merged_cells_appear_once(self) -> None:
        parsed = await SimpleParser().parse(_src(make_docx_with_merged_cells(), DOCX))
        assert [s.text for s in parsed.sections] == ["Merged | tail"]

    async def test_corrupt_docx(self) -> None:
        with pytest.raises(UnparseableDocumentError):
            await SimpleParser().parse(_src(b"garbage", DOCX))

    async def test_zip_that_is_not_a_docx(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("hello.txt", "hi")
        with pytest.raises(UnparseableDocumentError):
            await SimpleParser().parse(_src(buffer.getvalue(), DOCX))


class TestMarkdown:
    MD = (
        b"intro line\n\n# Title\n\nbody one\n\n## Sub A\n\ntext a\n\n```\n# not a heading\n```\n\n"
        b"## Sub B\n\ntext b\n\n# Next\n\nlast\n"
    )

    @staticmethod
    async def _parse(text: str) -> tuple[list[tuple[tuple[str, ...], str]], str]:
        parsed = await SimpleParser().parse(_src(text.encode(), "text/markdown", "file:///a/n.md"))
        return [(s.headings, s.text) for s in parsed.sections], parsed.title

    async def test_sections_by_heading_path(self) -> None:
        parsed = await SimpleParser().parse(_src(self.MD, "text/markdown", "file:///a/notes.md"))
        assert parsed.title == "Title"
        assert [(s.headings, s.text) for s in parsed.sections] == [
            ((), "intro line"),
            (("Title",), "body one"),
            (("Title", "Sub A"), "text a\n\n```\n# not a heading\n```"),
            (("Title", "Sub B"), "text b"),
            (("Next",), "last"),
        ]

    async def test_title_falls_back_to_file_name(self) -> None:
        parsed = await SimpleParser().parse(
            _src(b"## only h2\n\nx", "text/markdown", "s3://b/n.md")
        )
        assert parsed.title == "n"

    async def test_extension_fallback_for_generic_mime(self) -> None:
        parsed = await SimpleParser().parse(
            _src(b"# H\n\nx", "application/octet-stream", "file:///a/readme.md?v=1")
        )
        assert parsed.sections[0].headings == ("H",)

    async def test_closing_hash_sequence_needs_a_preceding_space(self) -> None:
        sections, _ = await self._parse(
            "# C#\n\na\n\n# Title #\n\nb\n\n# Tag#\n\nc\n\n## Two ##  \n\nd"
        )
        assert [h for h, _ in sections] == [("C#",), ("Title",), ("Tag#",), ("Tag#", "Two")]

    async def test_hashtag_without_space_is_not_a_heading(self) -> None:
        sections, _ = await self._parse("#hashtag line\n\n####### seven\n\n#")
        assert [h for h, _ in sections] == [()]

    async def test_setext_headings(self) -> None:
        sections, title = await self._parse("Main\n=====\n\ntext\n\nSub title\n---\n\nmore text")
        assert sections == [(("Main",), "text"), (("Main", "Sub title"), "more text")]
        assert title == "Main"

    async def test_thematic_break_after_a_blank_line_is_not_a_heading(self) -> None:
        sections, _ = await self._parse("para\n\n---\n\nnext")
        assert sections == [((), "para\n\n---\n\nnext")]

    async def test_fence_closes_only_with_the_same_char_and_length(self) -> None:
        text = (
            "````\n```\n# inner one\n````\n# Real\n\nx\n\n~~~\n```\n# inner two\n~~~\n# Last\n\ny"
        )
        sections, _ = await self._parse(text)
        assert [h for h, _ in sections] == [(), ("Real",), ("Last",)]
        assert "# inner one" in sections[0][1]
        assert "# inner two" in sections[1][1]

    async def test_trailing_heading_attribute_block_is_dropped(self) -> None:
        text = (
            "# Primeros Pasos { #first-steps }\n\na\n\n## Dos {#dos .cls key=v} ##\n\nb\n\n"
            "Setext {.big}\n---\n\nc\n\n## Use {x}\n\nd\n\n## Mid { #m } tail\n\ne"
        )
        sections, title = await self._parse(text)
        assert title == "Primeros Pasos"
        assert [h[-1] for h, _ in sections] == [
            "Primeros Pasos",
            "Dos",
            "Setext",
            "Use {x}",
            "Mid { #m } tail",
        ]

    async def test_attribute_block_variants(self) -> None:
        heads = ["# A {: #kd}", "## B {x=}", "## C {.5}", "## D {.cls}", "## E #", "## F # #"]
        heads += ["## G ##    {#foo}", "### H ### {: #hash }"]
        sections, _ = await self._parse("\n\nz\n\n".join(heads) + "\n\nz")
        assert [h[-1] for h, _ in sections] == ["A", "B {x=}", "C {.5}", "D", "E", "F #", "G", "H"]

    async def test_hostile_heading_is_parsed_in_linear_time(self) -> None:
        pad = " " * 100_000
        heads = [f"# a{pad}{{#x", f"# b{pad}#x", f"# c {pad}{{#x}}{pad}", f"# d {'#' * 100_000}x"]
        text = "\n\nz\n\n".join(heads) + "\n\nz"
        start = time.perf_counter()
        sections, _ = await self._parse(text)
        assert time.perf_counter() - start < 5  # quadratic took >5 s for the first line alone
        assert [h[-1] for h, _ in sections][2] == "c"

    async def test_yaml_front_matter_is_skipped(self) -> None:
        sections, title = await self._parse("---\ntitle: x\n# not a heading\n---\n# Real\ntext")
        assert sections == [(("Real",), "text")]
        assert title == "Real"

    async def test_unclosed_front_matter_is_kept_as_text(self) -> None:
        sections, _ = await self._parse("---\nkeep me\n")
        assert sections == [((), "---\nkeep me")]


class TestPlainText:
    async def test_single_section_with_decoded_text(self) -> None:
        parsed = await SimpleParser().parse(
            _src("﻿hola\r\n\r\nmundo ñ".encode(), "text/plain; charset=utf-8", "file:///t.txt")
        )
        assert parsed.title == "t"
        assert [s.text for s in parsed.sections] == ["hola\n\nmundo ñ"]

    async def test_empty_text_has_no_sections(self) -> None:
        parsed = await SimpleParser().parse(_src(b"  \n ", "text/plain"))
        assert parsed.sections == ()


class TestEncoding:
    async def _text(self, content: bytes, mime: str) -> str:
        parsed = await SimpleParser().parse(_src(content, mime, "file:///t.txt"))
        return parsed.sections[0].text

    async def test_declared_cp1252_charset_is_honored(self) -> None:
        raw = SPANISH.encode("cp1252")
        assert await self._text(raw, "text/plain; charset=windows-1252") == SPANISH

    async def test_declared_latin1_charset_is_honored(self) -> None:
        raw = "Camión señalización ¿qué?".encode("latin-1")
        assert (
            await self._text(raw, 'text/plain; charset="iso-8859-1"') == "Camión señalización ¿qué?"
        )

    async def test_undeclared_legacy_text_falls_back_to_cp1252(self) -> None:
        assert await self._text(SPANISH.encode("cp1252"), "text/plain") == SPANISH

    async def test_undeclared_utf8_is_decoded_strictly(self) -> None:
        assert await self._text(SPANISH.encode(), "text/plain") == SPANISH

    async def test_wrong_declared_charset_falls_back_to_detection(self) -> None:
        # Declared UTF-8 but the bytes are cp1252: strict decoding fails, detection recovers.
        assert await self._text(SPANISH.encode("cp1252"), "text/plain; charset=utf-8") == SPANISH

    async def test_unknown_declared_charset_falls_back_to_detection(self) -> None:
        assert await self._text(SPANISH.encode(), "text/plain; charset=x-bogus") == SPANISH

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
    async def test_utf16_with_bom_is_detected(self, encoding: str) -> None:
        raw = b"\xff\xfe" + SPANISH.encode("utf-16-le")
        if encoding == "utf-16-be":
            raw = b"\xfe\xff" + SPANISH.encode("utf-16-be")
        assert await self._text(raw, "text/plain") == SPANISH

    async def test_utf8_bom_is_dropped(self) -> None:
        assert await self._text(b"\xef\xbb\xbf" + SPANISH.encode(), "text/plain") == SPANISH

    async def test_undecodable_bytes_never_raise(self) -> None:
        assert "�" in await self._text(b"abc \x81\x8d\x8f\x90\x9d xyz", "text/plain")

    def test_markdown_honors_charset_too(self) -> None:
        assert decode_text("# Año\n".encode("cp1252"), "windows-1252") == "# Año\n"

    @pytest.mark.parametrize(
        ("mime", "expected"),
        [
            ("text/plain", None),
            ("text/plain; charset=UTF-8", "utf-8"),
            ('text/markdown; charset="latin-1"; format=flowed', "latin-1"),
        ],
    )
    def test_mime_charset_extraction(self, mime: str, expected: str | None) -> None:
        assert mime_charset(mime) == expected


class TestUnstorableCharacters:
    async def test_nul_is_stripped_from_plain_text(self) -> None:
        parsed = await SimpleParser().parse(_src(b"he\x00llo\x00 wor\x00ld", "text/plain"))
        assert [s.text for s in parsed.sections] == ["hello world"]

    async def test_nul_is_stripped_from_titles_and_headings(self) -> None:
        parsed = await SimpleParser().parse(_src(b"# Ti\x00tle\n\nbo\x00dy", "text/markdown"))
        assert parsed.title == "Title"
        assert [(s.headings, s.text) for s in parsed.sections] == [(("Title",), "body")]

    async def test_section_that_is_only_nul_disappears(self) -> None:
        parsed = await SimpleParser().parse(
            _src(b"# H\n\n\x00\x00\n\n# G\n\nreal", "text/markdown")
        )
        assert [s.headings for s in parsed.sections] == [("G",)]

    async def test_nul_is_stripped_from_pdf_text_and_metadata(self) -> None:
        parsed = await SimpleParser().parse(_src(make_pdf(["Page with a \x00 inside it."]), PDF))
        assert "\x00" not in parsed.sections[0].text

    def test_lone_surrogates_are_stripped_too(self) -> None:
        assert strip_unstorable("a\x00b\ud800c") == "abc"


class TestResourceLimits:
    async def test_oversized_source_is_rejected_before_parsing(self) -> None:
        parser = SimpleParser(ParseLimits(max_bytes=100))
        with pytest.raises(DocumentTooLargeError, match="limit 100"):
            await parser.parse(_src(b"x" * 101, "text/plain"))

    async def test_source_at_the_limit_is_accepted(self) -> None:
        parsed = await SimpleParser(ParseLimits(max_bytes=100)).parse(
            _src(b"x" * 100, "text/plain")
        )
        assert parsed.sections

    async def test_pdf_page_limit(self) -> None:
        pdf = make_pdf([f"Page number {i} has text." for i in range(4)])
        with pytest.raises(DocumentTooLargeError, match="4 pages"):
            await SimpleParser(ParseLimits(max_pages=3)).parse(_src(pdf, PDF))
        assert (await SimpleParser(ParseLimits(max_pages=4)).parse(_src(pdf, PDF))).sections

    async def test_zip_bomb_is_rejected_by_declared_uncompressed_size(self) -> None:
        bomb = with_extra_zip_member(make_docx([("p", "hi")]), "word/bomb.bin", b"\0" * (2 << 20))
        assert len(bomb) < 40_000  # a few KB expanding to 2 MiB
        parser = SimpleParser(ParseLimits(max_uncompressed_bytes=1 << 20))
        with pytest.raises(DocumentTooLargeError, match="expands to"):
            await parser.parse(_src(bomb, DOCX))

    async def test_zip_bomb_is_rejected_by_compression_ratio(self) -> None:
        bomb = with_extra_zip_member(make_docx([("p", "hi")]), "word/bomb.bin", b"\0" * (2 << 20))
        parser = SimpleParser(ParseLimits(max_compression_ratio=50.0))
        with pytest.raises(DocumentTooLargeError, match="compression ratio"):
            await parser.parse(_src(bomb, DOCX))

    async def test_normal_docx_passes_the_default_limits(self) -> None:
        parsed = await SimpleParser().parse(_src(make_docx([("h1", "A"), ("p", "text")]), DOCX))
        assert parsed.sections

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_bytes": 0},
            {"max_pages": 0},
            {"max_uncompressed_bytes": 0},
            {"max_compression_ratio": 1},
        ],
    )
    def test_invalid_limits_are_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="must be greater than"):
            ParseLimits(**kwargs)  # type: ignore[arg-type]


async def test_unsupported_type() -> None:
    with pytest.raises(UnsupportedDocumentTypeError, match="image/png"):
        await SimpleParser().parse(_src(b"\x89PNG", "image/png", "file:///a.png"))


class TestTitleFromUri:
    @pytest.mark.parametrize(
        ("uri", "title"),
        [
            ("https://h/apps/GetFile.php?id=003&namefile=NCG-SP/NP0000372.pdf", "NP0000372"),
            ("https://h/GetFile.php?namefile=a%2Fb%2FCAFP2425.pdf", "CAFP2425"),
            ("https://h/GetFile.php?file=C:\\docs\\norm.docx", "norm"),
            ("https://h/docs/manual.pdf", "manual"),
            ("https://h/docs/manual.pdf?download=1&v=2", "manual"),
            ("https://h/docs/manual.pdf?ref=other.pdf", "manual"),
            ("https://h/apps/GetFile.php?ref=x&namefile=NCG-SP/N0372.pdf", "N0372"),
            ("https://h/GetFile.php?id=003", "GetFile"),
            ("https://h/GetFile.php?next=https://x.cl/", "GetFile"),
            ("https://h/GetFile.php?note=hello+world.+Bye", "GetFile"),
            ("file:///a/readme.md?v=1", "readme"),
            ("s3://b/n.md", "n"),
            ("https://h/articles-14843_recurso_1.html", "articles-14843_recurso_1"),
            ("https://h/", "https://h/"),
        ],
    )
    def test_cases(self, uri: str, title: str) -> None:
        assert title_from_uri(uri) == title
