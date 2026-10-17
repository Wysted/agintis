"""Parser "ocr_index_card": text layer, else OCR of every page, else the index card."""

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from types import SimpleNamespace
from typing import Any

import pytest

from agintis.modules.knowledge.api import ScannedDocumentError, SourceFile
from agintis.modules.knowledge.infrastructure.parsers import (
    OcrIndexCardParser,
    OcrNotInstalledError,
    SimpleParser,
    require_ocr_extra,
)
from agintis.modules.knowledge.infrastructure.parsers.ocr import OCR_ENGINE, RapidOcrPages
from agintis.modules.knowledge.infrastructure.registry import build_parsers
from tests.support.documents import make_blank_pdf, make_pdf

CARD = {"title": "Resolución N° 1266", "description": "Complementa la individualización."}


class FakeOcr:
    """`PageOcr` returning canned page texts and recording each call."""

    def __init__(self, pages: Sequence[str]) -> None:
        self.pages = pages
        self.calls = 0

    def __call__(self, content: bytes) -> Sequence[str]:
        self.calls += 1
        return self.pages


def _pdf(content: bytes, **meta: Any) -> SourceFile:
    return SourceFile("u.pdf", "application/pdf", content, document_metadata=meta)


class TestOcrIndexCardParser:
    async def test_a_scanned_pdf_is_read_by_ocr_page_by_page(self) -> None:
        ocr = FakeOcr(["VISTOS:\nLo dispuesto", "", "RESUELVO:\n1. Apruébase"])
        parsed = await OcrIndexCardParser(ocr=ocr).parse(_pdf(make_blank_pdf(3), **CARD))
        assert parsed.title == "Resolución N° 1266"
        # Empty pages are dropped, the others keep their page number.
        assert [(s.text, s.page) for s in parsed.sections] == [
            ("VISTOS:\nLo dispuesto", 1),
            ("RESUELVO:\n1. Apruébase", 3),
        ]
        assert parsed.collected["content_mode"] == ("ocr",)
        assert ocr.calls == 1

    async def test_without_a_title_the_ocr_text_is_titled_by_its_uri(self) -> None:
        parsed = await OcrIndexCardParser(ocr=FakeOcr(["text"])).parse(_pdf(make_blank_pdf(1)))
        assert parsed.title == "u"
        assert parsed.sections[0].text == "text"

    async def test_no_ocr_text_falls_back_to_the_index_card(self) -> None:
        parser = OcrIndexCardParser(ocr=FakeOcr([" ", "\x00"]))
        parsed = await parser.parse(_pdf(make_blank_pdf(2), **CARD))
        assert [s.text for s in parsed.sections] == [
            "Resolución N° 1266\n\nComplementa la individualización."
        ]
        assert parsed.collected["content_mode"] == ("index_card",)

    async def test_no_ocr_text_and_no_card_still_fails(self) -> None:
        with pytest.raises(ScannedDocumentError):
            await OcrIndexCardParser(ocr=FakeOcr([""])).parse(_pdf(make_blank_pdf(1)))

    async def test_a_pdf_with_text_skips_ocr(self) -> None:
        ocr = FakeOcr(["never"])
        content = make_pdf(["Page one has plenty of text.", "Page two as well."])
        parsed = await OcrIndexCardParser(ocr=ocr).parse(_pdf(content, **CARD))
        assert parsed.sections == (await SimpleParser().parse(_pdf(content))).sections
        assert parsed.collected["content_mode"] == ("full_text",)
        assert ocr.calls == 0

    def test_its_version_names_the_text_parser_and_the_ocr_engine(self) -> None:
        assert OcrIndexCardParser.version == f"{SimpleParser.version}+{OCR_ENGINE}"

    def test_the_engine_in_the_version_is_the_installed_one(self) -> None:
        try:
            installed = installed_version("rapidocr")
        except PackageNotFoundError:
            pytest.skip("the ocr extra is not installed")
        assert OCR_ENGINE.startswith(f"rapidocr-{installed}/")

    def test_it_is_registered(self) -> None:
        parsers = build_parsers(
            max_bytes=10**6, max_pages=10, max_uncompressed_bytes=10**6, max_compression_ratio=10.0
        )
        assert isinstance(parsers["ocr_index_card"], OcrIndexCardParser)


class TestRequireOcrExtra:
    def test_a_missing_package_names_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agintis.modules.knowledge.infrastructure.parsers.ocr.find_spec",
            lambda name: None if name == "rapidocr" else object(),
        )
        with pytest.raises(OcrNotInstalledError, match=r"'ocr' extra.*Missing: rapidocr"):
            require_ocr_extra()


class TestRapidOcrPages:
    """The real renderer with a fake engine (needs pypdfium2 from the `ocr` extra)."""

    def test_each_page_is_rendered_and_its_text_boxes_joined_by_lines(self) -> None:
        pytest.importorskip("pypdfium2")
        images: list[Any] = []
        answers = iter([("line one", "line two"), None])

        def engine(image: Any) -> SimpleNamespace:
            images.append(image)
            return SimpleNamespace(txts=next(answers))

        created: list[int] = []

        def factory() -> Any:
            created.append(1)
            return engine

        pages = RapidOcrPages(factory)(make_pdf(["anything", ""]))
        assert pages == ["line one\nline two", ""]
        # 612 pt wide at 300 dpi, RGB (pdfium may round the height up a pixel).
        assert [image.shape[1:] for image in images] == [(2550, 3)] * 2
        RapidOcrPages(factory)  # building alone creates no engine
        assert created == [1]
