"""Parser "ocr_index_card": text layer, else OCR of every page, else the index card."""

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from types import SimpleNamespace
from typing import Any

import pytest

from agintis.modules.knowledge.api import (
    DocumentTooLargeError,
    ScannedDocumentError,
    SourceFile,
    UnparseableDocumentError,
)
from agintis.modules.knowledge.infrastructure.parsers import (
    OcrIndexCardParser,
    OcrNotInstalledError,
    ParseLimits,
    SimpleParser,
    require_ocr_extra,
)
from agintis.modules.knowledge.infrastructure.parsers import ocr as ocr_module
from agintis.modules.knowledge.infrastructure.parsers.ocr import OCR_ENGINE, RapidOcrPages
from agintis.modules.knowledge.infrastructure.registry import build_parsers
from tests.support.documents import make_blank_pdf, make_pdf

CARD = {"title": "Resolución N° 1266", "description": "Complementa la individualización."}


class FakeOcr:
    """`PageOcr` returning canned page texts and recording each call."""

    def __init__(self, pages: Sequence[str]) -> None:
        self.pages = pages
        self.calls = 0
        self.max_pages: int | None = None

    def __call__(self, content: bytes, max_pages: int) -> Sequence[str]:
        self.calls += 1
        self.max_pages = max_pages
        return self.pages


def _pdf(content: bytes, **meta: Any) -> SourceFile:
    return SourceFile("u.pdf", "application/pdf", content, document_metadata=meta)


class TestOcrIndexCardParser:
    async def test_a_scanned_pdf_is_read_by_ocr_page_by_page(self) -> None:
        ocr = FakeOcr(["VISTOS:\nLo dispuesto", "", "RESUELVO:\n1. Apruébase"])
        parsed = await OcrIndexCardParser(ocr=ocr).parse(_pdf(make_blank_pdf(3), **CARD))
        assert parsed.title == "Resolución N° 1266"
        # The index card leads (no page); empty pages are dropped, the others keep their number.
        assert [(s.text, s.page) for s in parsed.sections] == [
            ("Resolución N° 1266\n\nComplementa la individualización.", None),
            ("VISTOS:\nLo dispuesto", 1),
            ("RESUELVO:\n1. Apruébase", 3),
        ]
        assert parsed.collected["content_mode"] == ("ocr",)
        assert ocr.calls == 1

    async def test_without_a_title_the_ocr_text_is_titled_by_its_uri(self) -> None:
        parsed = await OcrIndexCardParser(ocr=FakeOcr(["text"])).parse(_pdf(make_blank_pdf(1)))
        assert parsed.title == "u"
        assert [s.text for s in parsed.sections] == ["text"]  # no title: no card section

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

    async def test_the_page_limit_reaches_the_ocr(self) -> None:
        ocr = FakeOcr(["text"])
        await OcrIndexCardParser(ParseLimits(max_pages=7), ocr).parse(_pdf(make_blank_pdf(1)))
        assert ocr.max_pages == 7

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

    @pytest.fixture(autouse=True)
    def _pdfium(self) -> None:
        pytest.importorskip("pypdfium2")

    def test_each_page_is_rendered_and_its_text_boxes_joined_by_lines(self) -> None:
        from PIL import Image

        images: list[Any] = []
        answers = iter([("line one", "line two"), None])

        def engine(image: Any) -> SimpleNamespace:
            images.append(image)
            return SimpleNamespace(txts=next(answers))

        created: list[int] = []

        def factory() -> Any:
            created.append(1)
            return engine

        pages = RapidOcrPages(factory)(make_pdf(["anything", ""]), 2)
        assert pages == ["line one\nline two", ""]
        # The engine gets the PIL image (it converts RGB to BGR itself; an array would be read
        # as BGR). 612 pt wide at 300 dpi (pdfium may round the height up a pixel).
        assert all(isinstance(image, Image.Image) and image.mode == "RGB" for image in images)
        assert [image.width for image in images] == [2550] * 2
        RapidOcrPages(factory)  # building alone creates no engine
        assert created == [1]

    def test_a_page_over_the_pixel_cap_is_rendered_smaller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cap = 100_000
        monkeypatch.setattr(ocr_module, "MAX_OCR_PIXELS", cap)
        sizes: list[tuple[int, int]] = []

        def engine(image: Any) -> SimpleNamespace:
            sizes.append(image.size)
            return SimpleNamespace(txts=None)

        RapidOcrPages(lambda: engine)(make_pdf([""]), 1)  # 612 x 792 pt: 8.4 Mpx at 300 dpi
        ((width, height),) = sizes
        assert width * height <= cap + width + height  # pdfium rounds up to a whole pixel
        assert width * height > cap * 0.95

    def test_more_pages_than_the_limit_are_refused_before_any_ocr(self) -> None:
        with pytest.raises(DocumentTooLargeError, match=r"3 pages for OCR \(limit 2\)"):
            RapidOcrPages(lambda: _unused_engine)(make_pdf(["a", "b", "c"]), 2)

    def test_a_broken_file_is_unparseable(self) -> None:
        with pytest.raises(UnparseableDocumentError, match="Cannot OCR PDF"):
            RapidOcrPages(lambda: _unused_engine)(b"%PDF-1.4 broken", 1)

    def test_a_page_that_fails_to_render_is_unparseable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pypdfium2 as pdfium

        def broken(*args: Any, **kwargs: Any) -> Any:
            raise pdfium.PdfiumError("Failed to render page")

        monkeypatch.setattr(pdfium.PdfPage, "render", broken)
        with pytest.raises(UnparseableDocumentError, match="Failed to render page"):
            RapidOcrPages(lambda: _unused_engine)(make_pdf(["a"]), 1)

    def test_an_engine_failure_is_unparseable(self) -> None:
        def engine(image: Any) -> Any:
            raise RuntimeError("onnxruntime: invalid input")

        with pytest.raises(UnparseableDocumentError, match="onnxruntime"):
            RapidOcrPages(lambda: engine)(make_pdf(["a"]), 1)

    def test_memory_errors_are_not_converted(self) -> None:
        def engine(image: Any) -> Any:
            raise MemoryError

        with pytest.raises(MemoryError):
            RapidOcrPages(lambda: engine)(make_pdf(["a"]), 1)


def _unused_engine(image: Any) -> Any:
    raise AssertionError("the engine must not run")
