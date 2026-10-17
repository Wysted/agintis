"""Each Spanish doc must carry the same fenced code blocks as its English original."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [
    ("README.md", "README.es.md"),
    ("docs/configuration.md", "docs/configuration.es.md"),
    ("case/fastapi_es/README.md", "case/fastapi_es/README.es.md"),
]
FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.MULTILINE | re.DOTALL)


def _blocks(path: str) -> list[str]:
    return [m.group(0) for m in FENCE.finditer((ROOT / path).read_text(encoding="utf-8"))]


def test_every_spanish_doc_has_an_english_original() -> None:
    translated = {
        p.relative_to(ROOT).as_posix() for p in ROOT.glob("**/*.es.md") if ".venv" not in p.parts
    }
    assert translated == {es for _, es in PAIRS}, "update PAIRS to cover every *.es.md"
    for en, _ in PAIRS:
        assert (ROOT / en).is_file(), f"missing English original {en}"


@pytest.mark.parametrize(("en", "es"), PAIRS, ids=[es for _, es in PAIRS])
def test_fenced_code_blocks_match(en: str, es: str) -> None:
    english, spanish = _blocks(en), _blocks(es)
    assert len(english) == len(spanish), f"{es}: {len(spanish)} blocks, {en} has {len(english)}"
    for i, (a, b) in enumerate(zip(english, spanish, strict=True)):
        assert a == b, f"{es}: block {i} differs\n--- {en}\n{a}\n--- {es}\n{b}"
