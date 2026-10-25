"""`foreign_script_tokens`: a non-Latin word the question and the sources do not contain."""

import pytest

from agintis.agent_kit.source_text import foreign_script_tokens, script_parameters


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("El plazo es नमस्ते de 24 meses", ["नमस्ते"]),  # a whole devanagari word
        ("El plazo es de 24 mesesদিন.", ["mesesদিন"]),  # a bengali glyph glued to a word
        ("tiene వ్యక్తి derecho y వ్యక్తి", ["వ్యక్తి"]),  # no repeats
    ],
)
def test_a_foreign_script_word_is_flagged(text: str, expected: list[str]) -> None:
    assert foreign_script_tokens(text, "¿Cuál es el plazo?") == expected


def test_a_letter_the_context_uses_is_not_flagged() -> None:
    assert foreign_script_tokens("El parámetro β vale 2", "la tasa β del fondo") == []
    assert foreign_script_tokens("El parámetro β vale 2", "sin griego") == ["β"]


@pytest.mark.parametrize(
    "text",
    [
        "¿Qué plazo tiene el niño? ¡Año nuevo! Está en la 3ª y 2º cláusula, 5 µg.",
        "Resolución Nº 12 del Manual — «cotización» (art. 1°)",
        "A decomposed a\u0301 and n\u0303 are Spanish letters",
        "Atención \u26a0\ufe0f",
        "Sí \u2714\ufe0f",
        "\u2139\ufe0f Nota",
        "1\ufe0f\u20e3 paso",
        "O\u02bc\u200bHiggins y D\u02bcAlessandro",  # modifier letter U+02BC
        "el valor \u2113 y \u2115 del fondo",  # Letterlike Symbols
        "\uff21\uff22\uff23 \uff41\uff42\uff43 \uff11\uff12",  # fullwidth Latin letters, digits
        "",
    ],
)
def test_plain_spanish_is_not_flagged(text: str) -> None:
    assert foreign_script_tokens(text, "") == []


def test_script_parameters_describe_the_rule() -> None:
    params = script_parameters()

    assert params["categories"] == ["Lu", "Ll", "Lt", "Lo"]
    assert params["excluded_ranges"] == [[0x2100, 0x214F]]
