"""How a numbered source is shown to a model: ONE pure rendering shared by the synthesis and the
judges (no langchain, no langgraph, no platform: only strings).

`[n] title (locator)` + the passage text, markup neutralized. The judges render the sources of a
dump with these very functions so a judge sees what the generator saw; the dump's own integrity
hash does NOT depend on them (see `judge.records`), so a rendering change never invalidates labels.
"""

import unicodedata
from collections.abc import Mapping, Sequence
from itertools import groupby
from typing import Any, Literal

NO_SOURCES_TEXT = "(no sources were found for this question)"

# How an answer refers to its sources: `inline` = `[n]` markers in the text, `none` = no markers
# (the answer stays grounded in the sources; the Sources event still lists them).
type CitationStyle = Literal["inline", "none"]


def neutralize(text: str) -> str:
    """No `<` / `>` survives, so source text can never close or reopen a delimiting block."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def one_line(text: str) -> str:
    return neutralize(" ".join(text.split()))


def render_source(number: int, item: Mapping[str, Any]) -> str:
    """`[n] title (locator)` + text of ONE source (n = the number the answer cites).

    `item` has `source` (uri), `title`, `locator` and `text` (a `SourceItem`). Text, title and
    locator are untrusted: markup characters are neutralized and title/locator collapsed to one
    line.
    """
    label = one_line(item["title"] or item["source"])
    if item["locator"]:
        label += f" ({one_line(item['locator'])})"
    return f"[{number}] {label}\n{neutralize(item['text'])}"


def render_sources(items: Sequence[Mapping[str, Any]]) -> str:
    """The sources in order, numbered from 1; expects the output of `select_sources`."""
    blocks = [render_source(number, item) for number, item in enumerate(items, start=1)]
    return "\n\n".join(blocks) if blocks else NO_SOURCES_TEXT


# Letters whose Unicode name is not LATIN... but that Spanish text uses (ordinal indicators, micro).
_LATIN_LIKE = frozenset("\u00aa\u00ba\u00b5")
# Only these categories can flag a word (Lm modifier letters such as U+02BC and every mark stay
# part of the token without flagging it: emoji variation selectors and keycaps are marks).
_FLAGGING_CATEGORIES = ("Lu", "Ll", "Lt", "Lo")
# Letterlike Symbols (script l, info sign, double-struck N...): symbols that are also letters.
_EXCLUDED_RANGES = ((0x2100, 0x214F),)
_LATIN_NAME_PREFIXES = ("LATIN", "FULLWIDTH LATIN")


def script_parameters() -> dict[str, object]:
    """Everything that decides what `foreign_script_tokens` flags (hashed into the report meta)."""
    return {
        "categories": list(_FLAGGING_CATEGORIES),
        "excluded_ranges": [list(r) for r in _EXCLUDED_RANGES],
        "latin_like": sorted(_LATIN_LIKE),
        "latin_name_prefixes": list(_LATIN_NAME_PREFIXES),
    }


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


def _is_foreign_script(ch: str) -> bool:
    if ch in _LATIN_LIKE or unicodedata.category(ch) not in _FLAGGING_CATEGORIES:
        return False
    if any(lo <= ord(ch) <= hi for lo, hi in _EXCLUDED_RANGES):
        return False
    return not unicodedata.name(ch, "").startswith(_LATIN_NAME_PREFIXES)


# shortcut: assumes a Latin-script target language (a Greek/Russian answer would flag every word),
# make the allowed script a parameter when a non-Latin case needs this check.
def foreign_script_tokens(text: str, context: str) -> list[str]:
    """Words of `text` with a letter outside the Latin script that `context` (the question
    and the sources) does not contain: a model slipping a devanagari or bengali word into Spanish
    text. A character that the context itself uses (a Greek beta quoted from a source) is not
    flagged. Both texts are NFC-normalized so a decomposed accent is not a stray combining mark.
    Emoji, other symbols and Letterlike Symbols never flag. Order of appearance, no repeats."""
    known = set(unicodedata.normalize("NFC", context))
    found: dict[str, None] = {}
    for is_word, run in groupby(unicodedata.normalize("NFC", text), _is_word_char):
        token = "".join(run)
        if is_word and any(_is_foreign_script(c) and c not in known for c in token):
            found[token] = None
    return list(found)
