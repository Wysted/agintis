"""Refusal detection for `refusal_correctness`: did the agent decline (or say it does not know)?

Two signals, deterministic and cheap:
1. The agent's own refuse path: scope verdict `out`, or any `refusal_reason` in the record (`social`
   turns report no scope): the `refuse` node answered with its fixed message (no retrieval, no
   synthesis), so it IS a refusal whatever the text says.
2. Otherwise a text heuristic over the answer: no `[n]` citation (the synthesis prompt forbids
   citing when it does not know), at most `MAX_REFUSAL_CHARS` characters (a long answer with a
   caveat sentence is an answer, not a blanket refusal) and a match of one of the
   `REFUSAL_PATTERNS` (Spanish "I don't know / I can't answer / it is not in the sources" forms,
   compared lowercase and without accents, with word boundaries; for a `citations: none` agent
   the match must start the answer, after an optional courtesy or "en las fuentes" lead-in,
   since a missing marker says nothing there).

Where the patterns come from, honestly: the synthesis prompt's contract ("say you do not know",
no citations on a refusal), the fixed out-of-scope message style of a case's agent ("solo puedo
responder ...", "no encontre nada relacionado ...") and the reviewer's list of false negatives
and positives of the first version. NOT from any evaluation set. They are narrow on purpose:
the reflexive "no se encuentra obligado" or a bare "no aparece un limite, el plazo es ..." are
answers, so "no se" only counts before a knowledge word ("no se cual", "no se si") and "no
aparece / no esta / no se encuentra" only before "en las fuentes", "disponible" or
"informacion". Known weak spots remain (a paraphrase nobody listed is a false negative; an
uncited answer that happens to say one of these phrases while answering is a false positive), so
`refusal_correctness` is the judge to audit first against labels (`cli.agreement`).

A REFUSAL, everywhere in this package, is an answer that gives NO information about the topic
asked ("no se", "the sources do not cover it"), even if it adds facts about another topic. A
partial answer (same-topic content or a pointer to where it is) is not a refusal: its claims are
judged. The text heuristic below cannot tell the topic of a contrastive continuation, so it keeps
reading "no se X, pero ..." as an answer. Known limit: ANY adjacent-topic answer WITH citations is
a false negative of the heuristic (the CITATION gate), and so is an uncited adjacent-topic "pero";
hybrid mode covers them through the model flag, and the scope path covers the agent's own refusals.
The LLM judges (faithfulness,
answer_relevance) do NOT use the text heuristic to decide whether they apply: they gate on the
scope verdict `out` only and let the judge model report a refusal (`is_refusal`); the aggregates
count how often model and heuristic disagree (`refusal_signals`) so the two can be audited
against each other.
"""

import re
import unicodedata

from agintis.agent_kit.judge.citations import citation_numbers
from agintis.agent_kit.judge.guard import strip_markers
from agintis.agent_kit.judge.records import AnswerRecord
from agintis.agent_kit.source_text import CitationStyle

MAX_REFUSAL_CHARS = 600
# Nouns for "the sources": deliberately NOT `normativa` (legal text says "la normativa no permite
# retiros", which is an answer).
_SOURCES_NOUN = r"(?:fuentes?|documentos?|textos?|informacion|material)"
_ADJECTIVES = r"(?: (?:proporcionad|disponible|consultad|entregad|recuperad|revisad|adjunt)[a-z]*)*"
_COVER_VERBS = (
    r"(?:contienen?|incluyen?|mencionan?|abordan?|tratan?|cubren?|dicen?|indican?|"
    r"especifican?|aclaran?|precisan?|entregan?|responden?|permiten? responder|tienen)"
)
# What may follow such a verb for the sentence to be "it does not say": the missing thing itself
# ("no contienen esa informacion"), a demonstrative pronoun ("no abordan eso") or the end of the
# clause ("no abordan."). NOT `la` / `las` / `lo` after the verb: they are mostly articles ("el
# texto no indica la fecha exacta, pero el plazo es 30 dias" is an answer). A verb followed by a
# specific object ("el texto no indica un plazo distinto") is an answer about the content, not a
# statement that the sources are silent. A pronoun BEFORE the verb is fine ("las fuentes no lo
# dicen"): it cannot be an article there.
_MISSING_TAIL = (
    r"(?:(?: \w+)? (?:informacion|respuesta|tema|consulta|pregunta|datos?|detalles?|antecedentes)"
    r"|\s*(?:[.,;:!?]|$)|\s+(?:eso|esto|ello)\b)"
)
# A contrastive continuation ("..., pero el fondo A...") normally asserts something: the answer
# is partial, not a refusal. A `; el ...` continuation is NOT a contrast ("no encontre X en los
# documentos; el Codigo del Trabajo no esta entre ellos" is a refusal with an explanation).
# Unless the contrast only offers help ("no lo se, pero puedes reformular la pregunta"). Known
# limit: a contrast that asserts nothing ("no lo se, pero me gustaria ayudarte") is read as an
# answer: no regex tells facts from filler.
_CONTRAST = re.compile(r"\b(?:pero|sin embargo|no obstante|aunque|salvo que)\b")
# Offers of help directed AT THE USER only: a bare "puede" or "consulta" is as likely to be a fact
# ("el afiliado puede solicitar el retiro en su AFP") as an offer.
_OFFER = re.compile(
    r"\b(?:puedes|podrias|te (?:recomiendo|sugiero)|intenta|reformul\w*|"
    r"consult(?:ar|a) (?:a|en|con)|contacta\w*|ayudarte)\b"
)
REFUSAL_PATTERNS: tuple[str, ...] = (
    r"\bno lo se\b",
    r"\bno se (?:la|el|los|las|cual|cuales|que|como|cuando|cuanto|cuantos|cuantas|donde|quien|si"
    r"|nada|de)\b",
    r"\bdesconozco\b",
    r"\bno sabria\b",
    r"\bno tengo (?:la |mas |suficientes? )?(?:informacion|respuesta|datos|antecedentes)\b",
    r"\bno (?:cuento|dispongo) (?:con|de) (?:la |esa |dicha |suficiente )?"
    r"(?:informacion|datos|antecedentes)\b",
    r"\bno se dispone de (?:informacion|datos|antecedentes)\b",
    r"\bno hay (?:informacion|datos|nada|referencias?|menciones?|antecedentes)\b",
    r"\bno (?:encontre|he encontrado)\b",
    r"\bno encuentro (?:esa|esta|dicha|la|ninguna|informacion|nada|datos|referencias?)\b",
    r"\bno se (?:encontro|encuentra|encuentran|menciona|mencionan|indica|especifica) "
    rf"(?:informacion|nada|ninguna|en (?:las|los|el|la|esas?|estos?) {_SOURCES_NOUN}|disponible)",
    rf"\bno (?:esta|estan|aparece|aparecen|figura|figuran) "
    rf"(?:disponibles?|en (?:las|los|el|la|esas?|estos?) {_SOURCES_NOUN}|informacion)",
    rf"\b(?:las|los|la|el) {_SOURCES_NOUN}{_ADJECTIVES} no (?:lo |la )?{_COVER_VERBS}"
    rf"{_MISSING_TAIL}",
    r"\bno (?:puedo|podria) (?:responder|ayudar|contestar|confirmar\w*|afirmar\w*|verificar\w*)\b",
    r"\bno es posible responder\b",
    r"\bsolo puedo responder\b",
    r"\bfuera del alcance\b",
)
_COMPILED = tuple(re.compile(p) for p in REFUSAL_PATTERNS)
# What may precede the anchored refusal of a `citations: none` agent: a courtesy ("lo siento,
# pero", "lamentablemente") and/or a place ("en las fuentes proporcionadas").
# shortcut: closed list, any other preamble ("Gracias por tu pregunta. No lo se.") is a false
# negative; widen it when `cli.agreement` shows misses on uncited agents.
_LEAD_IN = re.compile(
    r"(?:(?:lo siento|lamentablemente|desafortunadamente|disculpa|disculpe|perdon|me temo que)"
    r"(?:,? pero)?[,.:;!]?\s*)?"
    rf"(?:(?:en|segun) (?:las|los|el|la|esas?|estos?) {_SOURCES_NOUN}{_ADJECTIVES},?\s*)?"
)
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, accents removed, whitespace collapsed."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _SPACES.sub(" ", stripped).strip()


def _reads_as_missing(normalized: str) -> bool:
    """A refusal pattern matches and no contrastive continuation asserts something after it."""
    matches = [m for pattern in _COMPILED if (m := pattern.search(normalized))]
    if not matches:
        return False
    after = normalized[min(m.start() for m in matches) :]
    contrast = _CONTRAST.search(after)
    return contrast is None or _OFFER.search(after[contrast.start() :]) is not None


def looks_like_refusal(answer: str, *, citations: CitationStyle) -> bool:
    """The text heuristic alone (signal 2). With `citations: none` no answer carries markers, so
    their absence says nothing: a refusal pattern must then START the answer, after an optional
    `_LEAD_IN` ("Se renueva en el mostrador; no hay informacion sobre el costo" is a partial
    answer, "Lo siento, no tengo informacion" a refusal)."""
    # shortcut: the text cannot tell an adjacent-topic cited answer from a partial one, so any
    # cited answer is "not a refusal"; deterministic-only runs undercount refusals, rely on hybrid
    # or add a topic check before using deterministic mode for decisions.
    if not answer.strip() or len(answer) > MAX_REFUSAL_CHARS or citation_numbers(answer):
        return False
    normalized = normalize(answer)
    if citations == "none":
        lead_in = _LEAD_IN.match(normalized)
        assert lead_in is not None  # every group is optional
        if not any(p.match(normalized, lead_in.end()) for p in _COMPILED):
            return False
    return _reads_as_missing(normalized)


def states_missing_information(sentence: str) -> bool:
    """Does a (cited) sentence read as "the sources do not say it"? The same patterns as
    `looks_like_refusal`, without its length and no-citation gates: the `[n]` markers are
    stripped first. Backs the judge model's `states_missing_info` flag, which is honoured only
    when this agrees."""
    return _reads_as_missing(normalize(strip_markers(sentence)))


def parameters() -> dict[str, object]:
    """Everything that decides what the heuristic accepts (hashed into the report meta)."""
    return {
        "patterns": list(REFUSAL_PATTERNS),
        "max_chars": MAX_REFUSAL_CHARS,
        "contrast": _CONTRAST.pattern,
        "offer": _OFFER.pattern,
        "lead_in": _LEAD_IN.pattern,
    }


def refused_by_scope(record: AnswerRecord) -> bool:
    """The agent's own refuse path ran: scope verdict `out` or any `refusal_reason` (a `social`
    turn announces no scope, so an in-domain question misrouted there counts as refused)."""
    return record.refusal_reason is not None or (
        record.scope is not None and record.scope.level == "out"
    )


def is_refusal(record: AnswerRecord) -> bool:
    """The agent refused (scope `out`) or said it does not know (text heuristic)."""
    return refused_by_scope(record) or looks_like_refusal(record.answer, citations=record.citations)
