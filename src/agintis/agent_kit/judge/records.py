"""The answer dump: one JSON Lines record per question, the input of the offline judges.

The dump is the contract between the run that produced the answers (`cli.dump_answers`) and the
judges (`cli.judge`), so a judging run is reproducible and cheap to repeat: it needs no database,
no retrieval and not the generator model. A record carries everything a judge may look at: the
question, the answer, the numbered sources the synthesis saw (with the chunk TEXT, numbered like
the `[n]` markers of the answer), the scope verdict of the clarification policy and the models.

```json
{"schema": 1, "id": "in-cmp-01", "question": "...", "expected_scope": "in",
 "answer": "... [1] ...", "answer_sha256": "...", "scope": {"level": "in", "score": 0.7312},
 "refusal_reason": null, "standalone_query": "...", "query": "...", "attempts": 0,
 "citations": "inline",
 "sources": [{"n": 1, "title": "...", "locator": "p. 3", "uri": "...", "text": "..."}],
 "agent": "spensiones-rag", "models": {"analyzer": "default (openai:gpt-5.4-mini)", ...},
 "meta": {"git": "abc1234", "tenant": "default"}, "error": null}
```

`expected_scope` (`in` | `out` | null) is the reference label of the question when its file has
one. `error` is the failure class of a run that did not produce an answer (`answer` is then "").
`scope.score` is null when the policy refused for lack of any evidence or the request was off-domain
(`other`). `refusal_reason` (`out_of_scope` | `off_domain` | `social` | null; optional in old
dumps) is why the agent's refuse path ran: a `social` turn announces no scope, so this is the only
trace of it. `standalone_query` (optional, absent in old dumps) is the query the analyze node
condensed from the history and the question: the way to see what a follow-up row was understood
as; it is not part of `answer_sha256`. `query` (optional, absent in old dumps) is the text the
LAST retrieval ran on (the standalone query, or the last rewrite of it; a refused turn, see
`refusal_reason`, keeps the standalone query but retrieved nothing) and `attempts` (optional,
absent in old dumps) is how many rewrites ran this turn; neither is part of `answer_sha256`.
`citations` (`inline` | `none`; absent in old dumps = `inline`) is the agent's citation style: with
`none` the answer carries no `[n]` by design.
`answer_sha256` binds labels to what was judged: the sha256 of the canonical JSON of the question,
the expected scope, the answer and the raw source fields (not of any rendered text); a labeler
copies it into its rows and `cli.agreement` drops rows whose hash no longer matches. It is
recomputed on load, so an edited record is rejected.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from agintis.agent_kit import source_text
from agintis.agent_kit.source_text import CitationStyle
from agintis.modules.agents_catalog.api import ScopeLevel

SCHEMA_VERSION = 1
ExpectedScope = Literal["in", "out"]
EXPECTED_SCOPES: tuple[str, ...] = get_args(ExpectedScope)
SCOPE_LEVELS: tuple[str, ...] = get_args(ScopeLevel)
CITATION_STYLES: tuple[str, ...] = get_args(CitationStyle.__value__)


class RecordFormatError(ValueError):
    """A dump line does not follow the schema."""


@dataclass(frozen=True, slots=True)
class CitedSource:
    """The n-th source of the answer: `[n]` in the answer names it."""

    n: int
    title: str | None
    uri: str
    text: str
    locator: str | None = None

    def item(self) -> dict[str, Any]:
        """The source as a `SourceItem`-shaped mapping (what `source_text` renders)."""
        return {"source": self.uri, "title": self.title, "locator": self.locator, "text": self.text}

    def render(self) -> str:
        """Exactly the block `SynthesizeNode` showed the model for this source (display only)."""
        return source_text.render_source(self.n, self.item())


def _source_json(source: CitedSource) -> dict[str, Any]:
    """The raw fields of a source: what the integrity hash and the dump line both carry."""
    return {
        "n": source.n,
        "title": source.title,
        "locator": source.locator,
        "uri": source.uri,
        "text": source.text,
    }


@dataclass(frozen=True, slots=True)
class ScopeVerdict:
    level: ScopeLevel
    score: float | None  # None: refused for lack of any evidence, or off-domain


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    id: str
    question: str
    answer: str
    agent: str
    expected_scope: ExpectedScope | None = None
    scope: ScopeVerdict | None = None
    sources: tuple[CitedSource, ...] = ()
    models: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    refusal_reason: str | None = None  # why the refuse path ran (None = a normal answer)
    standalone_query: str | None = None  # the analyze node's condensed query (None: not run)
    query: str | None = None  # the text the last retrieval ran on (None: not run)
    attempts: int | None = None  # rewrites that ran this turn (None: not reported)
    # the agent's `citations` (`none` = no [n] by design); no default: only `record_from_json`
    # reads a missing one (old dumps) as `inline`
    citations: CitationStyle = field(kw_only=True)

    @property
    def rendered_sources(self) -> str:
        """The sources block as the synthesis rendered it (display only, never hashed)."""
        return source_text.render_sources([s.item() for s in self.sources])

    @property
    def answer_sha256(self) -> str:
        """sha256 of the canonical JSON of everything a label depends on: the question, the
        expected scope (an audit relabel of a question out -> in changes the right label), the
        answer and the raw source fields. Independent of any rendering code, so a change in how
        sources are displayed never invalidates labels."""
        canonical = json.dumps(
            {
                "question": self.question,
                "expected_scope": self.expected_scope,
                "answer": self.answer,
                "sources": [_source_json(s) for s in self.sources],
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        # surrogatepass: a lone surrogate in model output must not crash the hash
        return hashlib.sha256(canonical.encode("utf-8", "surrogatepass")).hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "id": self.id,
            "question": self.question,
            "expected_scope": self.expected_scope,
            "answer": self.answer,
            "answer_sha256": self.answer_sha256,
            "scope": None
            if self.scope is None
            else {"level": self.scope.level, "score": self.scope.score},
            "sources": [_source_json(s) for s in self.sources],
            "agent": self.agent,
            "models": dict(self.models),
            "meta": dict(self.meta),
            "error": self.error,
            "refusal_reason": self.refusal_reason,
            "standalone_query": self.standalone_query,
            "query": self.query,
            "attempts": self.attempts,
            "citations": self.citations,
        }


def _text(row: Mapping[str, Any], key: str, *, nullable: bool = False) -> Any:
    value = row.get(key)
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise RecordFormatError(f"{key!r} must be a string")
    return value


def _count(row: Mapping[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordFormatError(f"{key!r} must be an integer or null")
    return value


def _source(position: int, raw: object) -> CitedSource:
    if not isinstance(raw, dict):
        raise RecordFormatError("a source must be a JSON object")
    n = raw.get("n")
    if isinstance(n, bool) or n != position:
        raise RecordFormatError(f"sources must be numbered 1..N in order (got {n!r} at {position})")
    return CitedSource(
        n=position,
        title=_text(raw, "title", nullable=True),
        uri=_text(raw, "uri"),
        text=_text(raw, "text"),
        locator=_text(raw, "locator", nullable=True),
    )


def _scope(raw: object) -> ScopeVerdict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RecordFormatError("'scope' must be an object or null")
    level, score = raw.get("level"), raw.get("score")
    if level not in SCOPE_LEVELS:
        raise RecordFormatError(f"scope level must be one of {SCOPE_LEVELS}")
    if score is not None and (isinstance(score, bool) or not isinstance(score, int | float)):
        raise RecordFormatError("scope score must be a number or null")
    return ScopeVerdict(level, None if score is None else float(score))


def record_from_json(row: object) -> AnswerRecord:
    """Parse and validate one dump line (`RecordFormatError` names the problem)."""
    if not isinstance(row, dict):
        raise RecordFormatError("a record must be a JSON object")
    if row.get("schema") != SCHEMA_VERSION:
        raise RecordFormatError(f"unsupported schema {row.get('schema')!r}")
    expected = row.get("expected_scope")
    if expected is not None and expected not in EXPECTED_SCOPES:
        raise RecordFormatError(f"expected_scope must be one of {EXPECTED_SCOPES} or null")
    models, meta = row.get("models"), row.get("meta")
    if not isinstance(models, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in models.items()
    ):
        raise RecordFormatError("'models' must map role names to labels")
    if not isinstance(meta, dict):
        raise RecordFormatError("'meta' must be an object")
    sources = row.get("sources")
    if not isinstance(sources, list):
        raise RecordFormatError("'sources' must be a list")
    citations = row.get("citations", "inline")
    if citations not in CITATION_STYLES:
        raise RecordFormatError(f"citations must be one of {CITATION_STYLES}")
    record = AnswerRecord(
        id=_text(row, "id"),
        question=_text(row, "question"),
        answer=_text(row, "answer"),
        agent=_text(row, "agent"),
        expected_scope=expected,
        scope=_scope(row.get("scope")),
        sources=tuple(_source(i, s) for i, s in enumerate(sources, start=1)),
        models=dict(models),
        meta=dict(meta),
        error=_text(row, "error", nullable=True),
        refusal_reason=_text(row, "refusal_reason", nullable=True),
        standalone_query=_text(row, "standalone_query", nullable=True),
        query=_text(row, "query", nullable=True),
        attempts=_count(row, "attempts"),
        citations=citations,
    )
    if row.get("answer_sha256") != record.answer_sha256:
        raise RecordFormatError("'answer_sha256' does not match the answer and sources")
    return record
