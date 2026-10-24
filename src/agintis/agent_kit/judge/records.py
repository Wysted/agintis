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
 "is_clear": true, "confidence": 0.8, "citations": "inline",
 "usage": {"gpt-5.4-mini": {"input_tokens": 900, "output_tokens": 120, "total_tokens": 1020,
                            "cached_input_tokens": 0, "calls": 3}},
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
`is_clear` and `confidence` (optional, absent in old dumps) are the analyzer's clarity verdict and
confidence, logged for calibrating a future clarification policy; not acted on, not part of
`answer_sha256`.
`usage` (optional, absent in old dumps) is the LLM token usage of every call the run made for the
row (analyzer, rewrite, merge tie-break, synthesis, structured calls), per configured model name:
`{"<model>": {"input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "calls"}}`
(`cached_input_tokens` is part of `input_tokens`), null when no call ran; a failed row keeps what
was consumed before the error. Not part of `answer_sha256`.
`citations` (`inline` | `none`; absent in old dumps = `inline`) is the agent's citation style: with
`none` the answer carries no `[n]` by design.
`answer_sha256` binds labels to what was judged: the sha256 of the canonical JSON of the question,
the expected scope, the answer and the raw source fields (not of any rendered text); a labeler
copies it into its rows and `cli.agreement` drops rows whose hash no longer matches. It is
recomputed on load, so an edited record is rejected.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from agintis.agent_kit import source_text
from agintis.agent_kit.model_usage import ModelUsage
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


def total_usage(rows: Iterable[Mapping[str, ModelUsage] | None]) -> dict[str, ModelUsage]:
    """The per-model sum of the `usage` of dump records (rows without one add nothing)."""
    totals: dict[str, ModelUsage] = {}
    for row in rows:
        for model, usage in (row or {}).items():
            totals[model] = totals.get(model, ModelUsage()) + usage
    return totals


def format_usage(totals: Mapping[str, ModelUsage]) -> str:
    """One summary line, e.g. `gpt-x: 10 in / 5 out / 15 total (2 cached) over 3 calls`."""
    if not totals:
        return "tokens: none reported"
    return "tokens: " + "; ".join(
        f"{model}: {u.input_tokens} in / {u.output_tokens} out / {u.total_tokens} total "
        f"({u.cached_input_tokens} cached) over {u.calls} calls"
        for model, u in sorted(totals.items())
    )


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
    is_clear: bool | None = None  # the analyzer's clarity verdict (None: not reported)
    confidence: float | None = None  # the analyzer's confidence in [0, 1] (None: not reported)
    usage: Mapping[str, ModelUsage] | None = None  # LLM tokens per model (None: no call reported)
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
            "is_clear": self.is_clear,
            "confidence": self.confidence,
            "usage": None
            if self.usage is None
            else {model: u.to_json() for model, u in self.usage.items()},
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


def _flag(row: Mapping[str, Any], key: str) -> bool | None:
    value = row.get(key)
    if value is not None and not isinstance(value, bool):
        raise RecordFormatError(f"{key!r} must be a boolean or null")
    return value


def _unit(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        raise RecordFormatError(f"{key!r} must be a number in [0, 1] or null")
    return float(value)


_USAGE_FIELDS = tuple(ModelUsage().to_json())


def _usage(raw: object) -> dict[str, ModelUsage] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RecordFormatError("'usage' must be an object or null")
    usage: dict[str, ModelUsage] = {}
    for model, counts in raw.items():
        if not isinstance(model, str) or not isinstance(counts, dict):
            raise RecordFormatError("'usage' must map model names to token counts")
        # unknown extra keys are ignored: a later version may add counters
        values = [counts.get(k) for k in _USAGE_FIELDS]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise RecordFormatError(
                f"usage of {model!r} must have non-negative integers for {_USAGE_FIELDS}"
            )
        usage[model] = ModelUsage(**{k: counts[k] for k in _USAGE_FIELDS})
    return usage


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
        is_clear=_flag(row, "is_clear"),
        confidence=_unit(row, "confidence"),
        usage=_usage(row.get("usage")),
        citations=citations,
    )
    if row.get("answer_sha256") != record.answer_sha256:
        raise RecordFormatError("'answer_sha256' does not match the answer and sources")
    return record
