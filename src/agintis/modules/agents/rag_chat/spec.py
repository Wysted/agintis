"""AgentSpec: the configuration of one generic RAG chat agent (pure data, strict).

A case describes its agents in YAML (`case/<name>/agents/*.yaml`); the code of the agent is
generic and lives next to this file. Unknown keys are errors everywhere, like profiles.

`topics` is a non-empty list (unique names, unique profiles): with one topic the router is not
called (`router: auto`), with several the ROUTER model picks up to `max_topics` per question.
"""

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agintis.agent_kit.nodes import (
    DEFAULT_EVIDENCE_AGGREGATE,
    DEFAULT_EVIDENCE_MARGIN,
    DEFAULT_MAX_REWRITES,
    DEFAULT_MAX_TOPICS,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_MIN_SOURCES,
    DEFAULT_OUT_OF_SCOPE_MESSAGE,
    DEFAULT_PRIMARY_WEIGHT,
    DEFAULT_SECONDARY_MIN_CONFIDENCE,
    DEFAULT_SOCIAL_MESSAGE,
    DEFAULT_SUFFICIENCY,
    DEFAULT_SYNTHESIZE_PROMPTS,
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_MESSAGES,
    MAX_SOURCE_CHARS,
    MAX_SOURCES,
    CitationStyle,
    EvidenceAggregate,
    PromptSet,
    SourceLimits,
    SufficiencyKind,
    build_filter_schema,
    plan_filter_fields,
)
from agintis.modules.agents_catalog.api import AGENT_ID_PATTERN
from agintis.modules.knowledge.api import (
    DEFAULT_PROBE_K,
    NO_RERANKER_KEY,
    RagProfileRegistry,
)
from agintis.platform.llm import ChatRole
from agintis.shared_kernel.errors import NotFoundError

logger = logging.getLogger(__name__)
# A prompt that writes "[1]" or "[n]" is probably asking for citation markers.
_CITATION_MARKER = re.compile(r"\[(?:\d+|n)\]")

# Upper bounds that keep a spec from asking for an unbounded prompt.
MAX_SOURCES_LIMIT = 20
MAX_SOURCE_CHARS_LIMIT = 8000
MAX_CONTEXT_CHARS_LIMIT = 100_000
MAX_HISTORY_LIMIT = 50
MAX_TOPICS_LIMIT = 8
MAX_AGENT_TOPICS = 32
MAX_REWRITES_LIMIT = 3
MAX_PRIMARY_WEIGHT = 5
MAX_PROBE_K = 20
# Largest per-topic evidence bias (cosine units): a correction, never a way to force a topic.
MAX_EVIDENCE_BIAS = 0.1
MAX_TEXT_CHARS = 8000


class InvalidAgentSpecError(ValueError):
    """An agent spec file is malformed or inconsistent with the profiles (boot fails)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PromptsSpec(_Strict):
    """Per-node system prompt overrides; a missing one keeps the generic English default."""

    analyze: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)
    extract_filters: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)
    synthesize: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)
    route: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)
    rewrite: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)

    def to_prompt_set(self, *, citations: CitationStyle) -> PromptSet:
        """The prompts of the nodes; the default synthesize prompt follows `citations`."""
        defaults = {"synthesize": DEFAULT_SYNTHESIZE_PROMPTS[citations]}
        return PromptSet(**(defaults | self.model_dump(exclude_none=True)))


class TopicSpec(_Strict):
    name: str = Field(pattern=AGENT_ID_PATTERN)  # same shape as an id
    description: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    # Name of the RagProfile this topic retrieves from.
    profile: str = Field(min_length=1)
    # Metadata field name -> hint for the filter extractor. Doubles as the allowlist of the
    # fields it may fill; each must be a filterable field of the profile. Absent = every
    # filterable field (no hints); empty = no filtering (the extractor is never called).
    filters: Mapping[str, str] | None = None
    # Evidence router only: added to this topic's score before ranking, to correct a measured
    # one-sided offset (bigger profiles and longer chunks score higher under `max`). Calibrate it
    # on a DEV set, NEVER on the eval questions (held-out); 0 until there is such a set.
    evidence_bias: float = Field(default=0.0, ge=-MAX_EVIDENCE_BIAS, le=MAX_EVIDENCE_BIAS)


type ClarificationKind = Literal["none", "evidence"]


class ClarificationSpec(_Strict):
    """The scope check before retrieval (`ClarificationPolicy`, 3.5c).

    `none` (default) checks nothing. `evidence` needs `router: evidence` and reads the best raw
    similarity the router's probe finds in ANY topic (see `agent_kit/nodes/clarification.py`):
    above `low_confidence_below` the question is answered normally; in
    `(out_of_scope_below, low_confidence_below]` it is answered and flagged `low`; at or below
    `out_of_scope_below` (or with no evidence at all) it is refused with `out_of_scope_message`,
    with no retrieval and no synthesis call (the analyzer still runs). Both thresholds are
    cosine similarities of ONE corpus snapshot and embedding model (`cli.calibrate_evidence`):
    recalibrate after a re-ingest.

    The analyzer's `request_kind` is acted on by the same policy: small talk gets `social_message`
    (a courtesy reply that states the scope) and any other non-domain request gets
    `out_of_scope_message`, both with no retrieval. `low_scope_max_topics` (default: the agent's
    `max_topics`) is how many topics are searched in the `low` band, where routing is least
    reliable; above `max_topics` it takes the best topics whatever `evidence_margin` says.
    """

    policy: ClarificationKind = "none"
    low_confidence_below: float | None = Field(default=None, ge=0, le=1)
    out_of_scope_below: float | None = Field(default=None, ge=0, le=1)
    out_of_scope_message: str = Field(
        default=DEFAULT_OUT_OF_SCOPE_MESSAGE, min_length=1, max_length=MAX_TEXT_CHARS
    )
    social_message: str = Field(
        default=DEFAULT_SOCIAL_MESSAGE, min_length=1, max_length=MAX_TEXT_CHARS
    )
    low_scope_max_topics: int | None = Field(default=None, ge=1, le=MAX_TOPICS_LIMIT)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        given = self.model_fields_set - {"policy"}
        if self.policy == "none":
            if given:
                logger.warning(
                    "clarification: %s ignored (they only apply to `policy: evidence`)",
                    ", ".join(sorted(given)),
                )
            return self
        low, out = self.low_confidence_below, self.out_of_scope_below
        if low is None or out is None:
            raise ValueError(
                "policy 'evidence' needs low_confidence_below and out_of_scope_below "
                "(calibrate them with cli.calibrate_evidence)"
            )
        if not out < low:
            raise ValueError(
                "thresholds need 0 <= out_of_scope_below < low_confidence_below <= 1 "
                f"(got {out} and {low})"
            )
        if not self.out_of_scope_message.strip() or not self.social_message.strip():
            raise ValueError(
                "policy 'evidence' needs a non-blank out_of_scope_message and social_message"
            )
        return self


type RouterKind = Literal["auto", "single", "llm", "evidence"]


class AgentSpec(_Strict):
    id: str = Field(pattern=AGENT_ID_PATTERN)
    description: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    topics: list[TopicSpec] = Field(min_length=1, max_length=MAX_AGENT_TOPICS)
    prompts: PromptsSpec = PromptsSpec()
    # How the answer refers to its sources: `inline` = `[n]` markers, `none` = no markers (still
    # grounded in the sources; the Sources event is emitted either way). Picks the default
    # synthesize prompt; a `prompts.synthesize` override replaces it in both styles.
    citations: CitationStyle = "inline"
    # Routing: `auto` = single when there is one topic, else the LLM router; `single` needs
    # exactly one topic; `llm` asks the ROUTER model; `evidence` asks the corpus (no model: a
    # dense probe of every topic's profile with ONE query embedding). `llm` and `evidence` need
    # at least two topics.
    router: RouterKind = "auto"
    # LLM profile per ChatRole (`models: {analyzer: fast, synthesis: smart}`): names of the
    # platform's profile catalog (LLM_PROFILES_FILE; `default` = the LLM_* defaults). A role left
    # out uses the platform default; nodes never see the provider. Checked at boot (wiring).
    models: Mapping[ChatRole, Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=dict
    )
    # Topics searched in parallel per question (cap on the Send fan-out). The primary topic is
    # always searched; an extra one only when the router's confidence in it reaches
    # `secondary_min_confidence`.
    max_topics: int = Field(default=DEFAULT_MAX_TOPICS, ge=1, le=MAX_TOPICS_LIMIT)
    secondary_min_confidence: float = Field(default=DEFAULT_SECONDARY_MIN_CONFIDENCE, ge=0, le=1)
    # Evidence router only (`router: evidence`): `evidence_aggregate` turns a topic's `probe_k`
    # best similarities into its score (`max` = best passage, `mean_top_k` = their mean); the best
    # topic is always searched and another one only when its score is within `evidence_margin`
    # (cosine units) of the best, up to `max_topics`.
    evidence_aggregate: EvidenceAggregate = DEFAULT_EVIDENCE_AGGREGATE
    evidence_margin: float = Field(default=DEFAULT_EVIDENCE_MARGIN, ge=0, le=1)
    probe_k: int = Field(default=DEFAULT_PROBE_K, ge=1, le=MAX_PROBE_K)
    # Scope check before retrieval; `policy: evidence` needs `router: evidence`.
    clarification: ClarificationSpec = ClarificationSpec()
    # Merge: items per round of the primary topic (the others contribute one).
    primary_weight: int = Field(default=DEFAULT_PRIMARY_WEIGHT, ge=1, le=MAX_PRIMARY_WEIGHT)
    # Sufficiency loop: when the merged sources are not enough (`sufficiency`) the query is
    # rewritten and the topics searched again, at most `max_rewrites` times per turn (0 = never).
    # `relevance` (default) judges the hits' own evidence (rerank score only; without a reranker
    # there is no signal and it is always sufficient; `min_rerank_score` is its threshold) and
    # needs `min_sources` relevant ones; `count` only counts sources, which with top_k retrieval
    # practically never fires.
    sufficiency: SufficiencyKind = DEFAULT_SUFFICIENCY
    max_rewrites: int = Field(default=DEFAULT_MAX_REWRITES, ge=0, le=MAX_REWRITES_LIMIT)
    min_sources: int = Field(default=DEFAULT_MIN_SOURCES, ge=1, le=MAX_SOURCES_LIMIT)
    min_rerank_score: float = Field(default=DEFAULT_MIN_RERANK_SCORE, ge=0, le=1)
    # Retrieval and context knobs (all bounded).
    max_sources: int = Field(default=MAX_SOURCES, ge=1, le=MAX_SOURCES_LIMIT)
    max_source_chars: int = Field(default=MAX_SOURCE_CHARS, ge=100, le=MAX_SOURCE_CHARS_LIMIT)
    max_context_chars: int = Field(default=MAX_CONTEXT_CHARS, ge=500, le=MAX_CONTEXT_CHARS_LIMIT)
    max_history_messages: int = Field(default=MAX_HISTORY_MESSAGES, ge=1, le=MAX_HISTORY_LIMIT)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [t.name for t in self.topics]
        profiles = [t.profile for t in self.topics]
        for label, values in (("topic names", names), ("topic profiles", profiles)):
            duplicated = sorted({v for v in values if values.count(v) > 1})
            if duplicated:
                raise ValueError(f"{label} must be unique, repeated: {duplicated}")
        if self.router == "single" and len(self.topics) != 1:
            raise ValueError("router 'single' needs exactly one topic")
        if self.router in ("llm", "evidence") and len(self.topics) < 2:
            raise ValueError(
                f"router {self.router!r} needs at least two topics (use 'auto' or 'single')"
            )
        if self.clarification.policy == "evidence" and self.router != "evidence":
            raise ValueError("clarification policy 'evidence' needs `router: evidence`")
        low_topics = self.clarification.low_scope_max_topics
        if low_topics is not None and low_topics < self.max_topics:
            raise ValueError(
                "clarification.low_scope_max_topics cannot be below max_topics "
                "(equal means no widening)"
            )
        self._warn_inapplicable_knobs()
        if self.citations == "none" and _CITATION_MARKER.search(self.prompts.synthesize or ""):
            logger.warning(
                "agent %r: `citations: none` with a custom prompts.synthesize that mentions [n] "
                "markers: make sure it does not ask for them",
                self.id,
            )
        if ChatRole.ROUTER in self.models and not self.uses_llm_router:
            logger.warning(
                "agent %r: models.router is ignored (this agent's router makes no model call)",
                self.id,
            )
        if self.min_sources > self.max_sources:
            raise ValueError("min_sources cannot exceed max_sources (it could never be met)")
        return self

    def _warn_inapplicable_knobs(self) -> None:
        """Knobs of the other router are ignored: say so instead of letting a typo pass."""
        given = self.model_fields_set
        if self.router != "evidence":
            ignored = sorted(
                {"evidence_margin", "evidence_aggregate", "probe_k"} & given
                | {"evidence_bias" for t in self.topics if "evidence_bias" in t.model_fields_set}
            )
            if ignored:
                logger.warning(
                    "agent %r: %s ignored (they only apply to `router: evidence`)",
                    self.id,
                    ", ".join(ignored),
                )
        elif "secondary_min_confidence" in given:
            logger.warning(
                "agent %r: secondary_min_confidence is ignored by `router: evidence` "
                "(use evidence_margin)",
                self.id,
            )

    @property
    def uses_llm_router(self) -> bool:
        """Whether a question costs a ROUTER model call (one topic never does)."""
        return self.router == "llm" or (self.router == "auto" and len(self.topics) > 1)

    @property
    def uses_evidence_router(self) -> bool:
        """Whether routing probes the corpus (one embedding and one search per topic)."""
        return self.router == "evidence"

    @property
    def fan_out(self) -> int:
        """Most topics one attempt can search in the normal bands (the `low` band of the
        clarification policy may widen it up to `clarification.low_scope_max_topics`)."""
        multi = self.uses_llm_router or self.uses_evidence_router
        return min(self.max_topics, len(self.topics)) if multi else 1

    @property
    def limits(self) -> SourceLimits:
        return SourceLimits(self.max_sources, self.max_source_chars, self.max_context_chars)


def describe_errors(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors(include_url=False)
    )


def parse_agent_spec(raw: object, *, origin: str) -> AgentSpec:
    if not isinstance(raw, dict):
        raise InvalidAgentSpecError(f"{origin}: an agent spec file must be a YAML mapping")
    try:
        return AgentSpec.model_validate(raw)
    except ValidationError as exc:
        raise InvalidAgentSpecError(f"{origin}: {describe_errors(exc)}") from exc


def _has_profile(profiles: RagProfileRegistry, name: str) -> bool:
    try:
        profiles.get(name)
    except NotFoundError:
        return False
    return True


def validate_agent_specs(specs: Iterable[AgentSpec], profiles: RagProfileRegistry) -> None:
    """Fail fast (call from bootstrap): every topic profile exists and its filters are valid, and
    `min_sources` is reachable: at most the sum of the `top_k` of the largest profiles one attempt
    searches (`fan_out` of them; the primary alone for a single-topic agent).

    All problems are reported at once. Duplicate agent ids are caught by the loader.
    """
    problems: list[str] = []
    for spec in specs:
        profile_names = {t.profile for t in spec.topics}
        known = [profiles.get(n) for n in profile_names if _has_profile(profiles, n)]
        top_ks = sorted((p.top_k for p in known), reverse=True)
        if len(known) == len(profile_names) and spec.min_sources > sum(top_ks[: spec.fan_out]):
            problems.append(
                f"agent {spec.id!r}: min_sources={spec.min_sources} can never be met: one attempt "
                f"retrieves at most {sum(top_ks[: spec.fan_out])} hits (the top_k of the "
                f"{spec.fan_out} largest topic profile(s))"
            )
        if (
            spec.sufficiency == "relevance"
            and spec.max_rewrites > 0
            and len(known) == len(profile_names)
            and all(p.reranker_key == NO_RERANKER_KEY for p in known)
        ):
            logger.warning(
                "agent %r: sufficiency 'relevance' with max_rewrites=%d but no topic profile has "
                "a reranker: the rewrite loop never fires without a reranker",
                spec.id,
                spec.max_rewrites,
            )
        for topic in spec.topics:
            where = f"agent {spec.id!r} topic {topic.name!r}"
            try:
                profile = profiles.get(topic.profile)
            except NotFoundError:
                problems.append(f"{where}: unknown profile {topic.profile!r}")
                continue
            try:
                plans = plan_filter_fields(profile, topic.filters)
                if plans:
                    build_filter_schema(plans)  # a field name that cannot form a schema fails here
            except (ValueError, TypeError) as exc:
                problems.append(f"{where}: {exc}")
                continue
            if topic.filters is None and plans:
                logger.warning(
                    "%s: no `filters` allowlist, so ALL %d filterable fields of profile %r go to "
                    "the filter extractor (models invent values for fields the question never "
                    "states): list the fields to extract, with a hint each, under topic.filters",
                    where,
                    len(plans),
                    profile.name,
                )
    if problems:
        raise InvalidAgentSpecError("Invalid agent specs: " + "; ".join(problems))


__all__ = [
    "AgentSpec",
    "ClarificationSpec",
    "InvalidAgentSpecError",
    "PromptsSpec",
    "TopicSpec",
    "describe_errors",
    "parse_agent_spec",
    "validate_agent_specs",
]
