"""Retrieval baseline: hit@k and MRR of `RetrieveForTopic` over a questions file.

    uv run python -m agintis.cli.eval --profiles case/spensiones/profiles \
        --questions case/spensiones/profiles/eval_questions.jsonl \
        --k 1,3,5,10 --mode all --report /tmp/retrieval_report.json

Each question is retrieved with ITS profile and NO filters (raw retrieval: the locator is only
used to judge the hits), against the configured database, READ-ONLY (every connection opens with
`default_transaction_read_only=on`, so a bug cannot write). The profile's `top_k` is
raised to the largest `--k`, `--mode` overrides its `retrieval_mode` (`both` runs dense then
hybrid; `all`, the default, also bm25_hybrid where the database has pg_textsearch) and the
reranker is off unless `--rerank` (it is slow on CPU and would blur the comparison of the
retrieval modes). A hit is a chunk holding the expected snippet whose metadata
matches the locator (`agintis.cli.eval_metrics`). The query is embedded through the configured
embedding service.

`--extract-filters --agents DIR` also runs each question with the filters the agent would
extract: the SAME `FilterExtractor` code (real analyzer LLM, so it needs LLM_* configured and
costs one call per question; extracted once and reused across modes) and the same graded
zero-hit fallback (all filters -> only numeric/date ones -> none, see `retrieve_with_fallback`),
reported next to the plain run with the level each question ended at. Filters are extracted ONLY
for profiles an agent topic covers (its allowlist, hints and prompt come from the spec); the
other profiles run plain and the report says `skipped (no topic)`. By default the extractor reads
the raw question; `--analyze` runs the agent's AnalyzeNode first and extracts from its standalone
query (retrieval still uses the raw question, so the comparison stays like for like).

`--route --agents DIR` also runs each agent's TOPIC ROUTER per question (raw question, or the
standalone query with `--analyze`; one router
call per question, reused across modes; needs LLM_* unless every agent has a single topic). The
expected topic of a question is the topic whose profile owns it (`profile` of the question,
through the agent spec). Reported: router accuracy (the router's FIRST topic is the expected one
= hit@1; the expected topic is anywhere in the routed selection = any) and a routed retrieval:
each question is retrieved, plain (no filters), only in the profiles of the topics the router
selected, and the per-topic hit lists are merged by rank with the agent's own `interleave`
(weighted round-robin in router order, deduplicated by chunk id), cut at min(largest k, the
agent's max_sources) (`retrieval_cut` in the report). It sits next to the per-profile run of the
same mode, which is the oracle (what retrieval does when the router is always right) and is
reused, not repeated. The report adds each question's router confidences and secondary topics.
Questions whose profile no topic covers run plain on their own profile and are not routed.

`--router {spec,llm,evidence}` (default `spec`: each agent's own `router`) overrides the routing
Strategy of every multi-topic agent without editing the YAML, so the LLM router and the EVIDENCE
router (no model: one query embedding + a dense probe of each topic's profile, see
`EvidenceTopicRouter`) are compared in two invocations on the same questions. The report names
the router and adds its cost (LLM calls, probe searches and embeddings, seconds). With the
evidence router it also reports, per question, every topic's score, the expected topic's rank
and its gap to the best one; the distribution of the best score and of that gap when the router
is right vs wrong; and a margin sweep (`any_selected` and topics per question for a range of
`evidence_margin` values, at the agent's max_topics) to calibrate `evidence_margin`.

It is a measurement, not a gate: the exit code is 0 whatever the numbers. 2 = bad arguments, 3 =
the run was aborted (database or embedding service down, unknown profile).

The JSON report carries what is needed to reproduce a number: retrieval knobs (branch depth, RRF
k, HNSW ef_search, text search config), the embedding model, the git revision, the baseline
fingerprint of each profile used and the sha256 of the questions file. MRR is MRR@max(k).
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import statistics
import sys
import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agintis.agent_kit.nodes import (
    DEFAULT_PRIMARY_WEIGHT,
    AnalyzeNode,
    EvidenceTopicRouter,
    FilterClause,
    FilterExtractor,
    RoutedTopic,
    RouteScope,
    TopicEvidence,
    TopicRouter,
    fallback_topics,
    interleave,
    primary_weights,
    retrieve_with_fallback,
    select_by_evidence,
)
from agintis.agent_kit.runtime import QueryVectors
from agintis.bootstrap.container import Container, StartupError
from agintis.bootstrap.llm import build_chat_models
from agintis.bootstrap.settings import Settings, get_settings
from agintis.cli._common import (
    DEFAULT_TENANT,
    EXIT_ABORTED,
    EXIT_OK,
    EXIT_USAGE,
    UsageError,
    git_revision,
)
from agintis.cli.eval_metrics import (
    DEFAULT_KS,
    EvalQuestion,
    QuestionOutcome,
    first_relevant_rank,
    format_comparison,
    format_table,
    group_by_profile,
    load_questions,
    summarize,
    summary_to_dict,
)
from agintis.modules.agents.rag_chat import AgentSpec, TopicSpec, build_router, load_agent_specs
from agintis.modules.knowledge.api import (
    BM25_HYBRID_RETRIEVAL_MODE,
    DEFAULT_MAX_QUERY_CHARS,
    DENSE_RETRIEVAL_MODE,
    HYBRID_RETRIEVAL_MODE,
    RagProfile,
    TextSearchConfigError,
)
from agintis.platform.llm import ChatModels, ChatRole, LlmNotConfiguredError
from agintis.shared_kernel.ids import TenantId

BOTH = "both"
ALL = "all"
# Mode sets of `--mode`; `all` skips bm25_hybrid (with a note) where pg_textsearch is missing.
MODE_SETS = {
    BOTH: (DENSE_RETRIEVAL_MODE, HYBRID_RETRIEVAL_MODE),
    ALL: (DENSE_RETRIEVAL_MODE, HYBRID_RETRIEVAL_MODE, BM25_HYBRID_RETRIEVAL_MODE),
}
EXTRACTED = "extracted"
EXTRACTION_SKIPPED = "skipped (no topic)"
ROUTED = "routed"
ROUTE_SKIPPED = "skipped (no topic)"
FILTER_INPUT_QUESTION = "question"
FILTER_INPUT_STANDALONE = "standalone_query"
NO_RERANKER = "none"
ROUTER_SPEC = "spec"
ROUTER_CHOICES = (ROUTER_SPEC, "llm", "evidence")
# `evidence_margin` values the sweep tries (cosine-similarity units).
MARGIN_SWEEP = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
logger = logging.getLogger("agintis.cli.eval")


def parse_ks(raw: str) -> tuple[int, ...]:
    try:
        ks = tuple(sorted({int(part) for part in raw.split(",") if part.strip()}))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--k must be comma-separated integers, got {raw!r}"
        ) from None
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError("--k needs at least one value, all >= 1")
    return ks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profiles", type=Path, required=True, help="directory of YAML profiles")
    parser.add_argument("--questions", type=Path, required=True, help="JSON Lines questions")
    parser.add_argument(
        "--k",
        type=parse_ks,
        default=DEFAULT_KS,
        help="cutoffs, comma-separated (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--mode",
        choices=[
            DENSE_RETRIEVAL_MODE,
            HYBRID_RETRIEVAL_MODE,
            BM25_HYBRID_RETRIEVAL_MODE,
            BOTH,
            ALL,
        ],
        default=ALL,
        help=(
            "retrieval mode to measure: one mode, `both` (dense, hybrid) or `all` (both plus "
            "bm25_hybrid, skipped with a note when the database lacks pg_textsearch; an "
            "explicit bm25_hybrid fails there) (default: %(default)s)"
        ),
    )
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="tenant id (default: %(default)s)")
    parser.add_argument(
        "--rerank", action="store_true", help="keep each profile's reranker (default: off)"
    )
    parser.add_argument(
        "--extract-filters",
        action="store_true",
        help="also run with the filters the analyzer LLM extracts (needs LLM_*), side by side",
    )
    parser.add_argument(
        "--agents",
        type=Path,
        help="directory of agent specs: the topics (profiles) --extract-filters/--route cover",
    )
    parser.add_argument(
        "--route",
        action="store_true",
        help="also run the agents' topic router per question: router accuracy and the retrieval "
        "restricted to the routed topics (needs --agents; LLM_* unless single-topic agents)",
    )
    parser.add_argument(
        "--router",
        choices=ROUTER_CHOICES,
        default=ROUTER_SPEC,
        help="with --route: the routing strategy of the multi-topic agents: their own (spec), "
        "the LLM router or the evidence router (default: %(default)s)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="with --extract-filters: extract from the AnalyzeNode's standalone query",
    )
    return parser


def router_choice(args: argparse.Namespace) -> str:
    """`--router` (programmatic callers that never set it get `spec`)."""
    return str(getattr(args, "router", ROUTER_SPEC))


def check_usage(args: argparse.Namespace) -> None:
    if args.analyze and not (args.extract_filters or args.route):
        raise UsageError("--analyze needs --extract-filters or --route")
    if args.extract_filters and args.agents is None:
        raise UsageError(
            "--extract-filters needs --agents DIR: filters are extracted only for the profiles "
            "an agent topic covers"
        )
    if args.route and args.agents is None:
        raise UsageError("--route needs --agents DIR: the router and its topics come from there")
    if router_choice(args) != ROUTER_SPEC and not args.route:
        raise UsageError("--router needs --route")
    if args.agents is not None and not (args.extract_filters or args.route):
        logger.warning("--agents has no effect without --extract-filters or --route")


def eval_profile(
    container: Container, name: str, mode: str, ks: Sequence[int], rerank: bool
) -> RagProfile:
    """The registered profile as the eval runs it: depth = largest k, the mode under test and
    the reranker only on request."""
    base = container.profiles.get(name)
    return replace(
        base,
        top_k=max(ks),
        retrieval_mode=mode,
        reranker_key=base.reranker_key if rerank else NO_RERANKER,
    )


async def evaluate(
    container: Container,
    questions: Sequence[EvalQuestion],
    *,
    mode: str,
    ks: Sequence[int],
    tenant: TenantId,
    rerank: bool,
    filters: Mapping[str, list[FilterClause]] | None = None,
) -> list[QuestionOutcome]:
    """`filters` (question id -> clauses) switches to the agent's retrieval: the graded fallback
    ladder over those clauses (a question without clauses is simply plain). Without it: raw
    retrieval, exactly as always."""
    outcomes: list[QuestionOutcome] = []
    for question in questions:
        profile = eval_profile(container, question.profile, mode, ks, rerank)
        level: str | None = None
        relaxed = False
        if filters is None:
            result = await container.retrieve_for_topic.execute(tenant, profile, question.question)
        else:
            found = await retrieve_with_fallback(
                container.retrieve_for_topic,
                tenant,
                profile,
                question.question,
                filters.get(question.id, []),
            )
            result, level, relaxed = found.result, found.level, found.relaxed
        rank = first_relevant_rank(
            ((h.chunk.chunk.text, h.chunk.chunk.metadata) for h in result.hits), question
        )
        outcomes.append(
            QuestionOutcome(
                question.id,
                question.profile,
                question.type,
                rank,
                len(result.hits),
                relaxed,
                level,
            )
        )
    return outcomes


@dataclass(frozen=True)
class TopicBinding:
    """A topic and the agent spec that owns it (prompts, router and limits come from the spec)."""

    spec: AgentSpec
    topic: TopicSpec


def topics_by_profile(specs: Sequence[AgentSpec]) -> dict[str, TopicBinding]:
    """The topic that covers each profile; two topics on one profile are ambiguous."""
    topics: dict[str, TopicBinding] = {}
    for spec in specs:
        for topic in spec.topics:
            if topic.profile in topics:
                raise UsageError(
                    f"agents {topics[topic.profile].spec.id!r} and {spec.id!r} both cover "
                    f"profile {topic.profile!r}: the eval needs one topic per profile"
                )
            topics[topic.profile] = TopicBinding(spec, topic)
    return topics


def _agent_models(
    container: Container, spec: AgentSpec, *roles: ChatRole, verify: bool = True
) -> ChatModels:
    """The chat models the agent itself resolves for the roles a step calls (its `models:`
    bindings over the defaults; other roles are never resolved). Verified first (offline), so a
    profile-bound analyzer passes the structured-output check before the first question."""
    models = container.chat_models.bind(spec.models, agent=spec.id, roles=roles)
    if verify:
        models.verify()
    return models


def _resolved_models(
    container: Container, specs: Iterable[AgentSpec], role: ChatRole
) -> dict[str, str]:
    """Agent id -> `profile (provider:model)` the agent resolves for `role` (for the reports)."""
    return {
        spec.id: _agent_models(container, spec, role, verify=False).describe(role).label
        for spec in specs
    }


def check_llm_available(
    settings: Settings, specs: Collection[AgentSpec], needs: Mapping[ChatRole, str]
) -> None:
    """Fail BEFORE any boot work when an agent has no model for a role a flag needs: neither the
    platform default (`LLM_PROVIDER`) nor a profile bound under the spec's `models:`.

    `needs`: role -> the flag that needs it. A bad binding raises its own (descriptive) LlmError.
    """
    models = build_chat_models(settings)
    for role, flag in needs.items():
        missing = []
        for spec in specs:
            if role is ChatRole.ROUTER and not spec.uses_llm_router:
                continue
            try:
                models.bind(spec.models, agent=spec.id, roles=(role,)).describe(role)
            except LlmNotConfiguredError:
                missing.append(spec.id)
        if missing:
            raise RuntimeError(
                f"{flag} needs the LLM for the {role.value} of {sorted(missing)}: set LLM_PROVIDER "
                "(and LLM_*) or bind the role to a profile under `models:` in the spec"
            )


def _covering_specs(topics: Mapping[str, TopicBinding], used: Sequence[str]) -> list[AgentSpec]:
    """The distinct agents whose topics cover the profiles the questions use (spec order)."""
    specs = {topics[name].spec.id: topics[name].spec for name in used if name in topics}
    return list(specs.values())


def build_filter_extractors(
    container: Container,
    topics: Mapping[str, TopicBinding],
    profile_names: Sequence[str],
) -> dict[str, FilterExtractor]:
    """An extractor per covered profile, configured like the agent topic that covers it."""
    return {
        name: FilterExtractor(
            _agent_models(container, topics[name].spec, ChatRole.ANALYZER),
            container.profiles.get(name),
            topics[name].topic.filters,
            topics[name]
            .spec.prompts.to_prompt_set(citations=topics[name].spec.citations)
            .extract_filters,
        )
        for name in profile_names
        if name in topics
    }


def build_analyzers(
    container: Container, topics: Mapping[str, TopicBinding], profile_names: Sequence[str]
) -> dict[str, AnalyzeNode]:
    """The agent's own AnalyzeNode (its prompt and history window) per covered profile."""
    return {
        name: AnalyzeNode(
            _agent_models(container, topics[name].spec, ChatRole.ANALYZER),
            topics[name].spec.prompts.to_prompt_set(citations=topics[name].spec.citations).analyze,
            topics[name].spec.max_history_messages,
        )
        for name in profile_names
        if name in topics
    }


def apply_router_override(specs: Sequence[AgentSpec], router: str) -> list[AgentSpec]:
    """The specs with `router` forced on every multi-topic agent (`spec` keeps them as they
    are; a single-topic agent has nothing to route and is left alone)."""
    if router == ROUTER_SPEC:
        return list(specs)
    # Re-validated from the SET fields only: the override is a real spec (its validators run)
    # and the knobs the YAML never wrote do not look explicitly set to the warnings.
    return [
        spec
        if len(spec.topics) < 2
        else AgentSpec.model_validate({**spec.model_dump(exclude_unset=True), "router": router})
        for spec in specs
    ]


def build_routers(
    container: Container,
    topics: Mapping[str, TopicBinding],
    profile_names: Sequence[str],
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
) -> dict[str, TopicRouter]:
    """The agent's own router (strategy, prompt, knobs) per spec that owns a used profile."""
    specs = {topics[name].spec.id: topics[name].spec for name in profile_names if name in topics}
    return {
        spec_id: build_router(
            spec,
            _agent_models(container, spec, *([ChatRole.ROUTER] if spec.uses_llm_router else [])),
            probe=container.probe_profiles if spec.uses_evidence_router else None,
            profiles=container.profiles,
            max_query_chars=max_query_chars,
        )
        for spec_id, spec in specs.items()
    }


async def standalone_queries(
    questions: Sequence[EvalQuestion],
    analyzers: Mapping[str, AnalyzeNode],
) -> dict[str, str]:
    """The AnalyzeNode's standalone query of every question whose profile is covered."""
    return {
        q.id: (await analyzers[q.profile].analyze([HumanMessage(q.question)]))["standalone_query"]
        for q in questions
        if q.profile in analyzers
    }


async def route_questions(
    questions: Sequence[EvalQuestion],
    topics: Mapping[str, TopicBinding],
    routers: Mapping[str, TopicRouter],
    queries: Mapping[str, str] | None = None,
    *,
    tenant: TenantId = TenantId(DEFAULT_TENANT),  # noqa: B008 (immutable)
    scopes: dict[str, RouteScope] | None = None,
    evidence: dict[str, list[TopicEvidence]] | None = None,
) -> dict[str, list[RoutedTopic]]:
    """The topics (primary first, with confidences) the owning agent's router selects for each
    question whose profile is covered; one router call per question, on `queries[id]` (the
    standalone query, with `--analyze`) or the raw question. Others are not routed.

    Every question gets its own `RouteScope` (as the tenant, with a fresh vector memo); they are
    kept in `scopes` so the routed retrieval can reuse the query embedding an evidence router
    already computed. An evidence router's per-topic evidence (all topics, selected or not) is
    collected in `evidence`; it is probed once and the selection derived from it."""
    routes: dict[str, list[RoutedTopic]] = {}
    for question in questions:
        binding = topics.get(question.profile)
        if binding is None:
            continue
        text = (queries or {}).get(question.id, question.question)
        scope = RouteScope(tenant, QueryVectors())
        if scopes is not None:
            scopes[question.id] = scope
        router = routers[binding.spec.id]
        if evidence is not None and isinstance(router, EvidenceTopicRouter):
            found = await router.evidence(text, scope)
            evidence[question.id] = found
            routes[question.id] = router.select(found)
        else:
            routes[question.id] = await router.route(text, scope)
    return routes


def router_cost(
    questions: Sequence[EvalQuestion],
    topics: Mapping[str, TopicBinding],
    routes: Mapping[str, list[RoutedTopic]],
    seconds: float,
    queries: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """What routing cost: LLM router calls, evidence probes (one embedding and one dense search
    per topic of the agent, per question; a blank query probes nothing) and the wall time of
    the whole routing step. `queries` = the routed text per question (default: the question)."""
    llm_calls = probe_embeddings = probe_searches = 0
    for question in questions:
        binding = topics.get(question.profile)
        if binding is None or question.id not in routes:
            continue
        if binding.spec.uses_llm_router:
            llm_calls += 1
        elif binding.spec.uses_evidence_router:
            if not (queries or {}).get(question.id, question.question).strip():
                continue
            probe_embeddings += 1
            probe_searches += len(binding.spec.topics)
    return {
        "llm_calls": llm_calls,
        "probe_embeddings": probe_embeddings,
        "probe_searches": probe_searches,
        "seconds": round(seconds, 3),
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    """n, mean, min, median and max of `values` (zeros when empty)."""
    if not values:
        return {"n": 0, "mean": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def evidence_summary(
    questions: Sequence[EvalQuestion],
    topics: Mapping[str, TopicBinding],
    evidence: Mapping[str, Sequence[TopicEvidence]],
    margins: Sequence[float] = MARGIN_SWEEP,
) -> dict[str, Any]:
    """Calibration data of the evidence router over the questions it probed.

    Per question: every topic's score, the best topic, the expected topic's rank and score, and
    `gap_to_best` (best score - expected score, 0.0 when it is the best; None when the expected
    topic had no hit). Aggregates: the distribution of the best score and of `runner_up_gap`
    (best - second best) when the primary is right, and of the best score and `gap_to_best`
    when it is wrong; `margin_sweep` = for each candidate `evidence_margin` the share of
    questions whose selection (capped at the agent's max_topics) holds the expected topic and
    the mean number of topics selected."""
    rows: list[dict[str, Any]] = []
    caps: list[int] = []  # the agent's max_topics of each row
    names_of: list[list[str]] = []  # the agent's topics, for the router's fallback
    for question in questions:
        found = evidence.get(question.id)
        binding = topics.get(question.profile)
        if found is None or binding is None:
            continue
        caps.append(binding.spec.max_topics)
        names_of.append([t.name for t in binding.spec.topics])
        expected = binding.topic.name
        scored = sorted(
            ((e.topic, e.score) for e in found if e.score is not None), key=lambda p: -p[1]
        )
        best_topic, best_score = scored[0] if scored else (None, None)
        by_topic = dict(scored)
        expected_score = by_topic.get(expected)
        rows.append(
            {
                "id": question.id,
                "expected": expected,
                "scores": {e.topic: e.score for e in found},
                "best": best_topic,
                "best_score": best_score,
                "second_gap": scored[0][1] - scored[1][1] if len(scored) > 1 else None,
                "expected_score": expected_score,
                "expected_rank": next(
                    (i + 1 for i, (name, _) in enumerate(scored) if name == expected), None
                ),
                "gap_to_best": (
                    best_score - expected_score
                    if best_score is not None and expected_score is not None
                    else None
                ),
                "hit_at_1": best_topic == expected,
            }
        )

    def column(key: str, right: bool) -> list[float]:
        return [r[key] for r in rows if r["hit_at_1"] is right and r[key] is not None]

    sweep = []
    total = len(rows)
    for margin in margins:
        # The router's real selection, `fallback_topics` included (as `EvidenceTopicRouter.select`).
        selected = [
            [
                t["topic"]
                for t in select_by_evidence(evidence[row["id"]], cap, margin)
                or fallback_topics(names, cap)
            ]
            for row, cap, names in zip(rows, caps, names_of, strict=True)
        ]
        sweep.append(
            {
                "margin": margin,
                "any_selected": (
                    sum(r["expected"] in names for r, names in zip(rows, selected, strict=True))
                    / total
                    if total
                    else 0.0
                ),
                "mean_topics": sum(len(names) for names in selected) / total if total else 0.0,
            }
        )
    # Per topic, its score (without bias) when it IS the expected topic vs when it is NOT (its
    # "background" level on questions about other topics): a one-sided offset between topics
    # shows here. Judge it, but never tune a bias on this (held-out) set.
    topic_scores: dict[str, dict[str, Any]] = {}
    for row in rows:
        for found_topic in evidence[row["id"]]:
            raw = found_topic.raw_score
            if raw is None or not math.isfinite(raw):
                continue
            slot = topic_scores.setdefault(found_topic.topic, {"expected": [], "background": []})
            slot["expected" if found_topic.topic == row["expected"] else "background"].append(raw)
    return {
        "topic_scores": {
            name: {
                "when_expected": _distribution(values["expected"]),
                "background": _distribution(values["background"]),
            }
            for name, values in topic_scores.items()
        },
        "best_score_when_right": _distribution(column("best_score", True)),
        "best_score_when_wrong": _distribution(column("best_score", False)),
        "runner_up_gap_when_right": _distribution(column("second_gap", True)),
        "gap_to_best_when_wrong": _distribution(column("gap_to_best", False)),
        "margin_sweep": sweep,
        "rows": rows,
    }


def router_summary(
    questions: Sequence[EvalQuestion],
    topics: Mapping[str, TopicBinding],
    routes: Mapping[str, list[RoutedTopic]],
) -> dict[str, Any]:
    """Router accuracy over the routed questions: `hit_at_1` = the primary topic is the one
    whose profile owns the question; `any_selected` = it is anywhere in the selection. Rows carry
    the confidences and the secondary selections (everything after the primary);
    `confidence_when_right` / `confidence_when_wrong` average the primary's confidence by
    outcome, to calibrate `secondary_min_confidence`."""
    rows: list[dict[str, Any]] = []
    for question in questions:
        routed = routes.get(question.id)
        if routed is None:
            continue
        expected = topics[question.profile].topic.name
        names = [r["topic"] for r in routed]
        rows.append(
            {
                "id": question.id,
                "expected": expected,
                "routed": names,
                "confidences": [r["confidence"] for r in routed],
                "secondary": names[1:],
                "hit_at_1": names[0] == expected,
                "any_selected": expected in names,
            }
        )
    total = len(rows)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "questions": total,
        "hit_at_1": sum(r["hit_at_1"] for r in rows) / total if total else 0.0,
        "any_selected": sum(r["any_selected"] for r in rows) / total if total else 0.0,
        "mean_topics_selected": sum(len(r["routed"]) for r in rows) / total if total else 0.0,
        "with_secondary": sum(1 for r in rows if r["secondary"]),
        "confidence_when_right": mean([r["confidences"][0] for r in rows if r["hit_at_1"]]),
        "confidence_when_wrong": mean([r["confidences"][0] for r in rows if not r["hit_at_1"]]),
        "rows": rows,
    }


def routed_cut(spec: AgentSpec, ks: Sequence[int]) -> int:
    """Hits kept of the merged list: what the agent would show (`max_sources`), never more than
    the deepest k measured."""
    return min(max(ks), spec.max_sources)


async def evaluate_routed(
    container: Container,
    questions: Sequence[EvalQuestion],
    topics: Mapping[str, TopicBinding],
    routes: Mapping[str, list[RoutedTopic]],
    *,
    mode: str,
    ks: Sequence[int],
    tenant: TenantId,
    rerank: bool,
    scopes: Mapping[str, RouteScope] | None = None,
) -> list[QuestionOutcome]:
    """Plain retrieval (no filters) in the profiles of the ROUTED topics only, merged like the
    agent merges them (`interleave`: weighted round-robin, the primary topic `primary_weight`
    items per round, dedupe by chunk id) and cut at `routed_cut`. A question that was not routed
    runs on its own profile. `scopes` (from `route_questions`) hand over the query embedding an
    evidence router already computed for the raw question, so it is not embedded again."""
    outcomes: list[QuestionOutcome] = []
    for question in questions:
        routed = routes.get(question.id)
        binding = topics.get(question.profile)
        if routed is None or binding is None:
            names, weight, cut = [question.profile], DEFAULT_PRIMARY_WEIGHT, max(ks)
        else:
            by_name = {t.name: t.profile for t in binding.spec.topics}
            names = [by_name[r["topic"]] for r in routed]
            weight, cut = binding.spec.primary_weight, routed_cut(binding.spec, ks)
        scope = (scopes or {}).get(question.id)
        vector = scope.vectors.get(question.question) if scope is not None else None
        groups = []
        for name in names:
            profile = eval_profile(container, name, mode, ks, rerank)
            result = await container.retrieve_for_topic.execute(
                tenant, profile, question.question, query_vector=vector
            )
            groups.append(list(result.hits))
        hits = interleave(
            groups, lambda hit: hit.citation.chunk_id, primary_weights(len(groups), weight)
        )[:cut]
        rank = first_relevant_rank(
            ((h.chunk.chunk.text, h.chunk.chunk.metadata) for h in hits), question
        )
        outcomes.append(
            QuestionOutcome(question.id, question.profile, question.type, rank, len(hits))
        )
    return outcomes


async def extract_question_filters(
    questions: Sequence[EvalQuestion],
    extractors: Mapping[str, FilterExtractor],
    analyzers: Mapping[str, AnalyzeNode] | None = None,
) -> tuple[dict[str, list[FilterClause]], dict[str, str]]:
    """The clauses the extractor finds for every question whose profile has one, and (with
    `analyzers`) the standalone query each was extracted from. One LLM call per question and
    step; questions of uncovered profiles are skipped."""
    clauses: dict[str, list[FilterClause]] = {}
    inputs: dict[str, str] = {}
    for question in questions:
        extractor = extractors.get(question.profile)
        if extractor is None:
            continue
        text = question.question
        if analyzers is not None:
            analysis = await analyzers[question.profile].analyze([HumanMessage(text)])
            text = inputs[question.id] = analysis["standalone_query"]
        clauses[question.id] = await extractor.extract(text)
    return clauses, inputs


def report_for(
    outcomes: Sequence[QuestionOutcome],
    ks: Sequence[int],
    clauses: Mapping[str, list[FilterClause]] | None = None,
    extraction: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """`clauses` (filtered runs) adds each question's extracted filters, the fallback level it
    ended at and whether it relaxed; `extraction` (profile -> status) adds how its filters were
    obtained (`extracted` or `skipped (no topic)`)."""
    report: dict[str, Any] = {
        "overall": summary_to_dict(summarize(outcomes, ks)),
        "profiles": {
            name: summary_to_dict(summarize(group, ks))
            for name, group in group_by_profile(outcomes).items()
        },
        "questions": [
            {
                "id": o.question_id,
                "profile": o.profile,
                "type": o.type,
                "rank": o.rank,
                "retrieved": o.retrieved,
            }
            for o in outcomes
        ],
    }
    if clauses is not None:
        for row, o in zip(report["questions"], outcomes, strict=True):
            row["filters"] = clauses.get(o.question_id, [])
            row["filter_level"] = o.filter_level
            row["filters_relaxed"] = o.filters_relaxed
            if extraction is not None:
                row["filter_extraction"] = extraction.get(o.profile, EXTRACTION_SKIPPED)
    return report


def provenance(
    settings: Settings,
    profiles: Sequence[RagProfile],
    questions_file: Path,
    *,
    revision: str | None,
) -> dict[str, Any]:
    """Everything that can change a retrieval number besides the code under test."""
    return {
        "git": revision,
        "questions_sha256": hashlib.sha256(questions_file.read_bytes()).hexdigest(),
        "embeddings_model": settings.embeddings.model,
        "text_search_config": settings.knowledge.text_search_config,
        "hybrid_branch_depth": settings.knowledge.hybrid_branch_depth,
        "hybrid_rrf_k": settings.knowledge.hybrid_rrf_k,
        "hnsw_ef_search": settings.database.hnsw_ef_search,
        "hnsw_max_scan_tuples": settings.database.hnsw_max_scan_tuples,
        "profile_fingerprints": {p.name: p.fingerprint for p in sorted(profiles, key=_name)},
    }


def _name(profile: RagProfile) -> str:
    return profile.name


async def run(args: argparse.Namespace) -> dict[str, Any]:
    check_usage(args)
    settings = get_settings()
    uses_agents = args.extract_filters or args.route
    settings = settings.model_copy(
        update={
            "knowledge": settings.knowledge.model_copy(update={"profiles_dir": args.profiles}),
            # A measurement must never write: the server refuses it.
            "database": settings.database.model_copy(update={"read_only": True}),
            # No conversations either: no checkpointer pool, no checkpoint-table boot check.
            "checkpointer": settings.checkpointer.model_copy(update={"enabled": False}),
            # Agent specs are validated against the profiles of the RUNNING app, not these.
            "agents": settings.agents.model_copy(update={"dir": None}),
        }
    )
    questions = load_questions(args.questions)
    topics = (
        topics_by_profile(apply_router_override(load_agent_specs(args.agents), router_choice(args)))
        if uses_agents
        else {}
    )
    needs: dict[ChatRole, str] = {}
    if args.extract_filters or (args.route and args.analyze):
        needs[ChatRole.ANALYZER] = "--extract-filters" if args.extract_filters else "--analyze"
    if args.route:
        needs[ChatRole.ROUTER] = "--route"
    check_llm_available(settings, {b.spec.id: b.spec for b in topics.values()}.values(), needs)
    modes = list(MODE_SETS.get(args.mode, (args.mode,)))
    report: dict[str, Any] = {
        "tenant": args.tenant,
        "ks": list(args.k),
        "questions_file": str(args.questions),
        "rerank": args.rerank,
        "modes": {},
    }
    async with Container.build(settings) as container:
        if BM25_HYBRID_RETRIEVAL_MODE in modes:
            try:
                await container.require_pg_textsearch()
                # A tenant without its index fails here, before any model call.
                await container.chunks.search_bm25(TenantId(args.tenant), "probe", None, 1)
            except (StartupError, TextSearchConfigError) as exc:
                if args.mode != ALL:
                    raise
                print(f"skipping {BM25_HYBRID_RETRIEVAL_MODE}: {exc}")
                modes.remove(BM25_HYBRID_RETRIEVAL_MODE)
                report["skipped_modes"] = {BM25_HYBRID_RETRIEVAL_MODE: str(exc)}
        used = sorted({q.profile for q in questions})
        report["provenance"] = provenance(
            settings,
            [container.profiles.get(name) for name in used],
            args.questions,
            revision=git_revision(),
        )
        extracted: dict[str, list[FilterClause]] | None = None
        extraction: dict[str, str] = {}
        analysis: dict[str, str] = {}
        if args.extract_filters:
            extractors = build_filter_extractors(container, topics, used)
            analyzers = build_analyzers(container, topics, used) if args.analyze else None
            extraction = {n: EXTRACTED if n in extractors else EXTRACTION_SKIPPED for n in used}
            print("filter extraction per profile:")
            for name in used:
                print(f"  {name}: {extraction[name]}")
            print("extracting filters (one analyzer call per question and step)...")
            extracted, analysis = await extract_question_filters(questions, extractors, analyzers)
            report["extract_filters"] = True
            report["filter_input"] = (
                FILTER_INPUT_STANDALONE if args.analyze else FILTER_INPUT_QUESTION
            )
            report["analyzer_model"] = _resolved_models(
                container, _covering_specs(topics, used), ChatRole.ANALYZER
            )
            report["filter_extraction"] = extraction
            report["filters"] = extracted
            if args.analyze:
                report["standalone_queries"] = analysis
            _print_extracted(extracted)
        routes: dict[str, list[RoutedTopic]] | None = None
        scopes: dict[str, RouteScope] = {}
        if args.route:
            routers = build_routers(container, topics, used, settings.knowledge.max_query_chars)
            queries = analysis if args.extract_filters and args.analyze else None
            if args.analyze and queries is None:
                queries = await standalone_queries(
                    questions, build_analyzers(container, topics, used)
                )
            print("routing questions (one router call per question)...")
            evidence: dict[str, list[TopicEvidence]] = {}
            started = time.perf_counter()
            routes = await route_questions(
                questions,
                topics,
                routers,
                queries,
                tenant=TenantId(args.tenant),
                scopes=scopes,
                evidence=evidence,
            )
            seconds = time.perf_counter() - started
            summary = router_summary(questions, topics, routes)
            cost = router_cost(questions, topics, routes, seconds, queries)
            kinds = {b.spec.id: _router_kind(b.spec) for b in topics.values()}
            report["route"] = {
                "router": router_choice(args),
                "router_kinds": kinds,
                "cost": cost,
                # What `confidence` means in the rows below, per router kind.
                "confidence_kind": (
                    "cosine_score" if set(kinds.values()) == {"evidence"} else "model_confidence"
                ),
                **(
                    {
                        "router_model": _resolved_models(
                            container,
                            [b.spec for b in topics.values() if b.spec.uses_llm_router],
                            ChatRole.ROUTER,
                        )
                    }
                    if "llm" in kinds.values()
                    else {}
                ),
                "route_input": FILTER_INPUT_STANDALONE if queries is not None else "question",
                "status": {n: ROUTED if n in topics else ROUTE_SKIPPED for n in used},
                "retrieval_cut": {b.spec.id: routed_cut(b.spec, args.k) for b in topics.values()},
                **summary,
            }
            if evidence:
                report["route"]["evidence"] = evidence_summary(questions, topics, evidence)
            _print_routes(summary, report["route"]["confidence_kind"] == "cosine_score")
            print(
                f"router cost: {cost['llm_calls']} LLM calls, {cost['probe_embeddings']} probe "
                f"embeddings, {cost['probe_searches']} probe searches, {cost['seconds']:.1f} s"
            )
            if evidence:
                _print_evidence(report["route"]["evidence"])
        for mode in modes:
            outcomes = await evaluate(
                container,
                questions,
                mode=mode,
                ks=args.k,
                tenant=TenantId(args.tenant),
                rerank=args.rerank,
            )
            print(f"\nmode: {mode}")
            print(format_table(outcomes, args.k))
            misses = [o for o in outcomes if o.rank is None]
            print(
                f"not retrieved within k={max(args.k)}: "
                + (", ".join(o.question_id for o in misses) or "-")
            )
            report["modes"][mode] = report_for(outcomes, args.k)
            if extracted is not None:
                filtered = await evaluate(
                    container,
                    questions,
                    mode=mode,
                    ks=args.k,
                    tenant=TenantId(args.tenant),
                    rerank=args.rerank,
                    filters=extracted,
                )
                print(f"\nmode: {mode} WITH extracted filters")
                print(format_table(filtered, args.k))
                print(f"\nmode: {mode}, plain vs filters")
                print(format_comparison(outcomes, filtered, args.k))
                report["modes"][mode]["with_filters"] = report_for(
                    filtered, args.k, extracted, extraction
                )
            if routes is not None:
                # The per-profile run above is the oracle: it is reused, not repeated.
                routed = await evaluate_routed(
                    container,
                    questions,
                    topics,
                    routes,
                    mode=mode,
                    ks=args.k,
                    tenant=TenantId(args.tenant),
                    rerank=args.rerank,
                    scopes=scopes,
                )
                print(f"\nmode: {mode} ROUTED (merged retrieval in the routed topics)")
                print(format_table(routed, args.k))
                print(f"\nmode: {mode}, per-profile oracle vs routed")
                print(format_comparison(outcomes, routed, args.k, "routed"))
                report["modes"][mode]["routed"] = report_for(routed, args.k)
    return report


def _print_extracted(extracted: Mapping[str, list[FilterClause]]) -> None:
    with_filters = {qid: clauses for qid, clauses in extracted.items() if clauses}
    print(f"filters extracted for {len(with_filters)}/{len(extracted)} questions")
    for qid, clauses in with_filters.items():
        print(f"  {qid}: " + ", ".join(f"{c['field']} {c['op']} {c['value']!r}" for c in clauses))


def _router_kind(spec: AgentSpec) -> str:
    """The strategy a spec routes with (`auto` resolved)."""
    if spec.uses_evidence_router:
        return "evidence"
    return "llm" if spec.uses_llm_router else "single"


def _print_evidence(summary: Mapping[str, Any]) -> None:
    right, wrong = summary["best_score_when_right"], summary["best_score_when_wrong"]
    runner_up, gap = summary["runner_up_gap_when_right"], summary["gap_to_best_when_wrong"]
    print(
        f"evidence: best score {right['mean']:.3f} when right (n={right['n']}, "
        f"min {right['min']:.3f}), {wrong['mean']:.3f} when wrong (n={wrong['n']}, "
        f"max {wrong['max']:.3f}); runner-up gap when right: median {runner_up['median']:.3f} "
        f"(min {runner_up['min']:.3f}); expected-topic gap when wrong: median "
        f"{gap['median']:.3f} (max {gap['max']:.3f})"
    )
    print("  per topic score (no bias): when it is the expected topic | background (not expected)")
    for name, dists in summary["topic_scores"].items():
        exp, bg = dists["when_expected"], dists["background"]
        print(
            f"  {name}: expected n={exp['n']} median {exp['median']:.3f} | "
            f"background n={bg['n']} median {bg['median']:.3f} max {bg['max']:.3f}"
        )
    print("  evidence_margin -> expected topic selected, topics per question")
    for point in summary["margin_sweep"]:
        print(f"  {point['margin']:.2f}: {point['any_selected']:.3f}, {point['mean_topics']:.2f}")
    for row in summary["rows"]:
        if not row["hit_at_1"]:
            gap_text = "no hit" if row["gap_to_best"] is None else f"gap {row['gap_to_best']:.3f}"
            print(
                f"  {row['id']}: expected {row['expected']} (rank {row['expected_rank']}, "
                f"{gap_text}), best {row['best']} ({row['best_score']})"
            )


def _print_routes(summary: Mapping[str, Any], cosine: bool = False) -> None:
    """`cosine`: the confidences are evidence (cosine) scores, not model confidences."""
    label = "primary cosine score" if cosine else "primary confidence"
    print(
        f"router: hit@1 {summary['hit_at_1']:.3f}, "
        f"expected topic selected {summary['any_selected']:.3f}, "
        f"{summary['mean_topics_selected']:.2f} topics per question "
        f"({summary['questions']} routed, {summary['with_secondary']} with a secondary topic); "
        f"{label} {summary['confidence_when_right']:.2f} when right, "
        f"{summary['confidence_when_wrong']:.2f} when wrong"
    )
    for row in summary["rows"]:
        if not row["hit_at_1"]:
            routed = ", ".join(
                f"{n} ({c:.2f})" for n, c in zip(row["routed"], row["confidences"], strict=True)
            )
            print(f"  {row['id']}: expected {row['expected']}, routed {routed}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        report = asyncio.run(run(args))
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # database or embedding service down, unknown profile, bad file
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        return EXIT_ABORTED
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport written to {args.report}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
