"""Calibrate the evidence router's out-of-domain signal (embeddings; a chat model with --analyze).

    uv run python -m agintis.cli.calibrate_evidence --profiles DIR --agents DIR \\
        --agent ID FILE.jsonl [FILE.jsonl ...] [--tenant T] [--analyze] [--report FILE]

Each JSON Lines row is `{"id", "question", "label": "in" | "out"}` plus an optional `category`
(default: the second segment of the id, `out-pens-01` -> `pens`). `in` = answerable from the
corpus, `out` = not (the ingested corpus of the tenant holds no answer). The files are DEV sets:
never the held-out eval questions.

The score measured. For every query the agent's own `EvidenceTopicRouter` (built from the spec:
same topics, `evidence_aggregate`, `probe_k`, `default_filter` scopes, one query embedding) gathers
the per-topic evidence through `ProbeProfiles`; the score of the query is the MAXIMUM over ALL
topics of `TopicEvidence.raw_score`, the topic's aggregate of its probe similarities WITHOUT the
`evidence_bias`. With the default `max` aggregate that is the best cosine similarity of any chunk
of any topic's profile. It is exactly the quantity a runtime uncertainty check can read from the
route step, before any margin / bias / selection logic: the margin only decides which extra topics
to search, and a per-topic bias is a routing correction, not a statement about "is this question
covered at all". Nothing is duplicated: the CLI calls `EvidenceTopicRouter.evidence`.

`--analyze` scores what the runtime routes: the agent's own `AnalyzeNode` (needs LLM_*, one call
per question) produces the standalone query and that is what is embedded; the report carries the
standalone queries and `query_input` says which input was scored ("raw_question" | "analyzed").

Thresholds: the pre-registered rules of `calibrate_stats`, over the `out` scores only:
`low_confidence_below` = p95 and `out_of_scope_below` = p50 (linear interpolation). The `in`
scores only feed the reported recall, AUROC and the `in` queries that fall below each threshold.
A query is "above" a threshold when its score is strictly greater. CAVEAT: the `in` queries were
written looking at the documents, so recall and AUROC are optimistic. A threshold is valid only
for the corpus snapshot, tenant, embedding model and profiles in the report `meta`: recalibrate
after a re-ingest. Changing a rule needs a fresh OOD set.

Exit codes: 0 ok; 2 usage (bad arguments, unknown agent or one without `router: evidence`, a
missing / unreadable / malformed labelled file, a file without both labels, a report path that
cannot be written, `--analyze` without LLM_*), all checked BEFORE the services boot; 3 abort
(services down, unknown profile, any other failure, including writing the report).
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import AnyMessage, HumanMessage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agintis.agent_kit.nodes import (
    AnalyzeNode,
    EvidenceTopicRouter,
    RouteScope,
    TopicEvidence,
)
from agintis.agent_kit.runtime import QueryVectors
from agintis.bootstrap.container import Container
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
from agintis.cli.calibrate_stats import (
    IN_LABEL,
    LABELS,
    OUT_LABEL,
    ScoredQuery,
    calibrate,
    scores_table,
)
from agintis.modules.agents.rag_chat import AgentSpec, build_router, load_agent_specs
from agintis.platform.llm import ChatModels, ChatRole, LlmNotConfiguredError
from agintis.platform.yaml_config import yaml_files
from agintis.shared_kernel.ids import TenantId

logger = logging.getLogger("agintis.cli.calibrate_evidence")

SCORE_DESCRIPTION = (
    "max over ALL the agent's topics of TopicEvidence.raw_score (the topic's evidence_aggregate "
    "of its probe_k similarities, without evidence_bias)"
)
QUERY_INPUT_RAW = "raw_question"
QUERY_INPUT_ANALYZED = "analyzed"
CAVEATS = (
    "The 'in' queries were written looking at the documents: in-domain recall and AUROC are "
    "optimistic.",
    "A threshold is valid only for the corpus snapshot, tenant, embedding model and profiles in "
    "meta: recalibrate after a re-ingest. Changing a rule needs a fresh OOD set.",
)
READY_CHUNKS_SQL = text(
    "SELECT c.profile, count(*) FROM chunks c "
    "JOIN documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id "
    "AND d.profile = c.profile "
    "WHERE d.status = 'ready' AND c.tenant_id = :tenant GROUP BY c.profile"
)


@dataclass(frozen=True)
class LabelledQuery:
    id: str
    question: str
    label: str
    category: str | None = None


class EvidenceSource(Protocol):
    """What the scoring needs (`EvidenceTopicRouter` satisfies it)."""

    async def evidence(self, query: str, scope: RouteScope | None) -> list[TopicEvidence]: ...


class Analyzer(Protocol):
    """What `--analyze` needs (`AnalyzeNode` satisfies it)."""

    async def analyze(self, messages: Sequence[AnyMessage]) -> dict[str, Any]: ...


def _text_field(row: Any, key: str, *, required: bool = True) -> str | None:
    if key not in row:
        if required:
            raise KeyError(key)
        return None
    value = row[key]
    if not isinstance(value, str):
        raise TypeError(f"{key!r} must be a string, got {type(value).__name__}")
    return value


def load_labelled(paths: Sequence[Path]) -> list[LabelledQuery]:
    """Rows of JSON Lines files (blank lines skipped). Ids must be unique across ALL files."""
    queries: list[LabelledQuery] = []
    seen: dict[str, str] = {}
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise UsageError(f"{path}: cannot read ({type(exc).__name__}: {exc})") from exc
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{number}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("a row must be a JSON object")
                query = LabelledQuery(
                    id=str(_text_field(row, "id")),
                    question=str(_text_field(row, "question")),
                    label=str(_text_field(row, "label")),
                    category=_text_field(row, "category", required=False),
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise UsageError(f"{where}: invalid row ({type(exc).__name__}: {exc})") from exc
            if query.label not in LABELS:
                raise UsageError(f"{where}: label must be one of {LABELS}, got {query.label!r}")
            if not query.question.strip():
                raise UsageError(f"{where}: blank question")
            if query.id in seen:
                raise UsageError(f"{where}: id {query.id!r} already used in {seen[query.id]}")
            seen[query.id] = where
            queries.append(query)
    return queries


def check_labels(queries: Sequence[LabelledQuery]) -> None:
    if not any(q.label == IN_LABEL for q in queries) or not any(
        q.label == OUT_LABEL for q in queries
    ):
        raise UsageError("the files need at least one 'in' and one 'out' query")


def check_report_path(path: Path | None) -> None:
    """Fail before booting anything when the report cannot possibly be written."""
    if path is None:
        return
    if path.is_dir():
        raise UsageError(f"--report {path} is a directory")
    if not path.parent.is_dir():
        raise UsageError(f"--report: the directory {path.parent} does not exist")


async def analyze_queries(analyzer: Analyzer, queries: Sequence[LabelledQuery]) -> dict[str, str]:
    """The analyzer's standalone query of every question (one LLM call each)."""
    return {
        q.id: str((await analyzer.analyze([HumanMessage(q.question)]))["standalone_query"])
        for q in queries
    }


async def score_queries(
    source: EvidenceSource,
    tenant: TenantId,
    queries: Sequence[LabelledQuery],
    texts: Mapping[str, str] | None = None,
) -> list[ScoredQuery]:
    """Score every query (one fresh vector memo each, so each one embeds once).

    `texts` maps a query id to the text to embed (the standalone query with `--analyze`);
    the raw question is scored where it has no entry."""
    scored: list[ScoredQuery] = []
    for query in queries:
        text_to_score = (texts or {}).get(query.id, query.question)
        evidence = await source.evidence(text_to_score, RouteScope(tenant, QueryVectors()))
        candidates = [(e.raw_score, e.topic) for e in evidence if e.raw_score is not None]
        if not candidates:
            raise RuntimeError(
                f"query {query.id!r}: no topic has any evidence (is the corpus ingested "
                f"for tenant {tenant!r}?)"
            )
        score, topic = max(candidates)
        scored.append(
            ScoredQuery(query.id, query.question, query.label, score, topic, query.category)
        )
    return scored


def build_report(
    scored: Sequence[ScoredQuery],
    meta: Mapping[str, Any],
    *,
    with_scores: bool = True,
    standalone: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "meta": dict(meta),
        "score": SCORE_DESCRIPTION,
        "caveats": list(CAVEATS),
        **calibrate(scored),
    }
    if standalone is not None:
        report["standalone_queries"] = dict(standalone)
    if with_scores:
        report["scores"] = scores_table(scored)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.calibrate_evidence",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", type=Path, help="JSON Lines labelled queries")
    parser.add_argument("--profiles", type=Path, required=True, help="directory of YAML profiles")
    parser.add_argument("--agents", type=Path, required=True, help="directory of YAML agent specs")
    parser.add_argument("--agent", required=True, help="agent id (its router must be 'evidence')")
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="tenant id (default: %(default)s)")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="score the AnalyzeNode's standalone query, like the runtime route step (needs LLM_*)",
    )
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    return parser


def pick_spec(specs: Sequence[AgentSpec], agent_id: str) -> AgentSpec:
    for spec in specs:
        if spec.id == agent_id:
            if not spec.uses_evidence_router:
                raise UsageError(f"agent {agent_id!r} does not use 'router: evidence'")
            return spec
    raise UsageError(f"unknown agent {agent_id!r} (known: {sorted(s.id for s in specs)})")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profiles_digest(directory: Path) -> dict[str, Any]:
    """Hash of the profile files (name + bytes, sorted) and of each one."""
    files = {path.name: _sha256(path) for path in yaml_files(directory)}
    combined = hashlib.sha256()
    for name, digest in files.items():
        combined.update(f"{name}\0{digest}\n".encode())
    return {"sha256": combined.hexdigest(), "files": files}


async def ready_chunk_counts(
    engine: AsyncEngine, tenant: str, profile_names: Sequence[str]
) -> dict[str, int]:
    """READY chunks per profile for the tenant (0 for a profile with none)."""
    async with engine.connect() as connection:
        rows = (await connection.execute(READY_CHUNKS_SQL, {"tenant": tenant})).all()
    found = {str(profile): int(count) for profile, count in rows}
    return {name: found.get(name, 0) for name in profile_names}


def _interval(interval: Mapping[str, Any]) -> str:
    def side(key: str, rank: str) -> str:
        value = interval[key]
        return "unbounded" if value is None else f"{value:.3f} (rank {interval[rank]})"

    return (
        f"{interval['confidence']:.0%} order-statistic interval "
        f"[{side('lower', 'lower_rank')}, {side('upper', 'upper_rank')}] of n={interval['n']}"
    )


def _rate(rate: Mapping[str, Any]) -> str:
    return (
        f"{rate['rate']:.3f} ({rate['k']}/{rate['n']}), {rate['confidence']:.0%} "
        f"Clopper-Pearson [{rate['lower']:.3f}, {rate['upper']:.3f}]"
    )


def _print_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        print(f"  {row['id']}: {row['score']:.3f} [{row['topic']}] {row['question']}")


def _print(report: Mapping[str, Any]) -> None:
    meta = report["meta"]
    print(f"score: {report['score']}")
    print(f"query input: {meta['query_input']}")
    print(f"quantiles: {report['quantile_method']} interpolation")
    columns = ("n", "min", "p5", "p25", "p50", "p75", "p95", "p99", "max", "mean")
    print(f"{'label':<6}" + "".join(f"{c:>8}" for c in columns))
    for label in (IN_LABEL, OUT_LABEL):
        dist = report["distributions"][label]
        cells = "".join(f"{int(dist[c]):>8}" if c == "n" else f"{dist[c]:>8.3f}" for c in columns)
        print(f"{label:<6}{cells}")
    print(f"\nAUROC (in vs out): {report['auroc']:.4f}")
    print(f"\nlow_confidence_below ({report['rule']}): {report['threshold']:.4f}")
    print(f"  {_interval(report['threshold_interval'])}")
    print(f"  in-domain recall above it: {_rate(report['in_recall_interval'])}")
    print(f"  out-of-domain false 'in' rate: {_rate(report['out_false_in_rate_interval'])}")
    print(f"\nout queries above low_confidence_below ({len(report['out_above'])}):")
    _print_rows(report["out_above"])
    print(f"\nin queries at or below low_confidence_below ({len(report['in_below'])}):")
    _print_rows(report["in_below"])
    oos = report["out_of_scope"]
    print(f"\nout_of_scope_below ({oos['rule']}): {oos['threshold']:.4f}")
    print(f"  {_interval(oos['threshold_interval'])}")
    print(f"  in queries refused (at or below it): {_rate(oos['in_refused_rate_interval'])}")
    _print_rows(oos["in_below"])
    print(
        "\nper stratum (label: n, min / median / max, above low_confidence, "
        "at or below out_of_scope):"
    )
    for category, by_label in report["strata"].items():
        for label, st in by_label.items():
            print(
                f"  {category:<10}{label:<4} n={st['n']:<3} {st['min']:.3f} / {st['median']:.3f} / "
                f"{st['max']:.3f}  above={st['above_low_confidence']} "
                f"below_oos={st['below_out_of_scope']}"
            )
    if "standalone_queries" in report:
        changed = sum(
            1
            for row in report["scores"]
            if report["standalone_queries"].get(row["id"]) != row["question"]
        )
        print(
            f"\nstandalone queries differing from the question: {changed}/{len(report['scores'])}"
        )
    print("\ncaveats:")
    for caveat in report["caveats"]:
        print(f"  - {caveat}")


def _settings(args: argparse.Namespace) -> Settings:
    settings = get_settings()
    return settings.model_copy(
        update={
            "knowledge": settings.knowledge.model_copy(update={"profiles_dir": args.profiles}),
            "database": settings.database.model_copy(update={"read_only": True}),
            "checkpointer": settings.checkpointer.model_copy(update={"enabled": False}),
            "agents": settings.agents.model_copy(update={"dir": None}),
        }
    )


def check_analyzer_available(settings: Settings, spec: AgentSpec) -> None:
    """`--analyze` needs an analyzer model BEFORE anything boots: `LLM_PROVIDER` or a profile
    bound to the analyzer under the spec's `models:`."""
    try:
        build_chat_models(settings).bind(
            spec.models, agent=spec.id, roles=(ChatRole.ANALYZER,)
        ).describe(ChatRole.ANALYZER)
    except LlmNotConfiguredError as exc:
        raise UsageError(
            "--analyze needs the LLM: set LLM_PROVIDER (and LLM_*) or bind the analyzer "
            "to a profile under `models:` in the spec"
        ) from exc


def agent_chat_models(chat_models: ChatModels, spec: AgentSpec, *, analyze: bool) -> ChatModels:
    """The models the agent itself resolves (its `models:` over the platform defaults), verified
    offline before any question. Only the analyzer is ever called, and only with `--analyze` (the
    router is the evidence one), so without it no binding is resolved: an analyzer profile whose
    `api_key_env` is unset does not matter."""
    roles = (ChatRole.ANALYZER,) if analyze else ()
    view = chat_models.bind(spec.models, agent=spec.id, roles=roles)
    view.verify()
    return view


async def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = load_agent_specs(args.agents)
    spec = pick_spec(specs, args.agent)
    queries = load_labelled(args.files)
    check_labels(queries)
    check_report_path(args.report)
    settings = _settings(args)
    if args.analyze:
        check_analyzer_available(settings, spec)
    async with Container.build(settings) as container:
        agent_models = agent_chat_models(container.chat_models, spec, analyze=args.analyze)
        router = build_router(
            spec,
            agent_models,
            probe=container.probe_profiles,
            profiles=container.profiles,
            max_query_chars=settings.knowledge.max_query_chars,
        )
        if not isinstance(router, EvidenceTopicRouter):  # pragma: no cover (pick_spec checked)
            raise UsageError(f"agent {spec.id!r} did not build an evidence router")
        standalone: dict[str, str] | None = None
        if args.analyze:
            analyzer = AnalyzeNode(
                agent_models,
                spec.prompts.to_prompt_set(citations=spec.citations).analyze,
                spec.max_history_messages,
            )
            print("analyzing questions (one analyzer call per question)...")
            standalone = await analyze_queries(analyzer, queries)
        scored = await score_queries(router, TenantId(args.tenant), queries, standalone)
        profile_names = [t.profile for t in spec.topics]
        meta: dict[str, Any] = {
            "git": git_revision(),
            "agent": spec.id,
            "tenant": args.tenant,
            "query_input": QUERY_INPUT_ANALYZED if args.analyze else QUERY_INPUT_RAW,
            "evidence_aggregate": spec.evidence_aggregate,
            "probe_k": spec.probe_k,
            "topics": [t.name for t in spec.topics],
            "profiles": profile_names,
            "ready_chunks": await ready_chunk_counts(container.engine, args.tenant, profile_names),
            "profile_fingerprints": {
                n: container.profiles.get(n).fingerprint for n in profile_names
            },
            "profiles_files": profiles_digest(args.profiles),
            "embeddings_model": settings.embeddings.model,
            "embeddings_query_prefix": settings.embeddings.query_prefix,
            "hnsw_ef_search": settings.database.hnsw_ef_search,
            "max_query_chars": settings.knowledge.max_query_chars,
            "files": {str(p): _sha256(p) for p in args.files},
        }
        if args.analyze:
            meta["analyzer_model"] = agent_models.describe(ChatRole.ANALYZER).label
    return build_report(scored, meta, standalone=standalone)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        report = asyncio.run(run(args))
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # services down, unknown profile, any other failure
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        return EXIT_ABORTED
    _print(report)
    if args.report is not None:
        try:
            args.report.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            print(
                f"aborted: cannot write the report ({type(exc).__name__}: {exc})", file=sys.stderr
            )
            return EXIT_ABORTED
        print(f"\nreport written to {args.report}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
