"""Run an agent over a questions file and dump one JSON Lines record per question.

    uv run python -m agintis.cli.dump_answers --profiles DIR --agents DIR --agent ID \\
        --questions FILE.jsonl --out answers.jsonl [--tenant T] [--limit N] [--concurrency N] \\
        [--default-scope in|out]

The dump is the input of `python -m agintis.cli.judge`: judging runs offline over it, so it is
reproducible and cheap to repeat (no database, no retrieval, no generator). Each questions row is
`{"id", "question"}` plus an optional reference scope, `label` (`in` | `out`, the dev sets of
`case/<name>/calibration`) or `expected_scope`; rows without one get `--default-scope` (none by
default, which makes the `refusal_correctness` judge not applicable to them). A multi-turn row
also has `history`: `[{"role": "user" | "assistant", "content": "..."}, ...]`, the earlier turns
in conversation order, sent to the agent before `question` (the follow-up that is answered).
Roles are not checked for alternation. shortcut: the record keeps `question` but not `history`, so
the judge and the labelers see a follow-up without its context; carry `history` in the record
before judging a multi-turn set.

The agent runs through its own compiled graph (same nodes, same models, same limits as the
runtime; no checkpointer, one fresh conversation per question) against the configured database
READ-ONLY.
A record holds the final answer, the clarification policy's scope verdict, the sources EXACTLY as
the synthesis numbered and saw them (n, title, uri, chunk text) and the models behind the run as
"profile (provider:model)" labels, plus the LLM token usage of the question per model (every call
of the run, also of a failed one up to the failure; the totals are printed at the end); see
`agintis.agent_kit.judge.records` for the schema. A question whose run fails is written with its
`error` (the failure class) and an empty answer: one failure never aborts the dump.

The dump is written incrementally, in question order, to `OUT.partial` (mode 0600: it holds
retrieved document text and model output) and renamed to OUT when the run finishes, so an abort
or Ctrl-C keeps what was already paid for in the `.partial` file. It stops with exit 3 when an
infrastructure failure repeats (the first records all fail the same way, or more than half of
the records so far fail with one error class): nothing is gained by asking the remaining
questions. Keep dumps in a gitignored place such as `case/<name>/judge/` (ignored by
`.gitignore`): they contain corpus text.

Exit codes: 0 ok; 1 some questions failed (the dump is complete); 2 usage (arguments, unknown
agent, unreadable or malformed questions file, no LLM for the agent, output path), all checked
BEFORE the services boot; 3 abort (services down, a repeated failure, Ctrl-C, any other failure).
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agintis.agent_kit.judge import (
    EXPECTED_SCOPES,
    AnswerRecord,
    ExpectedScope,
    ModelUsageCounter,
    failure_reason,
    format_usage,
    record_from_state,
    total_usage,
)
from agintis.agent_kit.nodes import CitationStyle, SourceLimits
from agintis.agent_kit.runtime import DEFAULT_RECURSION_LIMIT, AgentContext
from agintis.bootstrap.container import Container
from agintis.bootstrap.llm import build_chat_models
from agintis.bootstrap.settings import Settings, get_settings
from agintis.cli._common import (
    DEFAULT_TENANT,
    EXIT_ABORTED,
    EXIT_OK,
    EXIT_USAGE,
    PartialWriter,
    UsageError,
    check_output_path,
    file_sha256,
    git_revision,
    read_jsonl,
)
from agintis.cli._ledger import record_usage
from agintis.modules.agents.rag_chat import (
    AgentSpec,
    RagChatDeps,
    agent_deps,
    build_graph,
    load_agent_specs,
    roles_used,
    validate_agent_specs,
)
from agintis.platform.llm import ChatModels, LlmError, LlmNotConfiguredError
from agintis.platform.usage import Purpose
from agintis.shared_kernel.ids import RunId, TenantId, new_id

logger = logging.getLogger("agintis.cli.dump_answers")

EXIT_PARTIAL = 1
FAIL_FAST_AFTER = 3  # records all failing the same way abort the dump...
MIN_SAMPLE_FOR_MAJORITY = 5  # ...and so does one error class on more than half of >= 5
DEFAULT_CONCURRENCY = 1
MAX_CONCURRENCY = 16
DUMP_TAG = "agintis:dump-answers"


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    expected_scope: ExpectedScope | None = None
    history: tuple[tuple[str, str], ...] = ()  # (role, content) turns before `question`


HISTORY_ROLES = ("user", "assistant")


class AnswerGraph(Protocol):
    """What the dump needs from a compiled graph (`Pregel` satisfies it)."""

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, *, context: Any = None
    ) -> Any: ...


def _history(where: str, raw: object) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise UsageError(f"{where}: 'history' must be a list")
    turns: list[tuple[str, str]] = []
    for i, turn in enumerate(raw):
        if not isinstance(turn, dict):
            raise UsageError(f"{where}: history[{i}] must be an object")
        role, content = turn.get("role"), turn.get("content")
        if role not in HISTORY_ROLES:
            raise UsageError(f"{where}: history[{i}] role must be one of {HISTORY_ROLES}")
        if not isinstance(content, str) or not content.strip():
            raise UsageError(f"{where}: history[{i}] content must be a non-empty string")
        turns.append((role, content))
    return tuple(turns)


def load_questions(path: Path, default_scope: ExpectedScope | None = None) -> list[Question]:
    """Rows of a JSON Lines file (blank lines skipped); ids must be unique."""
    questions: list[Question] = []
    seen: dict[str, str] = {}
    for where, row in read_jsonl(path):
        if not isinstance(row, dict):
            raise UsageError(f"{where}: a row must be a JSON object")
        ident, text = row.get("id"), row.get("question")
        if not isinstance(ident, str) or not ident or not isinstance(text, str) or not text.strip():
            raise UsageError(f"{where}: 'id' and 'question' must be non-empty strings")
        if ident in seen:
            raise UsageError(f"{where}: id {ident!r} already used at {seen[ident]}")
        seen[ident] = where
        scope = row.get("expected_scope", row.get("label"))
        if scope is not None and scope not in EXPECTED_SCOPES:
            raise UsageError(
                f"{where}: scope label must be one of {EXPECTED_SCOPES}, got {scope!r}"
            )
        questions.append(
            Question(
                ident,
                text,
                scope if scope is not None else default_scope,
                _history(where, row.get("history")),
            )
        )
    if not questions:
        raise UsageError(f"{path}: no questions")
    return questions


async def answer_question(
    graph: AnswerGraph,
    question: Question,
    *,
    tenant: TenantId,
    agent: str,
    limits: SourceLimits,
    models: Mapping[str, str],
    meta: Mapping[str, Any],
    citations: CitationStyle,
    counter: ModelUsageCounter | None = None,
) -> AnswerRecord:
    """One fresh conversation (the history, then the question); a failure becomes a record with
    `error` (never raises). `counter` collects the run's LLM usage (the caller may keep it to read
    the spend of a run that gets cancelled)."""
    counter = counter or ModelUsageCounter()  # sees every LLM call of the run, subgraphs too
    config: RunnableConfig = {
        "callbacks": [counter],
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
        "tags": [DUMP_TAG],
        "metadata": {"agintis_component": "dump_answers", "record_id": question.id, "agent": agent},
    }
    messages: list[BaseMessage] = [
        HumanMessage(content) if role == "user" else AIMessage(content)
        for role, content in question.history
    ]
    messages.append(HumanMessage(question.question))
    try:
        state = await graph.ainvoke(
            {"messages": messages},
            config,
            context=AgentContext(tenant_id=tenant, run_id=RunId(new_id())),
        )
    except Exception as exc:
        logger.warning("question %s failed: %s", question.id, type(exc).__name__)
        return AnswerRecord(
            id=question.id,
            question=question.question,
            answer="",
            agent=agent,
            expected_scope=question.expected_scope,
            models=dict(models),
            meta=dict(meta),
            error=failure_reason(exc),
            usage=counter.usage,  # what was consumed before the failure
            citations=citations,
        )
    return record_from_state(
        state,
        id=question.id,
        question=question.question,
        agent=agent,
        limits=limits,
        expected_scope=question.expected_scope,
        models=models,
        meta=meta,
        citations=citations,
        usage=counter.usage,
    )


class RepeatedFailureError(RuntimeError):
    """The same failure keeps repeating: the infrastructure is down, stop asking."""


def repeated_failure(errors: Sequence[str | None]) -> str | None:
    """The error class that makes continuing pointless, given the outcome of the records so
    far (None = ok): all of the first `FAIL_FAST_AFTER` or more failed the same way, or one class
    covers more than half of at least `MIN_SAMPLE_FOR_MAJORITY` records. Rate limits (HTTP 429,
    `RateLimitError`) count as infrastructure failures like a database or embeddings outage:
    asking on only burns the quota and the time."""
    counts = Counter(e for e in errors if e is not None)
    if not counts:
        return None
    error, count = counts.most_common(1)[0]
    if len(errors) >= FAIL_FAST_AFTER and count == len(errors):
        return error
    if len(errors) >= MIN_SAMPLE_FOR_MAJORITY and count * 2 > len(errors):
        return error
    return None


async def dump_answers(
    graph: AnswerGraph,
    questions: Sequence[Question],
    *,
    tenant: TenantId,
    agent: str,
    limits: SourceLimits,
    models: Mapping[str, str],
    meta: Mapping[str, Any],
    citations: CitationStyle,
    concurrency: int = DEFAULT_CONCURRENCY,
    sink: Callable[[AnswerRecord], None] | None = None,
    counters: list[ModelUsageCounter] | None = None,
) -> list[AnswerRecord]:
    """Every question, at most `concurrency` at a time; records come back (and reach `sink`) in
    file order. `RepeatedFailureError` (the finished records are already in the sink) when the
    same failure repeats. Each question's usage counter is appended to `counters` when it starts,
    so the spend of questions cancelled by an abort stays readable."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(question: Question) -> AnswerRecord:
        async with semaphore:
            counter = ModelUsageCounter()
            if counters is not None:
                counters.append(counter)
            return await answer_question(
                graph,
                question,
                tenant=tenant,
                agent=agent,
                limits=limits,
                models=models,
                meta=meta,
                citations=citations,
                counter=counter,
            )

    tasks = [asyncio.create_task(one(q)) for q in questions]
    records: list[AnswerRecord] = []
    try:
        for task in tasks:
            record = await task
            records.append(record)
            if sink is not None:
                sink(record)
            failing = repeated_failure([r.error for r in records])
            if failing is not None:
                raise RepeatedFailureError(
                    f"{failing} repeated over {len(records)} questions: the infrastructure looks "
                    "down, stopping"
                )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return records


class DumpWriter:
    """Appends records to `PATH.partial` (0600), one flushed line each, and renames it to PATH on
    `commit`. Without a commit the partial file stays: it is what an abort already paid for."""

    def __init__(self, path: Path) -> None:
        self._out = PartialWriter(path)
        self.path = path
        self.partial = self._out.partial

    def write(self, record: AnswerRecord) -> None:
        self._out.write_line(json.dumps(record.to_json(), ensure_ascii=False))

    def close(self) -> None:
        self._out.close()

    def commit(self) -> None:
        self._out.commit()


def write_dump(path: Path, records: Sequence[AnswerRecord]) -> None:
    """The whole dump at once (same file mode and rename as the incremental writer)."""
    writer = DumpWriter(path)
    try:
        for record in records:
            writer.write(record)
        writer.commit()
    finally:
        writer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.dump_answers",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profiles", type=Path, required=True, help="directory of YAML profiles")
    parser.add_argument("--agents", type=Path, required=True, help="directory of YAML agent specs")
    parser.add_argument("--agent", required=True, help="agent id")
    parser.add_argument("--questions", type=Path, required=True, help="JSON Lines questions")
    parser.add_argument("--out", type=Path, required=True, help="the answers dump to write")
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="tenant id (default: %(default)s)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"questions in flight at once, 1..{MAX_CONCURRENCY} (default: %(default)s)",
    )
    parser.add_argument(
        "--default-scope",
        choices=EXPECTED_SCOPES,
        help="expected scope of the rows that have none (default: none)",
    )
    return parser


def check_args(args: argparse.Namespace) -> None:
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        raise UsageError(f"--concurrency must be between 1 and {MAX_CONCURRENCY}")
    if args.limit is not None and args.limit < 1:
        raise UsageError("--limit must be at least 1")
    check_output_path(args.out, "--out")


def pick_spec(specs: Sequence[AgentSpec], agent_id: str) -> AgentSpec:
    for spec in specs:
        if spec.id == agent_id:
            return spec
    raise UsageError(f"unknown agent {agent_id!r} (known: {sorted(s.id for s in specs)})")


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


def check_models_available(settings: Settings, spec: AgentSpec) -> None:
    """The agent's roles must resolve (LLM_PROVIDER or bound profiles) BEFORE anything boots."""
    try:
        view = build_chat_models(settings).bind(spec.models, agent=spec.id, roles=roles_used(spec))
        for role in roles_used(spec):
            view.describe(role)
    except LlmNotConfiguredError as exc:
        raise UsageError(
            f"agent {spec.id!r} needs the LLM: set LLM_PROVIDER (and LLM_*) or bind its roles "
            "to profiles under `models:` in the spec"
        ) from exc
    except LlmError as exc:
        raise UsageError(str(exc)) from exc


def model_labels(view: ChatModels, spec: AgentSpec) -> dict[str, str]:
    return {role.value: view.describe(role).label for role in roles_used(spec)}


async def run(
    args: argparse.Namespace, writer: DumpWriter, counters: list[ModelUsageCounter]
) -> list[AnswerRecord]:
    specs = load_agent_specs(args.agents)
    spec = pick_spec(specs, args.agent)
    questions = load_questions(args.questions, args.default_scope)
    if args.limit is not None:
        questions = questions[: args.limit]
    check_args(args)
    settings = _settings(args)
    check_models_available(settings, spec)
    async with Container.build(settings) as container:
        validate_agent_specs([spec], container.profiles)
        deps = RagChatDeps(
            container.chat_models,
            container.retrieve_for_topic,
            settings.knowledge.max_query_chars,
            container.probe_profiles,
        )
        view = agent_deps(spec, deps).chat_models
        view.verify()  # offline: builds the agent's models, checks the structured-output roles
        graph = build_graph(spec, container.profiles, deps)
        meta = {
            "git": git_revision(),
            "tenant": args.tenant,
            "questions_sha256": file_sha256(args.questions),
        }
        print(f"asking {len(questions)} questions to {spec.id!r}...")
        return await dump_answers(
            graph,
            questions,
            tenant=TenantId(args.tenant),
            agent=spec.id,
            limits=spec.limits,
            models=model_labels(view, spec),
            meta=meta,
            citations=spec.citations,
            concurrency=args.concurrency,
            sink=writer.write,
            counters=counters,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    writer = DumpWriter(args.out)
    counters: list[ModelUsageCounter] = []  # the run's, to report the spend of an abort
    try:
        records = asyncio.run(run(args, writer, counters))
        writer.commit()
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (Exception, KeyboardInterrupt) as exc:  # services down, repeated failure, Ctrl-C...
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        if counters:
            spend = format_usage(total_usage(c.usage for c in counters))
            print(f"aborted run, {spend} (written rows plus questions in flight)", file=sys.stderr)
        if writer.partial.exists():
            print(f"the records already answered are kept in {writer.partial}", file=sys.stderr)
        return EXIT_ABORTED
    finally:
        writer.close()
        # Aborted runs too: the questions already asked were billed.
        record_usage(
            total_usage(c.usage for c in counters),
            Purpose.DUMP,
            f"dump_answers {args.questions.name}",
            tenant_id=args.tenant,
        )
    failed = [r.id for r in records if r.error is not None]
    refused = sum(1 for r in records if r.scope is not None and r.scope.level == "out")
    print(f"{len(records)} answers written to {args.out} ({refused} refused by the scope check)")
    print(format_usage(total_usage(r.usage for r in records)))
    if failed:
        print(f"{len(failed)} questions failed: {', '.join(failed[:10])}", file=sys.stderr)
        return EXIT_PARTIAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
