"""Judge the answers of a dump with four binary quality judges (LLM-as-judge, offline).

    uv run python -m agintis.cli.judge --answers answers.jsonl [--judges a,b] [--profile NAME] \\
        [--concurrency N] [--allow-self-judge] [--report FILE]

Input: the dump of `python -m agintis.cli.dump_answers`. No database, no retrieval, no generator:
only the judge model is called, so a run is cheap to repeat and the judging is reproducible from
the dump. Judges (all binary, `verdict` true = good):

- `faithfulness`: the answer's atomic claims are ALL supported by the chunk texts the synthesis saw;
- `citation_support`: every sentence carrying [n] is supported by the sources it cites (ALCE
  style); a [n] pointing to a source that does not exist fails WITHOUT a model call;
- `answer_relevance`: the answer addresses the question;
- `refusal_correctness`: an expected out-of-scope question must be refused, an expected in-scope
  one must not be; no expected scope in the dump = not applicable. "Refused" = the scope event
  `out`, OR the documented text heuristic of `agent_kit.judge.refusal`, OR (when faithfulness /
  answer_relevance ran) the judge model's own `is_refusal`: then the method is `hybrid`; with a
  `--judges` subset without them it stays `deterministic`. The pass rate the heuristic alone would
  give is reported next to it (`refusal_correctness_heuristic` in the aggregate).

A judge that does not apply reports `not_applicable` with a counted reason (no citations, or
`citations_disabled` when an agent with `citations: none` answered without [n] markers; a refusal
- by the agent's scope verdict `out` or reported by the judge model itself -, no expected scope)
and is left out of the pass rate. The judge model is the LLM profile `EVAL_JUDGE_PROFILE`
(`--profile` overrides it; it has no default: name one, `default` = the `LLM_*` analyzer model is
allowed explicitly), checked offline for structured output before the first record. A judge whose
provider:model equals the model that wrote an answer is refused (self-judging) unless
`--allow-self-judge`. Concurrency is bounded (`EVAL_JUDGE_CONCURRENCY`, `--concurrency`)
over the model calls. An error on one record or judge (provider failure, unparseable output) is
recorded as `error` and never aborts the run; a record whose agent run failed in the dump is
`run_failed` (counted apart, it is not a judge error). Tracing (OBSERVABILITY_*) works as in the
app; judge runs carry the tag `agintis:judge` and the metadata
{agintis_component=answer_judge, judge, record_id, judged_agent, judge_model}.

The JSON report (`--report`) holds the provenance (git, dump hash, judge model), per-record
judgments with rationales and details (per-claim / per-sentence lists), and per judge the counts,
pass rate with a Clopper-Pearson interval and the token usage; feed it to `cli.agreement`. The
report is private (mode 0600: it quotes corpus text). While the run goes each judged record is
appended to `REPORT.partial` as one JSON line, so an abort or Ctrl-C keeps the paid work; the
complete report replaces it at the end.
DeepSeek note: it processes data in China, mind what the dump contains.

Exit codes: 0 ok (a measurement, not a gate: the numbers never change it); 1 some judgments
errored (the report is complete; `run_failed` alone does not count); 2 usage (arguments,
unreadable or malformed dump, unknown judge, no judge profile, a profile that is unknown,
unconfigured or cannot do structured output, self-judging), all checked BEFORE the first call;
3 abort.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agintis.agent_kit.judge import (
    AnswerJudge,
    AnswerRecord,
    JudgeName,
    JudgeResult,
    JudgeStatus,
    RecordFormatError,
    Usage,
    aggregate,
    build_judges,
    judge_meta,
    judge_record,
    record_from_json,
)
from agintis.bootstrap.llm import bind_judge_model, build_chat_models
from agintis.bootstrap.observability import (
    TRACING_SHUTDOWN_GRACE_SECONDS,
    build_tracing_from_settings,
    shutdown_tracing,
)
from agintis.bootstrap.settings import Settings, get_settings
from agintis.cli._common import (
    EXIT_ABORTED,
    EXIT_OK,
    EXIT_USAGE,
    PartialWriter,
    UsageError,
    check_output_path,
    file_sha256,
    git_revision,
    read_lines,
    write_private,
)
from agintis.cli.calibrate_stats import rate_interval
from agintis.platform.llm import ChatModels, ChatRole, LlmError
from agintis.platform.observability import Tracing, TracingError

logger = logging.getLogger("agintis.cli.judge")

EXIT_PARTIAL = 1
MAX_CONCURRENCY = 64
COLUMN_WIDTH = 22
SYMBOLS = {"pass": "pass", "fail": "FAIL", "n/a": "n/a", "error": "ERROR", "run": "run_failed"}


def load_dump(path: Path) -> list[AnswerRecord]:
    """The records of a dump (blank lines skipped); ids must be unique."""
    records: list[AnswerRecord] = []
    seen: dict[str, str] = {}
    for where, line in read_lines(path):
        try:
            record = record_from_json(json.loads(line))
        except (ValueError, RecordFormatError) as exc:
            raise UsageError(f"{where}: invalid record ({exc})") from exc
        if record.id in seen:
            raise UsageError(f"{where}: id {record.id!r} already used at {seen[record.id]}")
        seen[record.id] = where
        records.append(record)
    if not records:
        raise UsageError(f"{path}: no records")
    return records


def parse_judges(raw: str | None) -> list[JudgeName]:
    if raw is None:
        return list(JudgeName)
    names: list[JudgeName] = []
    for part in raw.split(","):
        try:
            name = JudgeName(part.strip())
        except ValueError:
            known = ", ".join(j.value for j in JudgeName)
            raise UsageError(f"unknown judge {part.strip()!r} (known: {known})") from None
        if name not in names:
            names.append(name)
    return names


async def run_judges(
    records: Sequence[AnswerRecord],
    judges: Sequence[AnswerJudge],
    sink: Callable[[AnswerRecord, Mapping[JudgeName, JudgeResult]], None] | None = None,
) -> list[dict[JudgeName, JudgeResult]]:
    """Every record through every judge; records run concurrently (the model calls are bounded
    by the semaphore the judges were built with), results come back in file order. `sink` sees
    each record as soon as it is judged (completion order)."""

    async def one(record: AnswerRecord) -> dict[JudgeName, JudgeResult]:
        result = await judge_record(record, judges)
        if sink is not None:
            sink(record, result)
        return result

    return list(await asyncio.gather(*(one(r) for r in records)))


def record_entry(
    record: AnswerRecord, judgments: Mapping[JudgeName, JudgeResult], names: Sequence[JudgeName]
) -> dict[str, Any]:
    """The report row of one judged record."""
    return {
        "id": record.id,
        "answer_sha256": record.answer_sha256,
        "agent": record.agent,
        "expected_scope": record.expected_scope,
        "judgments": {n.value: judgments[n].to_json() for n in names if n in judgments},
    }


def build_report(
    records: Sequence[AnswerRecord],
    judgments: Sequence[Mapping[JudgeName, JudgeResult]],
    names: Sequence[JudgeName],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    summary = aggregate(judgments, names)
    for stats in summary.values():
        stats["pass_rate_interval"] = (
            rate_interval(stats["passed"], stats["decided"]) if stats["decided"] else None
        )
    total = sum((r.usage for j in judgments for r in j.values()), Usage())
    return {
        "meta": dict(meta),
        "aggregate": summary,
        "usage": total.to_json(),
        "records": [
            record_entry(record, j, names) for record, j in zip(records, judgments, strict=True)
        ],
    }


def _print(report: Mapping[str, Any], names: Sequence[JudgeName]) -> None:
    meta = report["meta"]
    print(f"judge model: {meta['judge_model']} ({meta['judge_profile']})")
    print(f"\n{'record':<28}" + "".join(f"{n.value:>{COLUMN_WIDTH}}" for n in names))
    for row in report["records"]:
        cells = "".join(f"{_cell_of(row['judgments'].get(n.value)):>{COLUMN_WIDTH}}" for n in names)
        print(f"{row['id']:<28}{cells}")
    print("\npass rates (decided = judged with a verdict):")
    for name in names:
        s = report["aggregate"][name.value]
        interval = s["pass_rate_interval"]
        rate = (
            "n/a"
            if s["pass_rate"] is None
            else f"{s['pass_rate']:.3f} ({s['passed']}/{s['decided']}) "
            f"CP95 [{interval['lower']:.3f}, {interval['upper']:.3f}]"
        )
        print(
            f"  {name.value:<{COLUMN_WIDTH}} {rate}  not_applicable={s['not_applicable']} "
            f"errors={s['errors']} run_failed={s['run_failed']} "
            f"no_claims={s['reasons'].get('no_claims', 0)}"
        )
        aux = s["refusal_correctness_heuristic"]
        if aux is not None:
            print(
                f"  {'':<{COLUMN_WIDTH}} heuristic alone: {aux['passed']}/{aux['decided']} "
                f"({aux['pass_rate']:.3f})"
            )
        if s.get("model_refusals"):
            mr = s["model_refusals"]
            print(
                f"  {'':<{COLUMN_WIDTH}} model refusals: {mr['total']} "
                f"(cited={mr['cited_refusal']}, unchecked_claims={mr['unchecked_claims']})"
            )
        if s["reasons"]:
            print(f"  {'':<{COLUMN_WIDTH}} reasons: {s['reasons']}")
    usage = report["usage"]
    print(
        f"\ntokens: {usage['input_tokens']} in / {usage['output_tokens']} out / "
        f"{usage['total_tokens']} total over {usage['calls']} calls"
    )


def _cell_of(judgment: Mapping[str, Any] | None) -> str:
    if judgment is None:
        return "-"
    if judgment["status"] == JudgeStatus.OK:
        return SYMBOLS["pass" if judgment["verdict"] else "fail"]
    if judgment["status"] == JudgeStatus.RUN_FAILED:
        return SYMBOLS["run"]
    return SYMBOLS["n/a" if judgment["status"] == JudgeStatus.NOT_APPLICABLE else "error"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.judge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--answers", type=Path, required=True, help="the dump_answers JSONL")
    parser.add_argument(
        "--judges", help="comma-separated subset of: " + ", ".join(j.value for j in JudgeName)
    )
    parser.add_argument("--profile", help="LLM profile of the judge (default: EVAL_JUDGE_PROFILE)")
    parser.add_argument(
        "--concurrency",
        type=int,
        help=f"judge model calls in flight, 1..{MAX_CONCURRENCY} (default: EVAL_JUDGE_CONCURRENCY)",
    )
    parser.add_argument(
        "--allow-self-judge",
        action="store_true",
        help="accept a judge whose provider:model wrote the answers (biased; off by default)",
    )
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    return parser


def provider_model(label: str) -> str | None:
    """`provider:model` of a "profile (provider:model)" label (None when it has no such part)."""
    match = re.search(r"\((.+)\)$", label)
    return match.group(1) if match else None


def check_not_self_judging(records: Sequence[AnswerRecord], judge_label: str) -> list[str]:
    """The judge must not be the model that wrote the answers (it would grade its own work):
    `UsageError` when provider:model is the same. Returns warnings (not failures) for the
    softer case of the same provider FAMILY (e.g. both openai: judges tend to favour the models
    of their own family)."""
    judge = provider_model(judge_label)
    answering = {provider_model(label) for r in records for label in r.models.values()} - {None}
    if judge in answering:
        raise UsageError(
            f"the judge model {judge} also wrote the answers under judgment: pick another judge "
            "profile, or pass --allow-self-judge to accept the bias knowingly"
        )
    family = judge.split(":", 1)[0] if judge else None
    relatives = sorted(m for m in answering if m and family and m.split(":", 1)[0] == family)
    if relatives:
        return [
            f"the judge ({judge}) shares its provider family with the model(s) that wrote the "
            f"answers ({', '.join(relatives)}): judges tend to favour their own family"
        ]
    return []


def prepare_model(settings: Settings, profile: str) -> ChatModels:
    """The judge model's view, resolved and checked offline (usage error when it cannot serve)."""
    try:
        view = bind_judge_model(build_chat_models(settings), profile)
        view.verify()
    except LlmError as exc:
        raise UsageError(f"judge profile {profile!r}: {exc}") from exc
    return view


async def run(
    args: argparse.Namespace,
    settings: Settings,
    tracing: Tracing,
    partial: PartialWriter | None = None,
) -> dict[str, Any]:
    names = parse_judges(args.judges)
    concurrency = settings.eval.judge_concurrency if args.concurrency is None else args.concurrency
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise UsageError(f"--concurrency must be between 1 and {MAX_CONCURRENCY}")
    records = load_dump(args.answers)
    check_output_path(args.report, "--report")
    profile = args.profile or settings.eval.judge_profile
    if profile is None:
        raise UsageError(
            "no judge profile: set EVAL_JUDGE_PROFILE or pass --profile (a named LLM profile of "
            "LLM_PROFILES_FILE; there is no default, so the generator never judges itself by "
            "accident)"
        )
    view = prepare_model(settings, profile)
    try:
        tracing.start()
    except TracingError as exc:
        raise UsageError(str(exc)) from exc
    label = view.describe(ChatRole.ANALYZER).label
    if not args.allow_self_judge:
        for warning in check_not_self_judging(records, label):
            print(f"warning: {warning}", file=sys.stderr)
    judges = build_judges(
        view.get(ChatRole.ANALYZER), label, names, semaphore=asyncio.Semaphore(concurrency)
    )
    print(f"judging {len(records)} records with {label}...")

    def keep(record: AnswerRecord, judged: Mapping[JudgeName, JudgeResult]) -> None:
        if partial is not None:
            partial.write_line(json.dumps(record_entry(record, judged, names), ensure_ascii=False))

    judgments = await run_judges(records, judges, keep)
    meta = {
        "git": git_revision(),
        "answers": str(args.answers),
        "answers_sha256": file_sha256(args.answers),
        "records": len(records),
        "judges": [n.value for n in names],
        "judge_profile": profile,
        "judge_model": label,
        **judge_meta(),
        "concurrency": concurrency,
        "answered_by": sorted({r.agent for r in records}),
        "answer_models": {r.agent: dict(r.models) for r in records if r.models},
    }
    return build_report(records, judgments, names, meta)


async def amain(args: argparse.Namespace, partial: PartialWriter | None) -> dict[str, Any]:
    settings = get_settings()
    tracing = build_tracing_from_settings(settings)
    try:
        return await run(args, settings, tracing, partial)
    finally:
        await shutdown_tracing(
            tracing, settings.observability.otlp_timeout_seconds + TRACING_SHUTDOWN_GRACE_SECONDS
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    partial = None if args.report is None else PartialWriter(args.report)
    try:
        report = asyncio.run(amain(args, partial))
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (Exception, KeyboardInterrupt) as exc:  # any other failure, Ctrl-C included
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.debug("abort details", exc_info=True)
        if partial is not None and partial.partial.exists():
            print(f"the records already judged are kept in {partial.partial}", file=sys.stderr)
        return EXIT_ABORTED
    finally:
        if partial is not None:
            partial.close()
    _print(report, [JudgeName(n) for n in report["meta"]["judges"]])
    if args.report is not None:
        try:
            write_private(args.report, json.dumps(report, indent=2, ensure_ascii=False))
        except OSError as exc:
            print(
                f"aborted: cannot write the report ({type(exc).__name__}: {exc})", file=sys.stderr
            )
            return EXIT_ABORTED
        print(f"\nreport written to {args.report}")
    errors = sum(s["errors"] for s in report["aggregate"].values())
    return EXIT_PARTIAL if errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
