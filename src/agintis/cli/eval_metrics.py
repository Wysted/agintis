"""Retrieval-quality metrics and the relevance judgement shared by the eval CLI and the tests.

A retrieved chunk is RELEVANT to an eval question when its body contains the expected snippet
(whitespace- and case-insensitive; the first line of a chunk is its heading prefix and is not
searched) AND its metadata satisfies every entry of the question's locator. That is the same
rule the chunking acceptance test applies to prove each question has exactly one golden chunk;
here it decides whether retrieval surfaced it. Pure functions over plain data: no database.
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_KS = (1, 3, 5, 10)
# Question sets without a gold profile (e.g. a colloquial bank) group under this row.
NO_PROFILE = "none"


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    type: str
    profile: str
    expected_snippet: str
    locator: Mapping[str, Any]


def load_questions(path: Path) -> list[EvalQuestion]:
    """Questions of a JSON Lines file (blank lines skipped). KeyError names a missing field."""
    questions: list[EvalQuestion] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            locator = dict(row.get("locator", {}))
            validate_locator(str(row["id"]), locator)
            questions.append(
                EvalQuestion(
                    id=str(row["id"]),
                    question=str(row["question"]),
                    type=str(row.get("type", "")),
                    profile=str(row.get("profile", NO_PROFILE)),
                    expected_snippet=str(row["expected_snippet"]),
                    locator=locator,
                )
            )
    return questions


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def chunk_body(text: str) -> str:
    """The chunk without its first line (the heading prefix)."""
    return text.split("\n", 1)[-1]


def validate_locator(question_id: str, locator: Mapping[str, Any]) -> None:
    """Locator values are scalars (equality / list contains) or `{"prefix": "<string>"}`."""
    for key, wanted in locator.items():
        if not isinstance(wanted, Mapping):
            continue
        if set(wanted) != {"prefix"} or not isinstance(wanted["prefix"], str):
            raise ValueError(
                f"{question_id}: locator {key!r} must be a scalar or "
                f'{{"prefix": "<string>"}}, got {dict(wanted)!r}'
            )


def locator_matches(metadata: Mapping[str, Any], key: str, wanted: object) -> bool:
    """Does `metadata[key]` satisfy the locator entry?

    A scalar `wanted` means equality (list fields: contains the value). `{"prefix": p}` means the
    stored string starts with `p` (list fields: any item does); it is declared per question, so
    the matcher knows nothing about any case's keys.
    """
    have = metadata.get(key)
    if have is None:
        return False
    if isinstance(wanted, Mapping):
        prefix = str(wanted["prefix"])
        items = have if isinstance(have, tuple | list) else (have,)
        return any(isinstance(item, str) and item.startswith(prefix) for item in items)
    if isinstance(have, tuple | list):  # list fields: contains the value
        return wanted in have
    return bool(have == wanted)


def is_relevant(text: str, metadata: Mapping[str, Any], question: EvalQuestion) -> bool:
    if normalize(question.expected_snippet) not in normalize(chunk_body(text)):
        return False
    return all(locator_matches(metadata, key, want) for key, want in question.locator.items())


@dataclass(frozen=True)
class QuestionOutcome:
    question_id: str
    profile: str
    type: str
    rank: int | None  # 1-based rank of the first relevant hit; None = not retrieved
    retrieved: int  # hits returned
    # Filtered runs only: a stricter filter set found nothing and the ladder moved down.
    filters_relaxed: bool = False
    # Filtered runs: the filter set the result came from (`full`, `strong` or `none`).
    filter_level: str | None = None


def first_relevant_rank(
    hits: Iterable[tuple[str, Mapping[str, Any]]], question: EvalQuestion
) -> int | None:
    """1-based rank of the first relevant `(text, metadata)` hit, None when there is none."""
    for rank, (text, metadata) in enumerate(hits, start=1):
        if is_relevant(text, metadata, question):
            return rank
    return None


@dataclass(frozen=True)
class Summary:
    questions: int
    hit_at: Mapping[int, float]
    mrr: float
    mrr_depth: int  # MRR@depth: ranks beyond it were never retrieved (the largest k)


def summarize(outcomes: Sequence[QuestionOutcome], ks: Sequence[int]) -> Summary:
    """hit@k = share of questions whose first relevant hit ranks <= k; MRR@max(k) over the ranks
    returned (the retrieval depth is the largest k, so a miss contributes 0)."""
    n = len(outcomes)
    depth = max(ks)
    if n == 0:
        return Summary(0, {k: 0.0 for k in ks}, 0.0, depth)
    hit_at = {k: sum(1 for o in outcomes if o.rank is not None and o.rank <= k) / n for k in ks}
    mrr = sum(1.0 / o.rank for o in outcomes if o.rank is not None) / n
    return Summary(n, hit_at, mrr, depth)


def group_by_profile(outcomes: Sequence[QuestionOutcome]) -> dict[str, list[QuestionOutcome]]:
    grouped: dict[str, list[QuestionOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.profile, []).append(outcome)
    return dict(sorted(grouped.items()))


def format_table(outcomes: Sequence[QuestionOutcome], ks: Sequence[int]) -> str:
    """One row per profile plus an overall row."""
    rows = [*group_by_profile(outcomes).items(), ("OVERALL", list(outcomes))]
    width = max([len("profile"), *(len(name) for name, _ in rows)])
    header = (
        f"{'profile':{width}}  {'n':>3}  "
        + "  ".join(f"hit@{k:<3}" for k in ks)
        + f"  MRR@{max(ks)}"
    )
    lines = [header]
    for name, group in rows:
        summary = summarize(group, ks)
        cells = "  ".join(f"{summary.hit_at[k]:7.3f}" for k in ks)
        lines.append(f"{name:{width}}  {summary.questions:>3}  {cells}  {summary.mrr:.3f}")
    return "\n".join(lines)


def summary_to_dict(summary: Summary) -> dict[str, Any]:
    return {
        "questions": summary.questions,
        "hit_at": {str(k): round(v, 4) for k, v in summary.hit_at.items()},
        "mrr": round(summary.mrr, 4),
        "mrr_at": summary.mrr_depth,
    }


def changed_cases(
    plain: Sequence[QuestionOutcome], filtered: Sequence[QuestionOutcome]
) -> list[tuple[str, int | None, int | None]]:
    """`(question id, plain rank, filtered rank)` for every question whose rank differs."""
    by_id = {o.question_id: o for o in filtered}
    return [
        (o.question_id, o.rank, by_id[o.question_id].rank)
        for o in plain
        if o.question_id in by_id and by_id[o.question_id].rank != o.rank
    ]


def _rank_text(rank: int | None) -> str:
    return "miss" if rank is None else f"#{rank}"


def format_comparison(
    plain: Sequence[QuestionOutcome],
    filtered: Sequence[QuestionOutcome],
    ks: Sequence[int],
    after_label: str = "filters",
) -> str:
    """Overall metrics of the plain run next to the other one (`after_label`: filtered or routed),
    then the questions whose rank changed (`miss` = not retrieved within the largest k). The
    relaxation line only makes sense for filtered runs."""
    before, after = summarize(plain, ks), summarize(filtered, ks)
    lines = [f"{'':8}  {'plain':>7}  {after_label:>7}  {'delta':>7}"]
    rows = [(f"hit@{k}", before.hit_at[k], after.hit_at[k]) for k in ks]
    rows.append((f"MRR@{max(ks)}", before.mrr, after.mrr))
    lines += [f"{name:8}  {a:7.3f}  {b:7.3f}  {b - a:+7.3f}" for name, a, b in rows]
    if after_label == "filters":
        relaxed = [f"{o.question_id}->{o.filter_level}" for o in filtered if o.filters_relaxed]
        lines.append("filters relaxed on zero hits (to level): " + (", ".join(relaxed) or "-"))
    changes = changed_cases(plain, filtered)
    lines.append(
        "rank changed: "
        + (", ".join(f"{i} {_rank_text(a)}->{_rank_text(b)}" for i, a, b in changes) or "-")
    )
    return "\n".join(lines)
