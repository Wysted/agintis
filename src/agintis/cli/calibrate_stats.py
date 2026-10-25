"""Statistics of the evidence-score calibration (pure functions, no I/O).

The question: a score `s(q)` (the evidence router's best probe similarity) is meant to tell
in-domain queries (`in`) from out-of-domain ones (`out`). Both thresholds are fixed by rules written
BEFORE looking at the data, over the `out` scores only:
- `low_confidence_below` = the 95th percentile (`THRESHOLD_RULE`): about 5% of out-of-domain queries
  are (wrongly) called in-domain;
- `out_of_scope_below` = the median (`OUT_OF_SCOPE_RULE`).
The `in` queries are used ONLY to report recall, AUROC and how many `in` fall below each threshold,
never to choose them.

Conventions (all documented in the report):
- Quantiles use linear interpolation between the closest ranks (numpy's default `linear`).
- A query is "above" the threshold when `score > threshold` (strict), so a query exactly on it is
  treated as out-of-domain.
- AUROC = P(score(in) > score(out)) + 0.5 * P(tie), computed from mid-ranks.
- Rates get an exact Clopper-Pearson interval; a quantile threshold gets the distribution-free
  order-statistic interval (the binomial law of how many observations fall below the true
  quantile). Both at `CONFIDENCE`; a quantile side the sample cannot bound is reported as None.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

IN_LABEL = "in"
OUT_LABEL = "out"
LABELS = (IN_LABEL, OUT_LABEL)
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
THRESHOLD_QUANTILE = 0.95
THRESHOLD_RULE = "p95 of the 'out' scores (pre-registered); 'in' only reports recall"
OUT_OF_SCOPE_QUANTILE = 0.50
OUT_OF_SCOPE_RULE = "p50 of the 'out' scores (pre-registered); 'in' only reports"
QUANTILE_METHOD = "linear"
CONFIDENCE = 0.95
BISECTION_STEPS = 100


@dataclass(frozen=True)
class ScoredQuery:
    """One labelled query and the score the runtime would compare against the threshold."""

    id: str
    question: str
    label: str
    score: float
    topic: str | None = None  # the topic that produced the best score (diagnostics only)
    category: str | None = None  # stratum; None = derived from the id (`category_of`)


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """The `q` quantile (0..1) of already sorted values, linear interpolation."""
    if not sorted_values:
        raise ValueError("quantile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be within [0, 1]")
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


def category_of(query_id: str) -> str:
    """The stratum of an id: `out-bill-01` -> `bill`, `in-ref-20` -> `ref` (else the id itself)."""
    parts = query_id.split("-")
    return parts[1] if len(parts) >= 3 else parts[0]


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k + 1)))


def _bisect(decreasing: bool, target: float, f: Callable[[float], float]) -> float:
    """The p in [0, 1] where the monotone `f(p)` crosses `target`."""
    low, high = 0.0, 1.0
    for _ in range(BISECTION_STEPS):
        mid = (low + high) / 2
        if (f(mid) > target) == decreasing:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def clopper_pearson(successes: int, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Exact two-sided binomial interval for `successes` out of `n` (conservative)."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("need n >= 1 and 0 <= successes <= n")
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else _bisect(False, alpha / 2, lambda p: 1.0 - _binom_cdf(successes - 1, n, p))
    )
    upper = (
        1.0 if successes == n else _bisect(True, alpha / 2, lambda p: _binom_cdf(successes, n, p))
    )
    return lower, upper


def quantile_interval(
    sorted_values: Sequence[float], q: float, confidence: float = CONFIDENCE
) -> dict[str, Any]:
    """Order-statistic interval of the `q` quantile: [x_(lower_rank), x_(upper_rank)].

    With N = how many observations fall below the true quantile, N ~ Binomial(n, q), and
    P(x_(l) <= quantile <= x_(u)) = P(l <= N <= u - 1). The ranks are the tightest that leave
    at most (1 - confidence) / 2 in each tail; a side the sample is too small to bound is None.
    `coverage` is the achieved probability (>= confidence when both sides exist).
    """
    n = len(sorted_values)
    if n < 1:
        raise ValueError("interval of an empty sample")
    tail = (1.0 - confidence) / 2
    lower_rank = max((r for r in range(1, n + 1) if _binom_cdf(r - 1, n, q) <= tail), default=None)
    upper_rank = min(
        (r for r in range(1, n + 1) if _binom_cdf(r - 1, n, q) >= 1.0 - tail), default=None
    )
    low_cdf = 0.0 if lower_rank is None else _binom_cdf(lower_rank - 1, n, q)
    high_cdf = 1.0 if upper_rank is None else _binom_cdf(upper_rank - 1, n, q)
    return {
        "confidence": confidence,
        "lower": None if lower_rank is None else sorted_values[lower_rank - 1],
        "upper": None if upper_rank is None else sorted_values[upper_rank - 1],
        "lower_rank": lower_rank,
        "upper_rank": upper_rank,
        "n": n,
        "coverage": high_cdf - low_cdf,
    }


def rate_interval(successes: int, n: int, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """`{k, n, rate, lower, upper}` with the Clopper-Pearson interval."""
    lower, upper = clopper_pearson(successes, n, confidence)
    return {
        "k": successes,
        "n": n,
        "rate": successes / n,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
    }


def distribution(scores: Sequence[float]) -> dict[str, float]:
    """n, min, max, mean and the standard quantiles (`p5`...`p99`) of a sample."""
    if not scores:
        raise ValueError("distribution of an empty sample")
    ordered = sorted(scores)
    stats: dict[str, float] = {
        "n": float(len(ordered)),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }
    for q in QUANTILES:
        stats[f"p{round(q * 100)}"] = quantile(ordered, q)
    return stats


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """P(a positive scores higher than a negative), ties counting half (mid-rank Mann-Whitney)."""
    if not positives or not negatives:
        raise ValueError("AUROC needs at least one positive and one negative")
    pooled = sorted([(s, True) for s in positives] + [(s, False) for s in negatives])
    rank_sum = 0.0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        mid_rank = (i + j) / 2 + 1  # ranks are 1-based
        rank_sum += mid_rank * sum(1 for k in range(i, j + 1) if pooled[k][1])
        i = j + 1
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _scores(queries: Sequence[ScoredQuery], label: str) -> list[float]:
    return [q.score for q in queries if q.label == label]


def _row(query: ScoredQuery) -> dict[str, Any]:
    return {"id": query.id, "score": query.score, "topic": query.topic, "question": query.question}


def _stratum(rows: Sequence[ScoredQuery], low: float, out_of_scope: float) -> dict[str, Any]:
    scores = sorted(q.score for q in rows)
    return {
        "n": len(scores),
        "min": scores[0],
        "median": quantile(scores, 0.5),
        "max": scores[-1],
        "above_low_confidence": sum(1 for x in scores if x > low),
        "below_out_of_scope": sum(1 for x in scores if x <= out_of_scope),
    }


def strata(queries: Sequence[ScoredQuery], low: float, out_of_scope: float) -> dict[str, Any]:
    """Per category and label: n, min, median, max and how many sit above / below the thresholds."""
    groups: dict[str, dict[str, list[ScoredQuery]]] = {}
    for q in queries:
        category = q.category or category_of(q.id)
        groups.setdefault(category, {}).setdefault(q.label, []).append(q)
    return {
        category: {label: _stratum(rows, low, out_of_scope) for label, rows in sorted(by.items())}
        for category, by in sorted(groups.items())
    }


def calibrate(queries: Sequence[ScoredQuery]) -> dict[str, Any]:
    """The calibration report body for labelled, scored queries.

    Needs at least one query of each label. `low_confidence_below` (also `threshold`) is the p95 of
    the `out` scores; `in_recall` the fraction of `in` queries strictly above it,
    `out_false_in_rate` the fraction of `out` queries strictly above it (about 5% by
    construction, more or less with ties). `out_of_scope_below` is the p50 of the `out` scores,
    with the `in` queries at or below it listed (they would be refused: the runtime refuses at
    `score <= out_of_scope_below`).
    """
    unknown = {q.label for q in queries} - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    if any(not math.isfinite(q.score) for q in queries):
        raise ValueError("scores must be finite")
    ins, outs = _scores(queries, IN_LABEL), _scores(queries, OUT_LABEL)
    if not ins or not outs:
        raise ValueError("calibration needs at least one 'in' and one 'out' query")
    ordered_outs = sorted(outs)
    threshold = quantile(ordered_outs, THRESHOLD_QUANTILE)
    out_of_scope = quantile(ordered_outs, OUT_OF_SCOPE_QUANTILE)
    out_above = sorted(
        (q for q in queries if q.label == OUT_LABEL and q.score > threshold),
        key=lambda q: (-q.score, q.id),
    )
    in_below = sorted(
        (q for q in queries if q.label == IN_LABEL and q.score <= threshold),
        key=lambda q: (q.score, q.id),
    )
    in_refused = sorted(
        (q for q in queries if q.label == IN_LABEL and q.score <= out_of_scope),
        key=lambda q: (q.score, q.id),
    )
    return {
        "rule": THRESHOLD_RULE,
        "quantile_method": QUANTILE_METHOD,
        "confidence": CONFIDENCE,
        "distributions": {IN_LABEL: distribution(ins), OUT_LABEL: distribution(outs)},
        "auroc": auroc(ins, outs),
        "threshold": threshold,
        "threshold_interval": quantile_interval(ordered_outs, THRESHOLD_QUANTILE),
        "in_recall": (len(ins) - len(in_below)) / len(ins),
        "in_recall_interval": rate_interval(len(ins) - len(in_below), len(ins)),
        "out_false_in_rate": len(out_above) / len(outs),
        "out_false_in_rate_interval": rate_interval(len(out_above), len(outs)),
        "out_above": [_row(q) for q in out_above],
        "in_below": [_row(q) for q in in_below],
        "out_of_scope": {
            "rule": OUT_OF_SCOPE_RULE,
            "threshold": out_of_scope,
            "threshold_interval": quantile_interval(ordered_outs, OUT_OF_SCOPE_QUANTILE),
            "in_refused_rate": len(in_refused) / len(ins),
            "in_refused_rate_interval": rate_interval(len(in_refused), len(ins)),
            "in_below": [_row(q) for q in in_refused],
        },
        "strata": strata(queries, threshold, out_of_scope),
    }


def scores_table(queries: Sequence[ScoredQuery]) -> list[Mapping[str, Any]]:
    """Every query with its score, highest first (for the JSON report)."""
    ordered = sorted(queries, key=lambda q: (-q.score, q.id))
    return [{**_row(q), "label": q.label} for q in ordered]
