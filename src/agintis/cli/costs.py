"""LLM cost report from the usage ledger (`llm_usage`).

    uv run python -m agintis.cli.costs report [--since DATE] [--until DATE]
        [--by purpose|day|tenant|thread|model] [--strict]

Tokens come from the ledger; USD is computed NOW with the price file (`LLM_PRICES_FILE`, default
`platform/usage/prices.yaml`), so a corrected price re-prices history. Rows the file cannot
price (unknown model, older than the model's first entry) are listed apart with their tokens and
never counted as $0. `--since` / `--until` are UTC dates, both inclusive.

Exit codes: 0 ok; 1 `--strict` and some rows are unpriced; 2 bad arguments or price file;
3 aborted (database unavailable).
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

from agintis.bootstrap.settings import get_settings
from agintis.cli._common import EXIT_ABORTED, EXIT_OK, EXIT_USAGE
from agintis.cli._ledger import usage_repository
from agintis.platform.usage import DEFAULT_PRICES_FILE, PriceError, load_price_catalog
from agintis.platform.usage.report import GroupBy, Line, Report, build_report

EXIT_UNPRICED = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agintis.cli.costs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="USD and tokens per group")
    report.add_argument("--since", type=date.fromisoformat, metavar="DATE")
    report.add_argument("--until", type=date.fromisoformat, metavar="DATE")
    report.add_argument("--by", type=GroupBy, choices=list(GroupBy), default=GroupBy.PURPOSE)
    report.add_argument("--strict", action="store_true", help="exit 1 if any row is unpriced")
    return parser


def _row(key: str, line: Line, usd: str) -> str:
    return (
        f"{key:<40} {usd:>12} {line.input_tokens:>14,} {line.cached_input_tokens:>14,} "
        f"{line.output_tokens:>12,} {line.calls:>8,}"
    )


def render(report: Report, by: GroupBy) -> str:
    lines = [f"{by.value:<40} {'USD':>12} {'INPUT':>14} {'CACHED':>14} {'OUTPUT':>12} {'CALLS':>8}"]
    lines += [_row(key, line, f"{line.cost_usd:.4f}") for key, line in report.priced.items()]
    lines.append(f"{'TOTAL (priced)':<40} {report.total_usd:>12.4f}")
    if report.unpriced:
        lines += [
            "",
            "UNPRICED (no price for the model at that date, or no token data; tokens only):",
        ]
        lines += [_row(model, line, "-") for model, line in report.unpriced.items()]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> Report:
    settings = get_settings()
    catalog = load_price_catalog(settings.llm.prices_file or DEFAULT_PRICES_FILE)
    since = None if args.since is None else datetime.combine(args.since, datetime.min.time(), UTC)
    until = (
        None
        if args.until is None
        else datetime.combine(args.until + timedelta(days=1), datetime.min.time(), UTC)
    )
    async with usage_repository(settings) as repository:
        rows = await repository.between(since, until)
    return build_report(rows, catalog, args.by)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = asyncio.run(run(args))
    except PriceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # database down, unreadable settings
        print(f"aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ABORTED
    print(render(report, args.by))
    return EXIT_UNPRICED if args.strict and report.unpriced else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
