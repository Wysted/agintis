from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from agintis.platform.usage import Purpose, UsageRow, load_price_catalog
from agintis.platform.usage.report import GroupBy, build_report

DAY = datetime(2026, 10, 24, 23, 30, tzinfo=UTC)


def _row(model: str, purpose: Purpose = Purpose.CHAT, when: datetime = DAY) -> UsageRow:
    return UsageRow("r", model, purpose, when, 1_000_000, 0, 0, 1)


def test_unpriced_rows_are_reported_apart_with_their_tokens_and_never_as_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text(
        "models:\n  known:\n"
        "  - {effective_from: 2026-10-24, input: 1, cached_input: 1, output: 1}\n"
    )
    old = datetime(2026, 10, 1, tzinfo=UTC)
    rows = [_row("known"), _row("known", Purpose.JUDGE), _row("mystery"), _row("known", when=old)]

    report = build_report(rows, load_price_catalog(path), GroupBy.PURPOSE)

    assert {k: line.cost_usd for k, line in report.priced.items()} == {
        "chat": Decimal(1),
        "judge": Decimal(1),
    }
    assert report.total_usd == 2
    assert report.unpriced["mystery"].input_tokens == 1_000_000
    assert report.unpriced["mystery"].cost_usd is None
    assert report.unpriced["known"].calls == 1  # before the first entry


def test_rows_with_calls_but_no_tokens_are_unpriced_not_zero_dollars(tmp_path: Path) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text(
        "models:\n  known:\n"
        "  - {effective_from: 2026-10-01, input: 1, cached_input: 1, output: 1}\n"
    )
    empty = UsageRow("r", "known", Purpose.CHAT, DAY, 0, 0, 0, 2)

    report = build_report([empty, _row("known")], load_price_catalog(path), GroupBy.MODEL)

    assert report.priced["known"].calls == 1
    assert report.unpriced["known (no token data)"].calls == 2
