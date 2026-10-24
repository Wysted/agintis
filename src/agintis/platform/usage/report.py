"""Cost report over ledger rows: tokens summed per group, USD computed at read time.

A row is priced with the catalog entry in force at its `created_at`. Rows the catalog cannot
price (unknown model, older than the first entry) are never counted as $0: they are summed per
model in `Report.unpriced`, with their tokens and no cost.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal
from enum import StrEnum
from typing import Final

from agintis.platform.usage.prices import PriceCatalog
from agintis.platform.usage.rows import UsageRow

NO_VALUE: Final = "-"


class GroupBy(StrEnum):
    PURPOSE = "purpose"
    DAY = "day"
    TENANT = "tenant"
    THREAD = "thread"
    MODEL = "model"


_KEYS: dict[GroupBy, Callable[[UsageRow], str]] = {
    GroupBy.PURPOSE: lambda row: row.purpose.value,
    GroupBy.DAY: lambda row: row.created_at.astimezone(UTC).date().isoformat(),
    GroupBy.TENANT: lambda row: row.tenant_id or NO_VALUE,
    GroupBy.THREAD: lambda row: row.thread_id or NO_VALUE,
    GroupBy.MODEL: lambda row: row.model,
}


@dataclass(slots=True)
class Line:
    """Tokens and calls of one group; `cost_usd` is None on unpriced lines."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    calls: int = 0
    cost_usd: Decimal | None = Decimal(0)

    def add(self, row: UsageRow, cost: Decimal | None) -> None:
        self.input_tokens += row.input_tokens
        self.output_tokens += row.output_tokens
        self.cached_input_tokens += row.cached_input_tokens
        self.calls += row.calls
        if self.cost_usd is not None and cost is not None:
            self.cost_usd += cost


@dataclass(frozen=True, slots=True)
class Report:
    priced: dict[str, Line]  # group key -> line
    unpriced: dict[str, Line]  # model -> line (cost_usd None)

    @property
    def total_usd(self) -> Decimal:
        return sum((line.cost_usd or Decimal(0) for line in self.priced.values()), Decimal(0))


def build_report(rows: Iterable[UsageRow], catalog: PriceCatalog, by: GroupBy) -> Report:
    """Groups priced rows by `by` (days are UTC dates); the rest go to `Report.unpriced`.

    Unpriced lines are per model; rows with calls but no tokens get a `(no token data)` line.
    """
    key_of = _KEYS[by]
    priced: dict[str, Line] = {}
    unpriced: dict[str, Line] = {}
    for row in rows:
        entry = catalog.price_at(row.model, row.created_at)
        # Calls without tokens (no usage reported, or a failed call): $0 would hide real spend.
        no_tokens = row.input_tokens == 0 and row.output_tokens == 0
        if entry is None or no_tokens:
            label = f"{row.model} (no token data)" if no_tokens else row.model
            unpriced.setdefault(label, Line(cost_usd=None)).add(row, None)
            continue
        cost = entry.cost(row.input_tokens, row.cached_input_tokens, row.output_tokens)
        priced.setdefault(key_of(row), Line()).add(row, cost)
    return Report(dict(sorted(priced.items())), dict(sorted(unpriced.items())))
