from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from agintis.platform.usage import (
    DEFAULT_PRICES_FILE,
    PriceCatalog,
    PriceError,
    load_price_catalog,
)

PRICES = """\
models:
  m:
    - effective_from: 2026-11-01
      input: 2
      cached_input: 1
      output: 4
    - effective_from: 2026-10-01
      input: 1
      cached_input: 0.5
      output: 2
"""


def _catalog(tmp_path: Path, text: str = PRICES) -> PriceCatalog:
    path = tmp_path / "prices.yaml"
    path.write_text(text, encoding="utf-8")
    return load_price_catalog(path)


def at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


def test_the_latest_entry_in_force_prices_the_row(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    def input_at(day: str) -> Decimal | None:
        entry = catalog.price_at("m", at(day))
        return entry.input if entry else None

    assert input_at("2026-10-15") == 1
    assert input_at("2026-11-01") == 2
    assert input_at("2027-01-01") == 2


def test_unknown_model_or_row_before_the_first_entry_is_unpriced(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    assert catalog.price_at("other", at("2026-11-01")) is None
    assert catalog.price_at("m", at("2026-09-30")) is None


def test_cached_tokens_use_the_cached_price(tmp_path: Path) -> None:
    entry = _catalog(tmp_path).price_at("m", at("2026-10-15"))
    assert entry is not None

    # 600k plain input at 1, 400k cached at 0.5, 100k output at 2 (USD per 1M)
    assert entry.cost(1_000_000, 400_000, 100_000) == Decimal("0.6") + Decimal("0.2") + Decimal(
        "0.2"
    )


@pytest.mark.parametrize(
    "text",
    [
        "models:\n  m: []\n",
        "models:\n  m:\n  - {effective_from: 2026-10-01, input: -1, cached_input: 0, output: 0}\n",
        "models:\n  m:\n    - {effective_from: 2026-10-01, input: 1, output: 1}\n",
        "models:\n  m:\n    - {effective_from: 2026-10-01, input: 1, cached_input: 1, output: 1}\n"
        "    - {effective_from: 2026-10-01, input: 1, cached_input: 1, output: 1}\n",
        "models: {}\nextra: 1\n",
        "not: a catalog\n",
    ],
)
def test_an_invalid_file_fails_loudly(tmp_path: Path, text: str) -> None:
    with pytest.raises(PriceError):
        _catalog(tmp_path, text)


def test_the_shipped_file_is_valid() -> None:
    assert load_price_catalog(DEFAULT_PRICES_FILE).price_at("gpt-6-luna", at("2026-10-24"))
