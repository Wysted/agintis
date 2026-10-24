"""Versioned LLM price catalog (USD per 1M tokens), declared in a strict YAML file.

```yaml
models:
  gpt-6-luna:
    - effective_from: 2026-10-01     # UTC date; the entry applies from this day on
      input: 0.10                    # per 1M input tokens (cached ones excluded)
      cached_input: 0.10             # per 1M input tokens served from the prompt cache
      output: 0.50
```

A usage row is priced with the latest entry whose `effective_from` is on or before the row's UTC
date. A model the file does not know, or a row older than the model's first entry, has NO price
(`price_at` returns None): reports list it as unpriced with its tokens, never as $0. The file is
read when the report runs, so correcting a price re-prices history; nothing is stored in USD.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from agintis.platform.yaml_config import YamlConfigError, load_yaml_file

TOKENS_PER_PRICE_UNIT = Decimal(1_000_000)
DEFAULT_PRICES_FILE = Path(__file__).with_name("prices.yaml")


class PriceError(ValueError):
    """The price file is unreadable or invalid (the message names the file)."""


class PriceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    effective_from: date
    input: Decimal = Field(ge=0)
    cached_input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)

    def cost(self, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> Decimal:
        """USD of one usage: cached tokens at `cached_input`, the rest of the input at `input`."""
        cached = min(cached_input_tokens, input_tokens)
        return (
            (input_tokens - cached) * self.input
            + cached * self.cached_input
            + output_tokens * self.output
        ) / TOKENS_PER_PRICE_UNIT


class _PriceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, list[PriceEntry]]

    @model_validator(mode="after")
    def _entries_are_usable(self) -> Self:
        for model, entries in self.models.items():
            if not entries:
                raise ValueError(f"{model!r}: needs at least one entry")
            dates = [entry.effective_from for entry in entries]
            if len(set(dates)) != len(dates):
                raise ValueError(f"{model!r}: duplicate effective_from")
            entries.sort(key=lambda entry: entry.effective_from)
        return self


@dataclass(frozen=True, slots=True)
class PriceCatalog:
    entries: Mapping[str, tuple[PriceEntry, ...]]  # per model, oldest first

    def price_at(self, model: str, when: datetime) -> PriceEntry | None:
        """The entry in force for `model` at `when` (aware datetimes; UTC date), else None."""
        day = when.astimezone(UTC).date()
        current = None
        for entry in self.entries.get(model, ()):
            if entry.effective_from > day:
                break
            current = entry
        return current


def load_price_catalog(path: Path = DEFAULT_PRICES_FILE) -> PriceCatalog:
    """Parse the file; `PriceError` on any problem (a wrong price must stop the report)."""
    try:
        data = load_yaml_file(path)
        parsed = _PriceFile.model_validate(data)
    except YamlConfigError as exc:
        raise PriceError(str(exc)) from exc
    except PydanticValidationError as exc:
        raise PriceError(f"{path.name}: invalid price file: {exc}") from exc
    return PriceCatalog({model: tuple(entries) for model, entries in parsed.models.items()})
