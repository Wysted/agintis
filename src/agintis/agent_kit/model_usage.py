"""Tokens of one model: a pure value type (no langchain), shared by judge records and usage."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Tokens and call count of one model; `cached_input_tokens` is the part of `input_tokens`
    served from the provider's prompt cache."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    calls: int = 0

    def __add__(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.total_tokens + other.total_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
            self.calls + other.calls,
        )

    def to_json(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "calls": self.calls,
        }
