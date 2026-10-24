"""Token usage of the LLM calls behind one dump row, per model (the audit trail of a run's cost).

`ModelUsageCounter` is a callback handler passed in the graph's invoke config: callbacks are
inherited by every LLM call of the run (nodes, subgraphs, structured-output calls, streamed or
not), so nothing a node does is missed. The model key is the configured model name the client
reports (`ls_model_name`), not the response's `model_name` (a provider may return a dated
variant), so roles bound to different profiles stay apart. Calls the provider reports no usage
for still count in `calls` (tokens 0); a failed call counts too (it may have been billed).
shortcut: SDK-level HTTP retries are invisible to callbacks, so a retried request is counted
once; raise it if retries become a real cost source.
"""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

from agintis.agent_kit.judge.records import ModelUsage

UNKNOWN_MODEL = "unknown"  # a client that reports no model name


def response_usage(response: LLMResult) -> ModelUsage:
    """What one finished LLM call reported (`calls` is 1 even when it reported no usage)."""
    call = ModelUsage(calls=1)
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            usage = message.usage_metadata if isinstance(message, AIMessage) else None
            if usage:
                cached = (usage.get("input_token_details") or {}).get("cache_read") or 0
                call += ModelUsage(
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("total_tokens", 0),
                    cached,
                )
    return call


class ModelUsageCounter(BaseCallbackHandler):
    """Sums `usage_metadata` over every LLM call it sees, keyed by model name."""

    run_inline = True

    def __init__(self) -> None:
        self._models: dict[UUID, str] = {}  # run id -> model of the call in flight
        self._usage: dict[str, ModelUsage] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._models[run_id] = str((metadata or {}).get("ls_model_name") or UNKNOWN_MODEL)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._add(run_id, ModelUsage(calls=1))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self._add(run_id, response_usage(response))

    def _add(self, run_id: UUID, call: ModelUsage) -> None:
        model = self._models.pop(run_id, UNKNOWN_MODEL)
        self._usage[model] = self._usage.get(model, ModelUsage()) + call

    @property
    def usage(self) -> dict[str, ModelUsage] | None:
        """Per model, or None when no LLM call ran."""
        return dict(self._usage) or None
