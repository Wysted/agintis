"""Token usage of the LLM calls of one run, per model (the audit trail of a run's cost).

`ModelUsageCounter` is a callback handler passed in the graph's invoke config: callbacks are
inherited by every LLM call of the run (nodes, subgraphs, structured-output calls, streamed or
not), so nothing a node does is missed. The model key is the configured model name the client
reports (`ls_model_name`), not the response's `model_name` (a provider may return a dated
variant), so roles bound to different profiles stay apart. Calls the provider reports no usage
for still count in `calls` (tokens 0); a failed call counts too (it may have been billed).
shortcut: SDK-level HTTP retries are invisible to callbacks, so a retried request is counted
once, and a stream that fails midway counts its call with 0 tokens: totals are a lower bound;
raise it if retries or failed streams become a real cost source.
"""

import asyncio
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

from agintis.agent_kit.model_usage import ModelUsage

logger = logging.getLogger(__name__)

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


# Ambient counter for tools that call models from many places without passing a run config (the
# measurement CLIs): langchain adds a handler found here to every run started in this context.
_ambient: ContextVar[ModelUsageCounter | None] = ContextVar("agintis_usage_counter", default=None)
register_configure_hook(_ambient, inheritable=True)


@contextmanager
def counting_usage() -> Iterator[ModelUsageCounter]:
    """Counts every LLM call started inside the block (tasks created in it included)."""
    counter = ModelUsageCounter()
    token = _ambient.set(counter)
    try:
        yield counter
    finally:
        _ambient.reset(token)


# A usage write is audit bookkeeping: it must never hold a run (or the shutdown) for long.
USAGE_WRITE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class RunUsage:
    """What one chat run spent: per model, with the identity the ledger keys on."""

    run_id: str
    agent_id: str
    tenant_id: str
    thread_id: str | None
    usage: Mapping[str, ModelUsage]


class UsageSink(Protocol):
    """Port: persists the usage of a finished run (idempotent per run and model)."""

    async def record(self, run: RunUsage) -> None: ...


class UsageRecorder:
    """Writes to a `UsageSink` in background tasks that never delay or break the caller.

    Strong references keep in-flight writes alive (a task held weakly may vanish); each write is
    bounded by a timeout and its failure is only logged. `aclose` drains them: call it before the
    sink's database closes.
    """

    def __init__(self, sink: UsageSink, timeout_seconds: float = USAGE_WRITE_TIMEOUT_SECONDS):
        self._sink = sink
        self._timeout = timeout_seconds
        self._tasks: set[asyncio.Task[None]] = set()

    def schedule(self, run: RunUsage) -> None:
        task = asyncio.create_task(self._write(run))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _write(self, run: RunUsage) -> None:
        try:
            await asyncio.wait_for(self._sink.record(run), self._timeout)
        except Exception:
            logger.warning("Could not record the LLM usage of run %s", run.run_id, exc_info=True)

    async def aclose(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
