"""Decorator that puts each run's identity on its trace spans.

`TracedRunner` wraps any `AgentRunner`; the composition root applies it only when tracing is on,
so agents and graph builders know nothing about observability. Spans carry `session.id` (the
thread id, when the run belongs to a thread) and `metadata` {tenant_id, agent_id, run_id}.
Correlation is `metadata.run_id`, the same id the SSE stream announces (the OpenTelemetry trace id
is random and unrelated to it).

OpenInference reads the context attributes when a span ENDS, so the scope also covers the
`aclose()` of the inner stream: the root span of an aborted run (client disconnect, deadline)
ends there.
"""

from collections.abc import AsyncGenerator

from agintis.modules.agents_catalog.api import (
    AgentRunner,
    Clarification,
    RunInput,
    RunnerEvent,
    TranscriptMessage,
)
from agintis.platform.observability import RunAttributes, Tracing
from agintis.shared_kernel.ids import ThreadId


class TracedRunner(AgentRunner):
    def __init__(self, runner: AgentRunner, tracing: Tracing) -> None:
        self._runner = runner
        self._tracing = tracing

    @property
    def agent_id(self) -> str:
        return self._runner.agent_id

    @property
    def description(self) -> str:
        return self._runner.description

    @property
    def supports_threads(self) -> bool:
        return self._runner.supports_threads

    def ensure_available(self) -> None:
        self._runner.ensure_available()

    async def pending_interrupts(self, thread_id: ThreadId) -> tuple[Clarification, ...]:
        return await self._runner.pending_interrupts(thread_id)

    async def thread_messages(self, thread_id: ThreadId) -> tuple[TranscriptMessage, ...]:
        return await self._runner.thread_messages(thread_id)

    async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
        attributes = RunAttributes(
            tenant_id=run_input.tenant_id,
            agent_id=self._runner.agent_id,
            run_id=run_input.run_id,
            session_id=run_input.thread_id,
        )
        stream = self._runner.astream(run_input)
        try:
            while True:
                # The scope is entered around ONE pull at a time, never across the `yield`: the
                # consumer may resume or close this generator from another task, and a context
                # attached in one task cannot be detached from another. Tasks the graph starts
                # during a pull inherit the attributes.
                with self._tracing.run_scope(attributes):
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration:
                        return
                yield event
        finally:
            # Spans that end while closing (the root one on an aborted run) read the attributes
            # now. Attached and detached in this very task, so no cross-task detach.
            with self._tracing.run_scope(attributes):
                await stream.aclose()
