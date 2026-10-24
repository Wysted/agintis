"""Events an agent run emits. Transport-agnostic: HTTP/SSE mapping lives at the edge."""

from dataclasses import dataclass
from typing import Literal

from agintis.shared_kernel.ids import RunId

StepStatus = Literal["started", "finished"]
# How well a question fits the agent's corpus: `in` = answer normally, `low` = answer but the
# match is weak, `out` = the agent refuses (no retrieval, no sources).
ScopeLevel = Literal["in", "low", "out"]

# Finish reason when a runner ends without reporting one.
DEFAULT_FINISH_REASON = "stop"
# Finish reason of a run that paused on an interrupt and waits for a resume (it carries the
# Clarification the user must answer). Thread runs end in the `interrupted` thread status.
INTERRUPTED_FINISH_REASON = "interrupted"
# Error code of a thread run that hit its deadline. `retryable` for a thread run means "safe to
# retry only after checking the thread state": the supersteps committed before the deadline
# persist in the checkpoint, so a retry is NOT idempotent.
RUN_TIMEOUT_CODE = "run_timeout"


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A document an answer was grounded on."""

    source: str
    title: str | None = None
    locator: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    """One chat turn of a thread, as a client shows it. `sources` are the ones an assistant
    turn's `[n]` citations index (empty for user turns and answers that cited nothing)."""

    role: Literal["user", "assistant"]
    text: str
    sources: tuple[SourceRef, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0  # the part of input_tokens served from the provider's prompt cache


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: RunId
    agent_id: str


@dataclass(frozen=True, slots=True)
class Token:
    text: str


@dataclass(frozen=True, slots=True)
class Step:
    """Progress hint. `label` tells apart parallel runs of the same node (a topic fan-out) or the
    steps inside them; None when the node has nothing to tell. Never carries user input."""

    node: str
    status: StepStatus
    label: str | None = None


@dataclass(frozen=True, slots=True)
class Sources:
    items: tuple[SourceRef, ...]


@dataclass(frozen=True, slots=True)
class Scope:
    """The clarification policy's verdict on the question, emitted once per turn, before any
    sources or tokens, and only by agents that run such a policy. `score` is the evidence it was
    based on (None when there was none)."""

    level: ScopeLevel
    score: float | None = None


@dataclass(frozen=True, slots=True)
class Clarification:
    """A question the run is paused on. In a thread run `interrupt_id` names the interrupt: the
    answer must come back keyed by it (empty only outside thread runs)."""

    question: str
    options: tuple[str, ...] = ()
    interrupt_id: str = ""


@dataclass(frozen=True, slots=True)
class RunFailed:
    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class RunCompleted:
    finish_reason: str
    usage: TokenUsage | None = None


# What a runner itself may emit: content plus an optional final RunCompleted (finish/usage).
RunnerEvent = Token | Step | Sources | Scope | Clarification | RunCompleted
# Everything a consumer of a run can see (the use case adds RunStarted / RunFailed).
RunEvent = RunStarted | RunnerEvent | RunFailed
