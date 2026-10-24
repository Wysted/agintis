"""One ledger row: the LLM usage of one run on one model (the grain of `llm_usage`)."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Purpose(StrEnum):
    CHAT = "chat"  # a production run through an agent
    DUMP = "dump"  # cli.dump_answers
    JUDGE = "judge"  # cli.judge
    EVAL = "eval"  # cli.eval
    CALIBRATE = "calibrate"  # cli.calibrate_evidence


@dataclass(frozen=True, slots=True)
class UsageRow:
    """`run_id` is the chat run id, or the id of one CLI invocation; two roles on the same model
    of a run are merged into one row. `source` names a CLI command and its input."""

    run_id: str
    model: str
    purpose: Purpose
    created_at: datetime
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    calls: int
    tenant_id: str | None = None
    agent_id: str | None = None
    thread_id: str | None = None
    source: str | None = None
    git_revision: str | None = None
