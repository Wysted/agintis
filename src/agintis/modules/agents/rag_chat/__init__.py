"""Generic, configurable RAG chat agent: one `AgentSpec` (YAML) = one agent."""

from collections.abc import Iterable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from agintis.agent_kit.runtime import LangGraphRunner
from agintis.agent_kit.usage import UsageRecorder
from agintis.modules.agents.rag_chat.graph import (
    RagChatDeps,
    agent_deps,
    build_graph,
    rag_chat_runner,
    roles_used,
)
from agintis.modules.agents.rag_chat.loader import load_agent_specs
from agintis.modules.agents.rag_chat.routing import build_clarification, build_router
from agintis.modules.agents.rag_chat.spec import (
    AgentSpec,
    ClarificationSpec,
    InvalidAgentSpecError,
    PromptsSpec,
    TopicSpec,
    parse_agent_spec,
    validate_agent_specs,
)
from agintis.modules.knowledge.api import RagProfileRegistry


def rag_chat_runners(
    specs: Iterable[AgentSpec],
    profiles: RagProfileRegistry,
    deps: RagChatDeps,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    usage: UsageRecorder | None = None,
) -> tuple[LangGraphRunner, ...]:
    """One runner per spec (validated against `profiles` first: unknown profile fails fast)."""
    specs = tuple(specs)
    validate_agent_specs(specs, profiles)
    return tuple(rag_chat_runner(spec, profiles, deps, checkpointer, usage) for spec in specs)


__all__ = [
    "AgentSpec",
    "ClarificationSpec",
    "InvalidAgentSpecError",
    "PromptsSpec",
    "RagChatDeps",
    "TopicSpec",
    "agent_deps",
    "build_clarification",
    "build_graph",
    "build_router",
    "load_agent_specs",
    "parse_agent_spec",
    "rag_chat_runner",
    "rag_chat_runners",
    "roles_used",
    "validate_agent_specs",
]
