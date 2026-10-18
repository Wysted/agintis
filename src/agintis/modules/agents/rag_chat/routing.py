"""Builds the `TopicRouter` of an agent spec (Strategy selection by `router`)."""

from collections.abc import Callable

from agintis.agent_kit.nodes import (
    ClarificationPolicy,
    EvidenceClarification,
    EvidenceTopic,
    EvidenceTopicRouter,
    LlmTopicRouter,
    NoClarification,
    SingleTopicRouter,
    TopicInfo,
    TopicProbe,
    TopicRouter,
)
from agintis.modules.agents.rag_chat.spec import AgentSpec, ClarificationSpec
from agintis.modules.knowledge.api import DEFAULT_MAX_QUERY_CHARS, RagProfileRegistry
from agintis.platform.llm import ChatModels


def build_router(
    spec: AgentSpec,
    chat_models: ChatModels,
    *,
    probe: TopicProbe | None = None,
    profiles: RagProfileRegistry | None = None,
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
) -> TopicRouter:
    """`SingleTopicRouter` (no model call), `EvidenceTopicRouter` (`router: evidence`: needs the
    knowledge `probe` and the `profiles` registry, calls no model) or `LlmTopicRouter` over the
    spec's topics, with the spec's `route` prompt, `max_topics` and router knobs."""
    if spec.uses_evidence_router:
        if probe is None or profiles is None:
            raise ValueError(
                f"agent {spec.id!r}: router 'evidence' needs the knowledge probe and the profiles"
            )
        return EvidenceTopicRouter(
            probe,
            [EvidenceTopic(t.name, profiles.get(t.profile), t.evidence_bias) for t in spec.topics],
            spec.max_topics,
            spec.evidence_margin,
            spec.evidence_aggregate,
            spec.probe_k,
            max_query_chars,
        )
    if not spec.uses_llm_router:
        return SingleTopicRouter(spec.topics[0].name)
    return LlmTopicRouter(
        chat_models,
        [TopicInfo(t.name, t.description) for t in spec.topics],
        spec.max_topics,
        spec.prompts.to_prompt_set(citations=spec.citations).route,
        spec.secondary_min_confidence,
    )


def _no_clarification(spec: ClarificationSpec, router: TopicRouter) -> ClarificationPolicy:
    return NoClarification()


def _evidence_clarification(spec: ClarificationSpec, router: TopicRouter) -> ClarificationPolicy:
    if not isinstance(router, EvidenceTopicRouter):
        raise ValueError("clarification policy 'evidence' needs `router: evidence`")
    if spec.low_confidence_below is None or spec.out_of_scope_below is None:
        raise ValueError("clarification policy 'evidence' needs both thresholds")
    return EvidenceClarification(
        router, spec.low_confidence_below, spec.out_of_scope_below, spec.low_scope_max_topics
    )


# Strategy registry: `clarification.policy` -> factory. A new policy is one entry (plus its spec
# value); an interrupt-based one would also add its node in the graph builder.
CLARIFICATION_POLICIES: dict[
    str, Callable[[ClarificationSpec, TopicRouter], ClarificationPolicy]
] = {
    "none": _no_clarification,
    "evidence": _evidence_clarification,
}


def build_clarification(spec: AgentSpec, router: TopicRouter) -> ClarificationPolicy:
    """The scope-check policy of the spec, over the SAME router instance the route node uses (the
    `evidence` policy reuses its probe, so a turn probes the corpus once)."""
    return CLARIFICATION_POLICIES[spec.clarification.policy](spec.clarification, router)
