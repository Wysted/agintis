r"""The generic RAG chat graph with multi-topic routing.

    analyze -> route -> [Send topic xN] -> merge -> (sufficiency) -> synthesize
                  ^   \                                   |
                  |    +-> refuse (out of scope)         |
                  +------------ rewrite <-----------------+

Built from an `AgentSpec` and generic `agent_kit` nodes; nothing here knows a case. Compiled
twice from one builder like direct-chat (plain for stateless runs, with the checkpointer for
thread runs, which keeps `messages` between turns).

`route` returns `Command(goto=[Send("topic", task)...])`: ONE compiled topic subgraph
(`extract_filters -> retrieve`, private state) runs once per selected topic in parallel and
returns a `topic_results` entry each; `merge` interleaves them by rank (weighted towards the
primary topic, which `merge_tiebreak: llm` lets the ROUTER model pick on every merge with 2+
topics with sources, earlier attempts' evidence last) into the numbered `sources`; `synthesize`
emits the `Sources` event from the list it numbers (every path to the answer goes through it).
When the sufficiency strategy says the sources are not enough, the graph goes through `rewrite`
(a new query, `attempts + 1`; an unchanged query goes straight to synthesis)
and `route` again, at most `max_rewrites` times per turn. A single-topic agent takes the same
path (`SingleTopicRouter` makes no model call; neither does the `evidence` router, which probes
the corpus and leaves its query embedding in the run context for the topic subgraphs). The edge
`analyze -> route` is where the clarification policy (3.5c) sits: it runs INSIDE `route` (first
route of the turn), so the probe it needs is the router's own (one per turn); a question it judges
out of scope goes to `refuse` (fixed message, no retrieval, no synthesis call) and ends the turn.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel

from agintis.agent_kit.nodes import (
    REFUSE_NODE,
    TOPIC_NODE,
    AnalyzeNode,
    FilterExtractor,
    MergeSourcesNode,
    RagState,
    RefuseNode,
    RewriteQueryNode,
    RouteNode,
    SynthesizeNode,
    TopicProbe,
    TopicRetrieval,
    build_sufficiency,
    build_topic_graph,
)
from agintis.agent_kit.nodes.sufficiency import REWRITE
from agintis.agent_kit.nodes.sufficiency import SYNTHESIZE as SYNTHESIZE_ROUTE
from agintis.agent_kit.runtime import (
    AgentContext,
    LangGraphRunner,
    StepLabeler,
    llm_availability,
)
from agintis.modules.agents.rag_chat.routing import build_clarification, build_router
from agintis.modules.agents.rag_chat.spec import AgentSpec, InvalidAgentSpecError
from agintis.modules.knowledge.api import DEFAULT_MAX_QUERY_CHARS, RagProfileRegistry
from agintis.platform.llm import ChatModels, ChatRole, LlmError

ANALYZE = "analyze"
ROUTE = "route"
TOPIC = TOPIC_NODE
MERGE = "merge"
REWRITE_NODE = REWRITE
SYNTHESIZE = "synthesize"
REFUSE = REFUSE_NODE


@dataclass(frozen=True)
class RagChatDeps:
    """Long-lived collaborators of every rag-chat agent (wired in bootstrap)."""

    chat_models: ChatModels
    retrieval: TopicRetrieval
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS
    # Corpus probe of an `evidence` router (None: such a spec fails to build).
    probe: TopicProbe | None = None


def roles_used(spec: AgentSpec) -> tuple[ChatRole, ...]:
    """The models a run of this spec can call (ROUTER only for the LLM router or the merge
    tie-break)."""
    roles = [ChatRole.ANALYZER, ChatRole.SYNTHESIS]
    if spec.uses_router_model:
        roles.append(ChatRole.ROUTER)
    return tuple(roles)


def agent_deps(spec: AgentSpec, deps: RagChatDeps) -> RagChatDeps:
    """`deps` with the agent's own `ChatModels` view: its `models:` bindings (only for the roles
    it really calls; the rest are never resolved nor built) over the platform defaults.

    Fails here, at wiring time, with an `InvalidAgentSpecError` naming the agent, when a bound
    profile is unknown or its `api_key_env` is not set. Nodes keep calling `chat_models.get(role)`.
    """
    try:
        view = deps.chat_models.bind(spec.models, agent=spec.id, roles=roles_used(spec))
    except LlmError as exc:
        raise InvalidAgentSpecError(str(exc)) from exc  # already names the profile and the agent
    return replace(deps, chat_models=view)


def _builder(
    spec: AgentSpec, profiles: RagProfileRegistry, deps: RagChatDeps
) -> StateGraph[Any, Any, Any, Any]:
    prompts = spec.prompts.to_prompt_set(citations=spec.citations)
    topic_profiles = {t.name: profiles.get(t.profile) for t in spec.topics}
    extractors = {
        t.name: FilterExtractor(
            deps.chat_models, topic_profiles[t.name], t.filters, prompts.extract_filters
        )
        for t in spec.topics
    }
    sufficiency = build_sufficiency(
        spec.sufficiency, spec.min_sources, spec.max_rewrites, spec.min_rerank_score
    )
    builder = StateGraph(RagState, context_schema=AgentContext)
    builder.add_node(
        ANALYZE,
        AnalyzeNode(deps.chat_models, prompts.analyze, spec.max_history_messages),
    )
    router = build_router(
        spec,
        deps.chat_models,
        probe=deps.probe,
        profiles=profiles,
        max_query_chars=deps.max_query_chars,
    )
    policy = build_clarification(spec, router)
    refusing = policy.may_refuse  # only then can a turn end in `refuse`
    builder.add_node(
        ROUTE,
        RouteNode(
            router,
            [t.name for t in spec.topics],
            spec.max_topics,
            TOPIC,
            policy,
            REFUSE,
            spec.clarification.low_scope_max_topics,
        ),
        destinations=(TOPIC, REFUSE) if refusing else (TOPIC,),
    )
    if refusing:
        builder.add_node(
            REFUSE,
            RefuseNode(spec.clarification.out_of_scope_message, spec.clarification.social_message),
        )
        builder.add_edge(REFUSE, END)
    builder.add_node(
        TOPIC,
        build_topic_graph(
            deps.retrieval, topic_profiles, extractors, spec.limits, deps.max_query_chars
        ),
    )
    tiebreak = deps.chat_models if spec.merge_tiebreak == "llm" else None
    builder.add_node(MERGE, MergeSourcesNode(spec.limits, spec.primary_weight, tiebreak))
    builder.add_node(
        REWRITE_NODE,
        RewriteQueryNode(
            deps.chat_models, prompts.rewrite, deps.max_query_chars, ROUTE, SYNTHESIZE
        ),
        destinations=(ROUTE, SYNTHESIZE),
    )
    builder.add_node(
        SYNTHESIZE,
        SynthesizeNode(
            deps.chat_models,
            prompts.synthesize,
            limits=spec.limits,
            max_history_messages=spec.max_history_messages,
        ),
    )
    builder.add_edge(START, ANALYZE)
    builder.add_edge(ANALYZE, ROUTE)
    builder.add_edge(TOPIC, MERGE)  # runs once, after every Send branch of the superstep
    builder.add_conditional_edges(
        MERGE, sufficiency, {REWRITE: REWRITE_NODE, SYNTHESIZE_ROUTE: SYNTHESIZE}
    )
    builder.add_edge(SYNTHESIZE, END)
    return builder


def build_graph(
    spec: AgentSpec,
    profiles: RagProfileRegistry,
    deps: RagChatDeps,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Pregel[Any, Any, Any, Any]:
    return _builder(spec, profiles, agent_deps(spec, deps)).compile(checkpointer=checkpointer)


def topic_step_label(topics: Iterable[str]) -> StepLabeler:
    """Names a topic branch after its topic. Closed over the spec's topic names: only one of them
    is ever returned, so no other text of the task input (the query) can reach the wire."""
    known = frozenset(topics)

    def label(node: str, task_input: Any) -> str | None:
        if node == TOPIC and isinstance(task_input, Mapping):
            topic = task_input.get("topic")
            if isinstance(topic, str) and topic in known:
                return topic
        return None

    return label


def rag_chat_runner(
    spec: AgentSpec,
    profiles: RagProfileRegistry,
    deps: RagChatDeps,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> LangGraphRunner:
    """The agent as a runner; with a `checkpointer` it also supports threads (multi-turn)."""
    deps = agent_deps(spec, deps)
    builder = _builder(spec, profiles, deps)
    return LangGraphRunner(
        spec.id,
        spec.description,
        builder.compile(),
        thread_graph=builder.compile(checkpointer=checkpointer) if checkpointer else None,
        availability=llm_availability(deps.chat_models, *roles_used(spec)),
        step_label=topic_step_label(t.name for t in spec.topics),
    )
