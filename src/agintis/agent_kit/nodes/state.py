"""State of the RAG nodes: one TypedDict plus the small plain-data records it carries.

Every value is plain data (strings, numbers, lists of TypedDicts), never a knowledge dataclass:
the state is checkpointed, and msgpack round-trips plain data without registering types.

Every node ALWAYS writes the keys it owns (an empty list/string when it has nothing), because a
thread keeps the whole state between turns: a key a node skipped would leak the previous
turn's value into the next one.

Shared vs private (3.5b): the parent graph sees `messages`, `standalone_query`, `is_clear`,
`confidence`, `query`, `scope_level`, `scope_score`, `routed_topics`, `topic_results`, `sources`,
`previous_sources`, `attempts`, `answer` and `cited_sources`. A topic subgraph (`TopicState`, one
Send per routed topic) keeps its retrieval working set (`filters`, the extracted conditions)
PRIVATE and returns only a `topic_results` entry.

`topic_results` is an ACCUMULATING channel (parallel Send branches each append one entry), and a
thread keeps channel values between turns, so a plain append would pile results up across turns
and rewrite attempts: the node that starts a fan-out (`RouteNode`) writes `Overwrite([])` first,
which bypasses the reducer and resets the channel.

`cited_sources` is the opposite: an `or_` channel that is never reset (one entry per answer of the
thread, keyed by message id), the only state a transcript needs besides `messages`.
"""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from agintis.modules.agents_catalog.api import ScopeLevel

type FilterOp = Literal["eq", "gte", "lte"]
# AnalyzeNode's reading of the last user message: a question about the agent's domain, small talk
# (greeting, thanks, "what can you do"), or anything else (tasks, personal advice, other topics).
RequestKind = Literal["domain", "social", "other"]
# Why a turn ended in the refuse node: the evidence band was `out`, the request was not about the
# domain, or it was small talk. None on a normal answer.
RefusalReason = Literal["out_of_scope", "off_domain", "social"]


class FilterClause(TypedDict):
    """One extracted metadata condition: `field <op> value`, typed as at ingest."""

    field: str
    op: FilterOp
    value: str | int | float


class SourceItem(TypedDict):
    """A retrieved passage the answer may cite. Its citation number is its position (1-based)."""

    source: str
    title: str | None
    locator: str | None
    text: str
    # The id of the retrieved chunk (dedupe key when topic results are merged).
    chunk_id: NotRequired[str]
    # Relevance evidence of the hit (RetrievedHit): the cross-encoder score when a reranker
    # scored it, and the 1-based rank in each hybrid branch (both None in dense-only retrieval).
    # Read by `RelevanceSufficiency`; absent = unknown.
    rerank_score: NotRequired[float]
    lexical_rank: NotRequired[int]
    dense_rank: NotRequired[int]


class CitedSourceItem(TypedDict):
    """What a client sees of a source (the SSE `sources` item); no text (lean checkpoints)."""

    source: str
    title: str | None
    locator: str | None


class RoutedTopic(TypedDict):
    """A topic the router selected and how sure it was (0.0 = unknown, e.g. a fallback)."""

    topic: str
    confidence: float
    # Evidence routers only: the raw probe similarities (best first) the confidence comes from.
    similarities: NotRequired[list[float]]


class TopicTask(TypedDict):
    """The argument of one `Send`: which topic to search and with what text."""

    topic: str
    query: str


class TopicResult(TypedDict):
    """What a topic subgraph returns: its capped sources (best first) and the filter level."""

    topic: str
    sources: list[SourceItem]
    filter_level: str


class TopicState(TypedDict):
    """PRIVATE state of the topic subgraph (input = `TopicTask`, output = `TopicOutput`)."""

    topic: str
    query: str
    filters: NotRequired[list[FilterClause]]
    topic_results: NotRequired[list[TopicResult]]


class TopicOutput(TypedDict):
    topic_results: list[TopicResult]


class RagState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # AnalyzeNode: the question rewritten to stand alone, plus the clarity verdict (kept for the
    # clarification policy, not acted on yet).
    standalone_query: NotRequired[str]
    is_clear: NotRequired[bool]
    confidence: NotRequired[float]
    # AnalyzeNode: what kind of request this is ("domain" when the analyzer degrades). Only acted
    # on when a clarification policy is active (see RouteNode).
    request_kind: NotRequired[RequestKind]
    # The text retrieval runs on (= standalone_query; the rewrite loop replaces it).
    query: NotRequired[str]
    # RouteNode: the topics searched this attempt, primary first (the merge order), with the
    # router's confidence in each.
    routed_topics: NotRequired[list[RoutedTopic]]
    # Topic subgraphs: one entry per routed topic. Accumulates over a fan-out; `RouteNode`
    # resets it with `Overwrite([])` (see the module docstring).
    topic_results: NotRequired[Annotated[list[TopicResult], operator.add]]
    # MergeSourcesNode: the numbered sources of the answer.
    sources: NotRequired[list[SourceItem]]
    # MergeSourcesNode: the evidence of the earlier attempts of this turn (the merged list of
    # the attempt before), merged LAST into the next attempt so a rewrite never discards it.
    # AnalyzeNode resets it each turn.
    previous_sources: NotRequired[list[SourceItem]]
    # AnalyzeNode resets it to 0 each turn; RewriteQueryNode counts the rewrites of the turn.
    attempts: NotRequired[int]
    # RouteNode's clarification policy, FIRST route of the turn only: the verdict (`in` | `low`
    # | `out`) and the evidence score behind it (None = none). AnalyzeNode resets them each turn
    # to None / None (= not checked, e.g. no policy); a rewrite round keeps them.
    scope_level: NotRequired[ScopeLevel | None]
    scope_score: NotRequired[float | None]
    # RouteNode: why the turn is refused (None = a normal answer). AnalyzeNode resets it each turn.
    refusal_reason: NotRequired[RefusalReason | None]
    # SynthesizeNode (or RefuseNode): the final text (it is also appended to `messages`).
    answer: NotRequired[str]
    # SynthesizeNode: the sources each answer cited, keyed by the id of its AIMessage. Merged with
    # `or_` so earlier turns are kept (the transcript reads them back); never reset. Kept out of
    # the message itself: state messages are replayed to the provider.
    cited_sources: NotRequired[Annotated[dict[str, list[CitedSourceItem]], operator.or_]]
