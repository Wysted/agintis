"""Scripted chat model for the RAG nodes: plain streamed replies and structured output.

`GenericFakeChatModel.with_structured_output` raises NotImplementedError, so this subclass of
`BaseChatModel` provides it the way real providers behave in a graph: the structured call is a
real (NON-streamed `_generate`, UNTAGGED) LLM run whose JSON text is parsed into the schema, with
usage and finish reason on the message (what `bind_structured` produces). That is what lets the
tests prove that analyzer output never reaches the client as `Token` events.
"""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from agintis.platform.llm import ChatModels, ChatRole, LlmConfig, LlmProfile, ModelCatalog

JSON_PIECE_CHARS = 8


class ScriptedChatModel(BaseChatModel):
    """`replies` are streamed word by word (popped per plain call); `structured` values
    (BaseModel, dict or raw JSON text) are popped per `with_structured_output` call.

    `seen` records the messages of every plain call, `structured_seen` those of every structured
    one and `schemas` the schemas it was bound with (the lists are shared with the copies made
    for structured calls). `responder` answers structured calls by schema instead of by order.
    """

    replies: list[str] = Field(default_factory=list)
    structured: list[Any] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    structured_seen: list[list[BaseMessage]] = Field(default_factory=list)
    schemas: list[Any] = Field(default_factory=list)
    as_json: bool = False
    # Optional `(schema, messages) -> reply`: answers a structured call by what it is bound to
    # (deterministic even when several structured calls run in parallel); `structured` is then
    # not consumed.
    responder: Any = None
    bound_schema: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages: list[BaseMessage]) -> str:
        if self.as_json:
            self.structured_seen.append(list(messages))
            if self.responder is not None:
                value = self.responder(self.bound_schema, list(messages))
            else:
                value = self.structured.pop(0)
            if isinstance(value, BaseModel):
                return value.model_dump_json()
            return value if isinstance(value, str) else json.dumps(value)
        self.seen.append(list(messages))
        return self.replies.pop(0)

    def _pieces(self, text: str) -> list[str]:
        if self.as_json:
            return [text[i : i + JSON_PIECE_CHARS] for i in range(0, len(text), JSON_PIECE_CHARS)]
        words = text.split(" ")
        return [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._next(messages)
        # Like a real provider's non-streamed reply: usage and finish reason ride on the message.
        message = AIMessage(
            content=text,
            usage_metadata=UsageMetadata(input_tokens=3, output_tokens=2, total_tokens=5),
            response_metadata={"finish_reason": "stop"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for piece in self._pieces(self._next(messages)):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata=UsageMetadata(input_tokens=3, output_tokens=2, total_tokens=5),
                response_metadata={"finish_reason": "stop"},
            )
        )

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[LanguageModelInput, Any]:
        """Like `bind_structured` with real providers: a non-streamed UNTAGGED LLM run
        (`_generate`, usage on the message), then a parser; with
        `include_raw` the result is `{raw, parsed, parsing_error}` and a parse failure does not
        raise (what langchain does)."""
        self.schemas.append(schema)
        parser: PydanticOutputParser[Any] = PydanticOutputParser(pydantic_object=schema)
        llm = self.model_copy(update={"as_json": True, "bound_schema": schema})
        if not include_raw:
            return llm | parser

        def wrap(raw: AIMessage) -> dict[str, Any]:
            try:
                return {"raw": raw, "parsed": parser.invoke(raw), "parsing_error": None}
            except Exception as exc:
                return {"raw": raw, "parsed": None, "parsing_error": exc}

        return llm | RunnableLambda(wrap)


def chat_models_for(
    model: BaseChatModel | None = None,
    *,
    by_role: dict[ChatRole, BaseChatModel] | None = None,
    profiles: Mapping[str, LlmProfile] | None = None,
) -> ChatModels:
    """Real `ChatModels` (so `.get(role)` and availability behave) built over fakes. `profiles`
    adds named profiles (a spec's `models:` bindings), served by the fallback with a dummy key."""
    fallback = model if model is not None else ScriptedChatModel()
    config = LlmConfig(provider="openai", models={role: role.value for role in ChatRole})
    chosen = by_role or {}

    def builder(model_id: str, **_kwargs: Any) -> BaseChatModel:
        role = ChatRole(model_id) if model_id in {r.value for r in ChatRole} else None
        return chosen.get(role, fallback) if role else fallback

    if profiles is None:
        return ChatModels(config, builder=builder)
    return ChatModels(
        ModelCatalog(config, profiles, environ=lambda _name: "test-key", builder=builder)
    )


def rate_limit_error() -> Exception:
    """A real openai 429, the way a provider surfaces it (classified `llm_rate_limited`)."""
    import httpx2  # the HTTP client openai >= 3 is built on
    import openai

    request = httpx2.Request("POST", "https://api.openai.test/v1/chat/completions")
    return openai.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=request), body=None
    )


class RaisingModel(BaseChatModel):
    """Every call, plain or structured, raises `error` (a provider or network failure)."""

    error: Any

    @property
    def _llm_type(self) -> str:
        return "raising"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise self.error

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[LanguageModelInput, Any]:
        def boom(_input: Any) -> Any:
            raise self.error

        return RunnableLambda(boom)


class NamedModel(ScriptedChatModel):
    """A scripted model that reports its name, like a real provider client."""

    model_name: str = "gpt-test"
    cached: int | None = None

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        result = super()._generate(*args, **kwargs)
        if self.cached is not None:
            message = result.generations[0].message
            assert isinstance(message, AIMessage)
            assert message.usage_metadata is not None
            message.usage_metadata["input_token_details"] = {"cache_read": self.cached}
        return result
