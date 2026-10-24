import logging
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel, GenericFakeChatModel
from langchain_core.language_models.model_profile import ModelProfile
from langchain_core.messages import AIMessage

from agintis.bootstrap.llm import build_chat_models
from agintis.platform.llm import (
    PROVIDER_KWARGS,
    ChatModels,
    ChatRole,
    LlmConfig,
    LlmError,
    ProviderNotInstalledError,
    StructuredOutputUnsupportedError,
    verify_structured_output,
)
from tests.support.settings import make_settings


class StructuredFake(GenericFakeChatModel):
    """Fake chat model that accepts `with_structured_output` (the base fake does not)."""

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self


def _messages() -> Iterator[AIMessage]:
    return iter([AIMessage(content="ok")])


def _config(**overrides: Any) -> LlmConfig:
    values: dict[str, Any] = {
        "provider": "openai",
        "models": {role: f"model-{role.value}" for role in ChatRole},
    }
    values.update(overrides)
    return LlmConfig(**values)


class RecordingBuilder:
    def __init__(self, model: BaseChatModel | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._model = model

    def __call__(self, model: str, **kwargs: Any) -> BaseChatModel:
        self.calls.append((model, kwargs))
        return self._model or StructuredFake(messages=_messages())


# --- settings ---------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PROVIDER", "MODEL", "API_KEY", "BASE_URL", "TEMPERATURE", "TIMEOUT_SECONDS"):
        monkeypatch.delenv(f"LLM_{name}", raising=False)
    for name in ("MAX_RETRIES", "ROUTER_MODEL", "ANALYZER_MODEL", "SYNTHESIS_MODEL"):
        monkeypatch.delenv(f"LLM_{name}", raising=False)
    for name in ("REASONING_EFFORT", "PROFILES_FILE"):
        monkeypatch.delenv(f"LLM_{name}", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_settings_defaults_and_role_fallback() -> None:
    llm = make_settings().llm
    assert llm.provider is None
    assert llm.api_key is None
    assert llm.temperature is None
    assert {llm.model_for(role) for role in ChatRole} == {llm.model}


def test_settings_role_overrides_fall_back_to_model() -> None:
    llm = make_settings(llm={"model": "base", "analyzer_model": "strong"}).llm
    assert llm.model_for(ChatRole.ANALYZER) == "strong"
    assert llm.model_for(ChatRole.ROUTER) == "base"
    assert llm.model_for(ChatRole.SYNTHESIS) == "base"


def test_settings_empty_strings_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PROVIDER", "API_KEY", "BASE_URL", "TEMPERATURE", "MAX_RETRIES", "ROUTER_MODEL"):
        monkeypatch.setenv(f"LLM_{name}", "")
    llm = make_settings().llm
    assert llm.provider is None
    assert llm.api_key is None
    assert llm.base_url is None
    assert llm.temperature is None
    assert llm.max_retries is None
    assert llm.model_for(ChatRole.ROUTER) == llm.model


def test_settings_read_flat_env_and_keep_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1/")
    llm = make_settings().llm
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "sk-secret"
    assert "sk-secret" not in repr(llm)
    assert llm.base_url == "http://localhost:1234/v1"


@pytest.mark.parametrize("provider", ["openai:gpt", "a b"])
def test_settings_reject_invalid_provider(provider: str) -> None:
    with pytest.raises(ValueError, match="provider"):
        make_settings(llm={"provider": provider})


def test_settings_default_to_luna_with_reasoning_off() -> None:
    llm = make_settings().llm
    assert llm.model == "gpt-6-luna"
    assert llm.reasoning_effort == "none"


def test_settings_empty_reasoning_effort_means_the_api_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_REASONING_EFFORT", "")
    llm = make_settings(llm={"provider": "anthropic"}).llm
    assert llm.reasoning_effort is None


@pytest.mark.parametrize("provider", ["anthropic", "ollama", "deepseek"])
def test_settings_refuse_an_explicit_reasoning_effort_for_other_providers(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    with pytest.raises(ValueError, match="LLM_REASONING_EFFORT is only supported"):
        make_settings(llm={"provider": provider})


def _default_kwargs(**llm: Any) -> dict[str, Any]:
    """The client kwargs the platform default profile maps to, built from the settings."""
    profile = build_chat_models(make_settings(llm=llm)).catalog.resolve(ChatRole.SYNTHESIS)[1]
    return PROVIDER_KWARGS[profile.provider](profile)


@pytest.mark.parametrize(
    "llm",
    [{"provider": "anthropic"}, {"provider": "ollama"}, {"provider": "deepseek", "api_key": "k"}],
    ids=lambda llm: llm["provider"],
)
def test_the_default_reasoning_effort_never_reaches_other_providers(llm: dict[str, Any]) -> None:
    assert "reasoning_effort" not in _default_kwargs(**llm)


def test_the_openai_default_turns_reasoning_off() -> None:
    assert _default_kwargs(provider="openai")["reasoning_effort"] == "none"


def test_an_unset_reasoning_effort_is_not_passed_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_REASONING_EFFORT", "")
    assert "reasoning_effort" not in _default_kwargs(provider="openai")


# --- factory ----------------------------------------------------------------------------------


def test_models_are_built_lazily_once_per_role() -> None:
    builder = RecordingBuilder()
    models = ChatModels(_config(api_key="sk-x", temperature=0.2, max_retries=1), builder)
    assert builder.calls == []
    first = models.get(ChatRole.ROUTER)
    assert models[ChatRole.ROUTER] is first
    assert builder.calls == [
        (
            "model-router",
            {"model_provider": "openai", "api_key": "sk-x", "temperature": 0.2, "max_retries": 1},
        )
    ]


def test_unset_options_are_not_passed() -> None:
    builder = RecordingBuilder()
    ChatModels(_config(), builder).get(ChatRole.SYNTHESIS)
    assert builder.calls[0][1] == {"model_provider": "openai"}


def test_config_requires_every_role() -> None:
    with pytest.raises(LlmError, match="roles"):
        LlmConfig(provider="openai", models={ChatRole.ROUTER: "m"})


def test_config_repr_hides_the_api_key() -> None:
    assert "sk-secret" not in repr(_config(api_key="sk-secret"))


def test_missing_provider_extra_names_the_uv_extra() -> None:
    def missing(model: str, **kwargs: Any) -> BaseChatModel:
        raise ImportError("langchain-anthropic missing")

    models = ChatModels(_config(provider="anthropic"), missing)
    with pytest.raises(ProviderNotInstalledError, match="uv sync --extra anthropic"):
        models.get(ChatRole.ROUTER)


def test_unknown_provider_is_a_clear_error() -> None:
    def unknown(model: str, **kwargs: Any) -> BaseChatModel:
        raise ValueError("Unable to infer model provider")

    with pytest.raises(LlmError, match="provider 'nope'"):
        ChatModels(_config(provider="nope"), unknown).get(ChatRole.ROUTER)


# --- structured output check ------------------------------------------------------------------


def test_verify_passes_when_the_analyzer_binds_structured_output() -> None:
    builder = RecordingBuilder()
    ChatModels(_config(), builder).verify()
    assert len(builder.calls) == len(ChatRole)


def test_verify_fails_when_the_analyzer_cannot_bind_structured_output() -> None:
    builder = RecordingBuilder(GenericFakeChatModel(messages=_messages()))
    with pytest.raises(StructuredOutputUnsupportedError, match="analyzer"):
        ChatModels(_config(), builder).verify()


def test_profile_without_structured_output_fails_fast() -> None:
    profile: ModelProfile = {"structured_output": False}
    model = StructuredFake(messages=_messages(), profile=profile)
    with pytest.raises(StructuredOutputUnsupportedError, match="LLM_ANALYZER_MODEL"):
        verify_structured_output(model, ChatRole.ANALYZER)


def test_verify_rejects_profile_without_structured_output_and_tool_calling() -> None:
    profile: ModelProfile = {"structured_output": False, "tool_calling": False}
    model = StructuredFake(messages=_messages(), profile=profile)
    with pytest.raises(StructuredOutputUnsupportedError):
        verify_structured_output(model, ChatRole.ANALYZER)


def test_tool_calling_models_pass_even_if_structured_output_is_false() -> None:
    profile: ModelProfile = {"structured_output": False, "tool_calling": True}
    verify_structured_output(
        StructuredFake(messages=_messages(), profile=profile), ChatRole.ANALYZER
    )


def test_unknown_profile_warns_and_passes(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="agintis.platform.llm.models"):
        verify_structured_output(StructuredFake(messages=_messages()), ChatRole.ANALYZER, "openai")
    assert "Cannot verify structured output for openai" in caplog.text


def test_known_profile_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    profile: ModelProfile = {"structured_output": True}
    with caplog.at_level(logging.WARNING, logger="agintis.platform.llm.models"):
        verify_structured_output(
            StructuredFake(messages=_messages(), profile=profile), ChatRole.ANALYZER
        )
    assert caplog.text == ""


# --- optional capability, builder errors, per-provider kwargs ----------------------------------


def test_disabled_models_verify_is_a_noop_and_get_raises() -> None:
    builder = RecordingBuilder()
    models = ChatModels(None, builder)
    models.verify()
    assert not models.enabled
    with pytest.raises(LlmError, match="LLM_PROVIDER"):
        models.get(ChatRole.ROUTER)
    assert builder.calls == []


def test_any_builder_error_becomes_llm_error_without_the_key() -> None:
    class SdkError(Exception):
        pass

    def no_key(model: str, **kwargs: Any) -> BaseChatModel:
        raise SdkError("api_key must be set")

    with pytest.raises(LlmError, match="api_key must be set") as caught:
        ChatModels(_config(api_key="sk-secret"), no_key).get(ChatRole.ROUTER)
    assert "sk-secret" not in str(caught.value)


def test_config_models_are_read_only() -> None:
    config = _config()
    with pytest.raises(TypeError):
        config.models[ChatRole.ROUTER] = "x"  # type: ignore[index]


def test_ollama_maps_timeout_and_base_url_to_client_kwargs() -> None:
    builder = RecordingBuilder()
    config = _config(provider="ollama", base_url="http://h:11434", timeout_seconds=7.0)
    ChatModels(config, builder).get(ChatRole.ROUTER)
    assert builder.calls[0][1] == {
        "model_provider": "ollama",
        "base_url": "http://h:11434",
        "client_kwargs": {"timeout": 7.0},
    }


@pytest.mark.parametrize("option", [{"api_key": "k"}, {"max_retries": 1}])
def test_ollama_rejects_unsupported_options(option: dict[str, Any]) -> None:
    builder = RecordingBuilder()
    with pytest.raises(LlmError, match="ollama"):
        ChatModels(_config(provider="ollama", **option), builder).get(ChatRole.ROUTER)
    assert builder.calls == []


def test_anthropic_and_unknown_providers_get_the_generic_kwargs() -> None:
    for provider in ("anthropic", "someprovider"):
        builder = RecordingBuilder()
        config = _config(provider=provider, api_key="k", timeout_seconds=3.0, max_retries=1)
        ChatModels(config, builder).get(ChatRole.ROUTER)
        assert builder.calls[0][1] == {
            "model_provider": provider,
            "api_key": "k",
            "timeout": 3.0,
            "max_retries": 1,
        }


def test_openai_maps_stream_usage() -> None:
    builder = RecordingBuilder()
    ChatModels(_config(stream_usage=True), builder).get(ChatRole.ROUTER)
    assert builder.calls[0][1]["stream_usage"] is True


def test_stream_usage_unset_is_not_passed() -> None:
    builder = RecordingBuilder()
    ChatModels(_config(), builder).get(ChatRole.ROUTER)
    assert "stream_usage" not in builder.calls[0][1]


@pytest.mark.parametrize("provider", ["anthropic", "ollama"])
def test_stream_usage_is_rejected_for_other_providers(provider: str) -> None:
    builder = RecordingBuilder()
    with pytest.raises(LlmError, match="stream_usage"):
        ChatModels(_config(provider=provider, stream_usage=False), builder).get(ChatRole.ROUTER)
    assert builder.calls == []
