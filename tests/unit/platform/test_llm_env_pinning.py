"""A NAMED profile never takes its server or key from the process environment (the same bug class
as an OpenAI key shipped to a foreign `base_url`): the real integrations are built offline."""

import pytest
from pydantic import SecretStr

from agintis.platform.llm import (
    ChatModels,
    ChatRole,
    LlmConfig,
    LlmError,
    LlmProfile,
    ModelCatalog,
)

EVIL = "https://evil.example/v1"
ENV_KEYS = {
    "OPENAI_BASE_URL": EVIL,
    "OPENAI_API_BASE": EVIL,
    "OPENAI_API_KEY": "env-openai-key",
    "ANTHROPIC_BASE_URL": EVIL,
    "ANTHROPIC_API_KEY": "env-anthropic-key",
    "DEEPSEEK_API_BASE": EVIL,
    "DEEPSEEK_API_KEY": "env-deepseek-key",
}


@pytest.fixture(autouse=True)
def _hostile_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in ENV_KEYS.items():
        monkeypatch.setenv(name, value)


def _named(profile: LlmProfile, defaults: LlmConfig | None = None) -> ChatModels:
    catalog = ModelCatalog(defaults, {"p": profile}, environ={"P_KEY": "profile-key"}.get)
    return ChatModels(catalog).bind({ChatRole.SYNTHESIS: "p"}, agent="a")


def _key(secret: object) -> str:
    return secret.get_secret_value() if isinstance(secret, SecretStr) else str(secret)


def test_a_named_openai_profile_ignores_openai_base_url_and_key() -> None:
    model = _named(LlmProfile("openai", "gpt-x", api_key_env="P_KEY")).get(ChatRole.SYNTHESIS)

    assert model.openai_api_base == "https://api.openai.com/v1"  # type: ignore[attr-defined]
    assert _key(model.openai_api_key) == "profile-key"  # type: ignore[attr-defined]
    assert model.stream_usage is True  # type: ignore[attr-defined]  # kept as without a base_url


def test_a_named_openai_profile_keeps_its_own_base_url_and_stream_usage_choice() -> None:
    profile = LlmProfile(
        "openai",
        "m",
        api_key_env="P_KEY",
        base_url="https://llm.internal/v1",
        stream_usage=False,
    )
    model = _named(profile).get(ChatRole.SYNTHESIS)
    assert model.openai_api_base == "https://llm.internal/v1"  # type: ignore[attr-defined]
    assert model.stream_usage is False  # type: ignore[attr-defined]


def test_a_named_anthropic_profile_ignores_anthropic_base_url_and_key() -> None:
    model = _named(LlmProfile("anthropic", "claude-x", api_key_env="P_KEY")).get(ChatRole.SYNTHESIS)

    assert model.anthropic_api_url == "https://api.anthropic.com"  # type: ignore[attr-defined]
    assert _key(model.anthropic_api_key) == "profile-key"  # type: ignore[attr-defined]


def test_a_named_deepseek_profile_ignores_deepseek_api_base_and_key() -> None:
    pytest.importorskip("langchain_deepseek")
    model = _named(LlmProfile("deepseek", "deepseek-chat", api_key_env="P_KEY")).get(
        ChatRole.SYNTHESIS
    )

    assert model.api_base == "https://api.deepseek.com/v1"  # type: ignore[attr-defined]
    assert _key(model.api_key) == "profile-key"  # type: ignore[attr-defined]


def test_a_named_deepseek_profile_keeps_a_base_url_it_names() -> None:
    pytest.importorskip("langchain_deepseek")
    profile = LlmProfile("deepseek", "m", api_key_env="P_KEY", base_url="https://proxy.example/v1")
    model = _named(profile).get(ChatRole.SYNTHESIS)
    assert model.api_base == "https://proxy.example/v1"  # type: ignore[attr-defined]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "deepseek"])
def test_a_named_profile_without_its_own_key_fails_at_bind_not_with_the_env_key(
    provider: str,
) -> None:
    with pytest.raises(LlmError, match="api_key_env"):
        _named(LlmProfile(provider, "m"))


def test_ollama_profiles_stay_keyless() -> None:
    pytest.importorskip("langchain_ollama")
    model = _named(LlmProfile("ollama", "llama")).get(ChatRole.SYNTHESIS)
    assert type(model).__name__ == "ChatOllama"


def test_the_deepseek_default_profile_also_demands_an_explicit_key() -> None:
    pytest.importorskip("langchain_deepseek")
    defaults = LlmConfig(provider="deepseek", models={r: "deepseek-chat" for r in ChatRole})
    with pytest.raises(LlmError, match="explicit API key"):
        ChatModels(ModelCatalog(defaults)).get(ChatRole.SYNTHESIS)

    keyed = LlmConfig(
        provider="deepseek", models={r: "deepseek-chat" for r in ChatRole}, api_key="cfg-key"
    )
    model = ChatModels(ModelCatalog(keyed)).get(ChatRole.SYNTHESIS)
    assert model.api_base == "https://api.deepseek.com/v1"  # type: ignore[attr-defined]
    assert _key(model.api_key) == "cfg-key"  # type: ignore[attr-defined]


def test_the_implicit_default_profile_keeps_the_integrations_environment_fallback() -> None:
    """Documented backward compatibility (see `LlmConfig`): the platform default is NOT pinned,
    so OPENAI_BASE_URL in the process environment redirects it by design."""
    defaults = LlmConfig(provider="openai", models={r: "gpt-x" for r in ChatRole})
    model = ChatModels(ModelCatalog(defaults)).get(ChatRole.SYNTHESIS)
    assert model.openai_api_base == EVIL  # type: ignore[attr-defined]


def test_a_deepseek_profile_sends_reasoning_effort_in_the_request_body() -> None:
    pytest.importorskip("langchain_deepseek")
    profile = LlmProfile("deepseek", "deepseek-flash", api_key_env="P_KEY", reasoning_effort="none")
    model = _named(profile).get(ChatRole.SYNTHESIS)

    payload = model._get_request_payload("hi")  # type: ignore[attr-defined]

    assert payload["reasoning_effort"] == "none"
