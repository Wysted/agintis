"""Contract of the real langchain-openai adapter: construction and binding are offline."""

from agintis.platform.llm import ChatModels, ChatRole, LlmConfig


def _config(**overrides: object) -> LlmConfig:
    values: dict[str, object] = {
        "provider": "openai",
        "models": {role: "gpt-6-luna" for role in ChatRole},
        "api_key": "sk-dummy",
    }
    values.update(overrides)
    return LlmConfig(**values)  # type: ignore[arg-type]


def test_real_openai_model_is_built_offline_with_a_dummy_key() -> None:
    models = ChatModels(
        _config(
            base_url="http://localhost:1234/v1", temperature=0.0, timeout_seconds=5, max_retries=0
        )
    )
    model = models.get(ChatRole.ANALYZER)
    assert type(model).__name__ == "ChatOpenAI"
    assert getattr(model, "openai_api_base", None) == "http://localhost:1234/v1"


def test_real_openai_analyzer_passes_the_offline_check() -> None:
    ChatModels(_config()).verify()


def test_real_openai_without_key_is_an_llm_error() -> None:
    import pytest

    from agintis.platform.llm import LlmError

    with pytest.raises(LlmError, match="Cannot build"):
        ChatModels(_config(api_key=None)).get(ChatRole.ROUTER)
