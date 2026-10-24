"""Named LLM profiles: the YAML catalog, per-agent bindings, cache per profile and boot checks."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel, GenericFakeChatModel
from langchain_core.language_models.model_profile import ModelProfile
from langchain_core.messages import AIMessage

from agintis.platform.llm import (
    PROVIDER_KWARGS,
    ChatModels,
    ChatRole,
    InvalidLlmProfilesError,
    LlmConfig,
    LlmError,
    LlmNotConfiguredError,
    LlmProfile,
    ModelCatalog,
    ProviderNotInstalledError,
    StructuredOutputUnsupportedError,
    UnknownProfileError,
    load_llm_profiles,
    parse_llm_profiles,
)
from tests.fakes.llm import ScriptedChatModel
from tests.support.settings import make_settings

DEFAULTS = LlmConfig(
    provider="openai", models={role: f"base-{role.value}" for role in ChatRole}, api_key="sk-base"
)
FAST = LlmProfile(provider="openai", model="gpt-fast", api_key_env="FAST_KEY", temperature=0)
SMART = LlmProfile(provider="anthropic", model="claude-smart", api_key_env="SMART_KEY")
PROFILES = {"fast": FAST, "smart": SMART}
ENV = {"FAST_KEY": "sk-fast", "SMART_KEY": "sk-smart"}


class Recorder:
    """A builder that records every build and returns a fresh scripted model per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, model: str, **kwargs: Any) -> BaseChatModel:
        self.calls.append((model, kwargs))
        return ScriptedChatModel()


def _catalog(
    builder: Recorder | None = None,
    *,
    defaults: LlmConfig | None = DEFAULTS,
    env: dict[str, str] | None = None,
) -> ModelCatalog:
    return ModelCatalog(
        defaults,
        PROFILES,
        environ=(env if env is not None else ENV).get,
        builder=builder or Recorder(),
    )


def _models(builder: Recorder | None = None, **kwargs: Any) -> ChatModels:
    return ChatModels(_catalog(builder, **kwargs))


# --- the YAML file ----------------------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "llm_profiles.yaml"
    path.write_text(text)
    return path


def test_a_profiles_file_declares_named_profiles(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
profiles:
  fast:
    provider: openai
    model: gpt-fast
    api_key_env: FAST_KEY
    temperature: 0
  judge:
    provider: openai
    model: deepseek-chat
    base_url: https://api.deepseek.com/v1/
    api_key_env: JUDGE_KEY
    timeout_seconds: 30
    max_retries: 2
    stream_usage: true
""",
    )

    profiles = load_llm_profiles(path)

    assert profiles["fast"] == LlmProfile(
        provider="openai", model="gpt-fast", api_key_env="FAST_KEY", temperature=0
    )
    assert profiles["judge"].base_url == "https://api.deepseek.com/v1"
    assert (profiles["judge"].timeout_seconds, profiles["judge"].max_retries) == (30, 2)
    assert profiles["judge"].api_key is None  # the key is only ever resolved from the environment


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("profiles: {}", "profiles"),
        ("- not a mapping", "must be a YAML mapping"),
        ("profiles:\n  a: {provider: openai}", "model"),
        ("profiles:\n  a: {provider: openai, model: m, api_key: sk-inline}", "api_key"),
        ("profiles:\n  a: {provider: 'openai:x', model: m}", "provider"),
        (
            "profiles:\n  a: {provider: openai, model: m, base_url: nope, api_key_env: K}",
            "base_url",
        ),
        ("profiles:\n  a: {provider: openai, model: m, api_key_env: 'not a name'}", "api_key_env"),
        ("profiles:\n  a: {provider: openai, model: m, temperature: 3}", "temperature"),
        (
            "profiles:\n  a: {provider: openai, model: m, api_key_env: K, reasoning_effort: big}",
            "reasoning_effort",
        ),
        ("profiles:\n  default: {provider: openai, model: m, api_key_env: K}", "reserved"),
        (
            "profiles:\n  Bad Name: {provider: openai, model: m, api_key_env: K}",
            "invalid profile name",
        ),
        ("profiles: {a: {provider: openai, model: m}}\nextra: 1", "extra"),
    ],
)
def test_an_invalid_profiles_file_names_the_problem(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(InvalidLlmProfilesError, match=message):
        load_llm_profiles(_write(tmp_path, text))


def test_a_missing_or_duplicated_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(InvalidLlmProfilesError, match="does not exist"):
        load_llm_profiles(tmp_path / "nope.yaml")
    with pytest.raises(InvalidLlmProfilesError, match="duplicate key"):
        load_llm_profiles(_write(tmp_path, "profiles:\n  a: {provider: p, model: m}\n  a: {}\n"))


def test_parse_accepts_plain_data() -> None:
    raw = {"profiles": {"a": {"provider": "ollama", "model": "llama"}}}
    assert parse_llm_profiles(raw, origin="x") == {"a": LlmProfile("ollama", "llama")}


def test_the_setting_reads_the_file_path_and_empty_means_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert make_settings().llm.profiles_file is None
    monkeypatch.setenv("LLM_PROFILES_FILE", "case/x/llm_profiles.yaml")
    assert make_settings().llm.profiles_file == Path("case/x/llm_profiles.yaml")
    monkeypatch.setenv("LLM_PROFILES_FILE", "")
    assert make_settings().llm.profiles_file is None


# --- backward compatibility -------------------------------------------------------------------


def test_without_bindings_every_role_uses_the_platform_defaults() -> None:
    builder = Recorder()
    models = ChatModels(DEFAULTS, builder)

    for role in ChatRole:
        models.get(role)

    assert [model for model, _ in builder.calls] == [f"base-{r.value}" for r in ChatRole]
    assert builder.calls[0][1] == {"model_provider": "openai", "api_key": "sk-base"}
    assert models.describe(ChatRole.ROUTER).label == "default (openai:base-router)"


def test_bind_without_bindings_is_the_same_view() -> None:
    models = _models()
    assert models.bind({}, agent="a") is models


# --- bindings ---------------------------------------------------------------------------------


def test_bound_roles_use_their_profile_and_the_rest_the_defaults() -> None:
    builder = Recorder()
    bound = _models(builder).bind({ChatRole.ANALYZER: "fast"}, agent="docs")

    bound.get(ChatRole.ANALYZER)
    bound.get(ChatRole.SYNTHESIS)

    assert builder.calls == [
        (
            "gpt-fast",
            {
                "model_provider": "openai",
                "api_key": "sk-fast",
                "base_url": "https://api.openai.com/v1",  # pinned: never the environment's
                "temperature": 0,
                "stream_usage": True,  # what langchain-openai would default to without base_url
            },
        ),
        ("base-synthesis", {"model_provider": "openai", "api_key": "sk-base"}),
    ]
    assert bound.describe(ChatRole.ANALYZER).label == "fast (openai:gpt-fast)"
    assert bound.describe(ChatRole.SYNTHESIS).label == "default (openai:base-synthesis)"


def test_the_default_profile_can_be_named_explicitly() -> None:
    bound = _models().bind({ChatRole.ANALYZER: "default"}, agent="docs")
    assert bound.describe(ChatRole.ANALYZER).profile == "default"


def test_providers_can_be_mixed_inside_one_agent() -> None:
    builder = Recorder()
    bound = _models(builder).bind(
        {ChatRole.ANALYZER: "fast", ChatRole.SYNTHESIS: "smart"}, agent="docs"
    )

    bound.get(ChatRole.ANALYZER)
    bound.get(ChatRole.SYNTHESIS)

    assert [(m, kw["model_provider"], kw["api_key"]) for m, kw in builder.calls] == [
        ("gpt-fast", "openai", "sk-fast"),
        ("claude-smart", "anthropic", "sk-smart"),
    ]


def test_clients_are_cached_per_profile_across_roles_and_agents() -> None:
    builder = Recorder()
    models = _models(builder)
    one = models.bind({ChatRole.ANALYZER: "fast", ChatRole.SYNTHESIS: "fast"}, agent="one")
    two = models.bind({ChatRole.ROUTER: "fast"}, agent="two")

    clients = {
        id(one.get(ChatRole.ANALYZER)),
        id(one.get(ChatRole.SYNTHESIS)),
        id(two.get(ChatRole.ROUTER)),
    }

    assert len(clients) == 1
    assert len(builder.calls) == 1


def test_default_roles_on_one_model_share_a_client() -> None:
    builder = Recorder()
    config = LlmConfig(provider="openai", models={role: "same" for role in ChatRole})
    models = ChatModels(config, builder)

    assert models.get(ChatRole.ROUTER) is models.get(ChatRole.SYNTHESIS)
    assert len(builder.calls) == 1


def test_an_agent_can_run_on_profiles_alone_when_the_defaults_are_off() -> None:
    models = _models(defaults=None)
    bound = models.bind({role: "fast" for role in ChatRole}, agent="docs")

    assert not models.enabled
    bound.verify()  # structured-output check on the bound roles only
    assert bound.get(ChatRole.ANALYZER) is bound.get(ChatRole.SYNTHESIS)
    with pytest.raises(LlmNotConfiguredError, match="agent 'docs'"):
        _models(defaults=None).bind({ChatRole.ANALYZER: "fast"}, agent="docs").get(
            ChatRole.SYNTHESIS
        )


# --- boot validation --------------------------------------------------------------------------


def test_an_unknown_profile_names_the_profile_the_agent_and_the_known_ones() -> None:
    with pytest.raises(UnknownProfileError) as caught:
        _models().bind({ChatRole.SYNTHESIS: "smrt"}, agent="docs")
    message = str(caught.value)
    assert "'smrt'" in message
    assert "'docs'" in message
    assert "'synthesis'" in message
    assert "fast, smart" in message


def test_an_unset_api_key_env_names_the_profile_the_agent_and_the_variable() -> None:
    models = _models(env={"FAST_KEY": "sk-fast"})
    with pytest.raises(LlmError, match=r"profile 'smart' of agent 'docs'.*'SMART_KEY'"):
        models.bind({ChatRole.SYNTHESIS: "smart"}, agent="docs")
    models.bind({ChatRole.SYNTHESIS: "fast"}, agent="docs")  # the others are not checked


def test_an_empty_api_key_env_counts_as_unset() -> None:
    with pytest.raises(LlmError, match="SMART_KEY"):
        _models(env={"SMART_KEY": ""}).bind({ChatRole.SYNTHESIS: "smart"}, agent="docs")


def test_an_ollama_profile_builds_without_a_key() -> None:
    builder = Recorder()
    models = ChatModels(
        ModelCatalog(
            DEFAULTS,
            {"local": LlmProfile("ollama", "llama", base_url="http://h:11434")},
            builder=builder,
        )
    )
    models.bind({ChatRole.ROUTER: "local"}, agent="a").get(ChatRole.ROUTER)
    assert builder.calls[0][1] == {"model_provider": "ollama", "base_url": "http://h:11434"}


def test_a_missing_provider_extra_names_the_profile_the_agent_and_the_extra() -> None:
    def missing(model: str, **kwargs: Any) -> BaseChatModel:
        raise ImportError("langchain-anthropic missing")

    bound = ChatModels(ModelCatalog(DEFAULTS, PROFILES, environ=ENV.get, builder=missing)).bind(
        {ChatRole.SYNTHESIS: "smart"}, agent="docs"
    )

    with pytest.raises(
        ProviderNotInstalledError,
        match=r"profile 'smart' of agent 'docs'.*uv sync --extra anthropic",
    ):
        bound.get(ChatRole.SYNTHESIS)


def test_verify_builds_the_bound_views_too() -> None:
    def missing(model: str, **kwargs: Any) -> BaseChatModel:
        if kwargs["model_provider"] == "anthropic":
            raise ImportError("missing")
        return ScriptedChatModel()

    models = ChatModels(ModelCatalog(DEFAULTS, PROFILES, environ=ENV.get, builder=missing))
    models.bind({ChatRole.SYNTHESIS: "smart"}, agent="docs")

    with pytest.raises(ProviderNotInstalledError, match="agent 'docs'"):
        models.verify()


class _NoStructured(GenericFakeChatModel):
    """A model whose offline profile says it cannot do structured output."""


def _messages() -> Iterator[AIMessage]:
    return iter([AIMessage(content="ok")])


def test_the_structured_output_check_runs_per_profile_and_names_it() -> None:
    profile: ModelProfile = {"structured_output": False, "tool_calling": False}

    def builder(model: str, **kwargs: Any) -> BaseChatModel:
        if model == "gpt-fast":
            return _NoStructured(messages=_messages(), profile=profile)
        return ScriptedChatModel()

    models = ChatModels(ModelCatalog(DEFAULTS, PROFILES, environ=ENV.get, builder=builder))
    models.bind({ChatRole.ANALYZER: "fast"}, agent="docs")

    with pytest.raises(StructuredOutputUnsupportedError) as caught:
        models.verify()
    message = str(caught.value)
    assert "profile 'fast' of agent 'docs'" in message
    assert "openai:gpt-fast" in message
    assert "LLM_ANALYZER_MODEL" not in message  # the fix is another profile, not the env default


def test_a_default_role_without_structured_output_still_points_at_the_env_setting() -> None:
    profile: ModelProfile = {"structured_output": False, "tool_calling": False}
    models = ChatModels(
        DEFAULTS, lambda *_a, **_k: _NoStructured(messages=_messages(), profile=profile)
    )
    with pytest.raises(StructuredOutputUnsupportedError, match="LLM_ANALYZER_MODEL"):
        models.verify()


def test_a_bound_profile_is_checked_for_structured_output_only_on_the_analyzer() -> None:
    profile: ModelProfile = {"structured_output": False, "tool_calling": False}

    def builder(model: str, **kwargs: Any) -> BaseChatModel:
        if model == "claude-smart":
            return _NoStructured(messages=_messages(), profile=profile)
        return ScriptedChatModel()

    models = ChatModels(ModelCatalog(DEFAULTS, PROFILES, environ=ENV.get, builder=builder))
    models.bind({ChatRole.SYNTHESIS: "smart"}, agent="docs")
    models.verify()  # synthesis writes text: no structured output needed


# --- review hardening -------------------------------------------------------------------------


def _one(**fields: Any) -> dict[str, Any]:
    return {"profiles": {"p": {"provider": "openai", "model": "m", **fields}}}


@pytest.mark.parametrize("provider", ["openai", "anthropic", "deepseek"])
def test_a_keyed_provider_without_its_own_key_variable_is_refused_at_parse(provider: str) -> None:
    with pytest.raises(InvalidLlmProfilesError, match="api_key_env is required"):
        parse_llm_profiles(_one(provider=provider), origin="x")
    raw = _one(provider=provider, base_url="https://api.example.com/v1")
    with pytest.raises(InvalidLlmProfilesError, match="api_key_env is required"):
        parse_llm_profiles(raw, origin="x")
    parse_llm_profiles(
        _one(provider=provider, base_url="https://h/v1", api_key_env="K"), origin="x"
    )


def test_a_local_ollama_server_needs_no_key_variable() -> None:
    raw = _one(provider="ollama", base_url="http://localhost:11434")
    assert parse_llm_profiles(raw, origin="x")["p"].base_url == "http://localhost:11434"


@pytest.mark.parametrize("effort", ["none", "minimal", "medium", "max"])
def test_an_openai_profile_passes_its_reasoning_effort(effort: str) -> None:
    profile = parse_llm_profiles(_one(api_key_env="K", reasoning_effort=effort), origin="x")["p"]
    assert profile.reasoning_effort == effort
    assert PROVIDER_KWARGS["openai"](profile)["reasoning_effort"] == effort


def test_an_unset_reasoning_effort_is_not_passed() -> None:
    profile = parse_llm_profiles(_one(api_key_env="K"), origin="x")["p"]
    assert profile.reasoning_effort is None
    assert "reasoning_effort" not in PROVIDER_KWARGS["openai"](profile)


def test_a_deepseek_profile_accepts_reasoning_effort() -> None:
    raw = _one(provider="deepseek", api_key_env="K", reasoning_effort="none")
    assert parse_llm_profiles(raw, origin="x")["p"].reasoning_effort == "none"


@pytest.mark.parametrize("provider", ["anthropic", "ollama", "someprovider"])
def test_reasoning_effort_is_refused_at_parse_for_other_providers(provider: str) -> None:
    raw = _one(provider=provider, api_key_env="K", reasoning_effort="low")
    with pytest.raises(InvalidLlmProfilesError, match="reasoning_effort is only supported"):
        parse_llm_profiles(raw, origin="x")


@pytest.mark.parametrize(
    "env", ["DATABASE_URL", "auth_audience", "LLM_API_KEY", "APP_ENV", "LLM", "database"]
)
def test_api_key_env_cannot_name_an_internal_setting(env: str) -> None:
    prefixes = ("DATABASE_", "AUTH_", "LLM_", "APP_")
    with pytest.raises(InvalidLlmProfilesError, match="own settings"):
        parse_llm_profiles(_one(api_key_env=env), origin="x", forbidden_env_prefixes=prefixes)
    parse_llm_profiles(
        _one(api_key_env="DEEPSEEK_API_KEY"), origin="x", forbidden_env_prefixes=prefixes
    )


def test_every_settings_prefix_is_forbidden_for_api_key_env() -> None:
    from agintis.bootstrap.settings import _GROUPS, ENV_PREFIXES, Settings

    flat = {n.split("_")[0].upper() + "_" for n in Settings.model_fields if n not in _GROUPS}
    assert {"APP_", "DATABASE_", "LOG_", "STARTUP_"} <= flat  # derived from the Settings fields
    assert flat <= set(ENV_PREFIXES)
    assert {"AUTH_", "LLM_", "OBSERVABILITY_", "KNOWLEDGE_", "EMBEDDINGS_"} <= set(ENV_PREFIXES)
    assert {"POSTGRES_", "TEST_DATABASE_"} <= set(ENV_PREFIXES)


def test_a_plain_http_base_url_is_refused_except_localhost_or_local_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plain = _one(base_url="http://llm.internal:8000/v1", api_key_env="K")
    with pytest.raises(InvalidLlmProfilesError, match="must be https"):
        parse_llm_profiles(plain, origin="x")
    for host in ("localhost", "127.0.0.1", "[::1]"):
        parse_llm_profiles(_one(base_url=f"http://{host}:8000/v1", api_key_env="K"), origin="x")
    with caplog.at_level("WARNING"):
        parse_llm_profiles(plain, origin="x", allow_insecure_urls=True)
    assert "not https" in caplog.text


def test_credentials_in_a_base_url_are_never_echoed() -> None:
    raw = _one(base_url="ftp://user:hunter2@host/v1", api_key_env="K")
    with pytest.raises(InvalidLlmProfilesError) as caught:
        parse_llm_profiles(raw, origin="x")
    assert "hunter2" not in str(caught.value)


def test_a_defined_but_empty_key_variable_is_an_error() -> None:
    with pytest.raises(LlmError, match="FAST_KEY"):
        _models(env={"FAST_KEY": ""}).bind({ChatRole.ANALYZER: "fast"}, agent="docs")


def test_a_catalog_and_a_builder_cannot_be_combined() -> None:
    with pytest.raises(TypeError, match="builder"):
        ChatModels(_catalog(), Recorder())


def test_real_chat_openai_gets_the_profiles_own_base_url_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-process-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.example/v1")
    judge = LlmProfile(
        "openai",
        "deepseek-chat",
        api_key_env="JUDGE_KEY",
        base_url="https://api.deepseek.example/v1",
    )
    models = ChatModels(ModelCatalog(None, {"judge": judge}, environ={"JUDGE_KEY": "sk-judge"}.get))

    client = models.bind({ChatRole.ANALYZER: "judge"}, agent="a").get(ChatRole.ANALYZER)

    assert client.openai_api_key.get_secret_value() == "sk-judge"  # type: ignore[attr-defined]
    assert client.openai_api_base == "https://api.deepseek.example/v1"  # type: ignore[attr-defined]


def test_real_chat_openai_gets_the_profiles_reasoning_effort() -> None:
    pytest.importorskip("langchain_openai")
    luna = LlmProfile("openai", "gpt-6-luna", api_key_env="K", reasoning_effort="none")
    models = ChatModels(ModelCatalog(None, {"luna": luna}, environ={"K": "sk-luna"}.get))

    client = models.bind({ChatRole.ANALYZER: "luna"}, agent="a").get(ChatRole.ANALYZER)

    assert client.reasoning_effort == "none"  # type: ignore[attr-defined]


# --- views ------------------------------------------------------------------------------------


def test_views_are_deduplicated_by_agent_bindings_and_roles() -> None:
    models = _models()
    roles = (ChatRole.ANALYZER, ChatRole.SYNTHESIS)
    one = models.bind({ChatRole.ANALYZER: "fast"}, "docs", roles=roles)

    assert models.bind({ChatRole.ANALYZER: "fast"}, "docs", roles=roles) is one
    assert one.bind({ChatRole.ANALYZER: "fast"}, "docs", roles=roles) is one  # from a view too
    assert models.bind({ChatRole.ANALYZER: "fast"}, "other", roles=roles) is not one
    assert len(models.catalog.views) == 2


def test_a_view_bound_from_another_view_is_still_verified_by_the_root() -> None:
    def builder(model: str, **kwargs: Any) -> BaseChatModel:
        if kwargs["model_provider"] == "anthropic":
            raise ImportError("missing")
        return ScriptedChatModel()

    models = ChatModels(ModelCatalog(DEFAULTS, PROFILES, environ=ENV.get, builder=builder))
    models.bind({ChatRole.ANALYZER: "fast"}, "a", roles=tuple(ChatRole)).bind(
        {ChatRole.SYNTHESIS: "smart"}, "b", roles=tuple(ChatRole)
    )

    with pytest.raises(ProviderNotInstalledError, match="agent 'b'"):
        models.verify()


def test_bindings_outside_the_views_roles_are_ignored() -> None:
    builder = Recorder()
    models = _models(builder)

    view = models.bind({ChatRole.ROUTER: "nope"}, "a", roles=(ChatRole.ANALYZER,))
    view.verify()

    assert [m for m, _ in builder.calls] == ["base-analyzer"]  # only the roles the agent uses


def test_a_view_that_uses_the_router_checks_its_structured_output_too() -> None:
    profile: ModelProfile = {"structured_output": False, "tool_calling": False}

    def builder(model: str, **kwargs: Any) -> BaseChatModel:
        if model == "gpt-fast":
            return _NoStructured(messages=_messages(), profile=profile)
        return ScriptedChatModel()

    uses_router = (ChatRole.ANALYZER, ChatRole.SYNTHESIS, ChatRole.ROUTER)
    llm_router = _models_with(builder)
    llm_router.bind({ChatRole.ROUTER: "fast"}, "llm-router", roles=uses_router)
    evidence = _models_with(builder)
    evidence.bind({ChatRole.ROUTER: "fast"}, "evidence", roles=uses_router[:2])  # router unused

    with pytest.raises(StructuredOutputUnsupportedError, match="agent 'llm-router'"):
        llm_router.verify()
    evidence.verify()  # the evidence agent never calls the router: nothing to check


def test_the_registry_of_views_is_read_only() -> None:
    with pytest.raises(TypeError):
        _models().catalog.views[("x",)] = _models()  # type: ignore[index]


def _models_with(builder: Any) -> ChatModels:
    return ChatModels(ModelCatalog(DEFAULTS, PROFILES, environ=ENV.get, builder=builder))


def test_a_keyless_provider_needs_no_key_variable_even_with_a_base_url() -> None:
    parse_llm_profiles(_one(provider="ollama", base_url="http://localhost:11434"), origin="x")
    parse_llm_profiles(_one(provider="ollama"), origin="x")
