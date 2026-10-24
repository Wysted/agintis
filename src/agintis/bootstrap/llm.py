"""LLM wiring shared by the container and the CLIs (offline: local files only, no network)."""

import os
from collections.abc import Callable, Mapping
from functools import lru_cache

from dotenv import dotenv_values

from agintis.bootstrap.settings import ENV_PREFIXES, Settings
from agintis.platform.llm import (
    ChatModels,
    ChatRole,
    LlmConfig,
    ModelCatalog,
    load_llm_profiles,
)

EnvLookup = Callable[[str], str | None]


@lru_cache(maxsize=1)
def _dotenv_values() -> Mapping[str, str | None]:
    return dotenv_values(".env")


def process_env(name: str) -> str | None:
    """A variable of the process environment; defined-but-empty counts as unset."""
    return os.environ.get(name) or None


def process_and_dotenv_env(name: str) -> str | None:
    """Value of an environment variable for an LLM profile's `api_key_env`.

    The process environment wins, and a variable DEFINED there (even empty = unset) never falls
    through to `.env`; `.env` is the fallback because the settings groups read it themselves
    (without exporting it), so a key kept next to the other settings must be found.
    """
    if name in os.environ:
        return process_env(name)
    return _dotenv_values().get(name) or None


def build_chat_models(settings: Settings, environ: EnvLookup | None = None) -> ChatModels:
    """The platform defaults plus the named profiles of `LLM_PROFILES_FILE`.

    A profile's `api_key_env` is looked up when an agent binds the profile. Isolated settings
    (tests) never read `.env`, here included. A bad profiles file raises
    `InvalidLlmProfilesError`.
    """
    llm = settings.llm
    defaults = (
        LlmConfig(
            provider=llm.provider,
            models={role: llm.model_for(role) for role in ChatRole},
            api_key=llm.api_key.get_secret_value() if llm.api_key else None,
            base_url=llm.base_url,
            temperature=llm.temperature,
            timeout_seconds=llm.timeout_seconds,
            max_retries=llm.max_retries,
            stream_usage=llm.stream_usage,
            # The default effort is for openai only (Settings refuses an explicit one elsewhere).
            reasoning_effort=llm.reasoning_effort if llm.provider == "openai" else None,
        )
        if llm.provider
        else None
    )
    profiles = (
        load_llm_profiles(
            llm.profiles_file,
            forbidden_env_prefixes=ENV_PREFIXES,
            allow_insecure_urls=settings.explicit_local_env,
        )
        if llm.profiles_file
        else {}
    )
    lookup = environ or (process_and_dotenv_env if settings.reads_dotenv else process_env)
    return ChatModels(ModelCatalog(defaults, profiles, environ=lookup))


# The agent id the judge's model view is registered under (it shows in error messages).
JUDGE_AGENT = "answer-judge"
JUDGE_ROLES: tuple[ChatRole, ...] = (ChatRole.ANALYZER,)


def bind_judge_model(chat_models: ChatModels, profile: str) -> ChatModels:
    """The view serving the answer judges with the named LLM `profile`.

    The catalog resolves a profile per ROLE, and the judge is not an agent role: it reuses the
    ANALYZER role, the structured-output one (the judge needs exactly that), instead of adding a
    role that every platform default would have to configure. `profile` `default` therefore
    means the `LLM_*` defaults' ANALYZER model. Binding fails now when the profile is unknown or
    its `api_key_env` is unset; `view.verify()` then builds the client and runs the offline
    structured-output check.
    """
    return chat_models.bind(
        {ChatRole.ANALYZER: profile},
        agent=JUDGE_AGENT,
        roles=JUDGE_ROLES,
        fix_hint="Choose another judge profile (EVAL_JUDGE_PROFILE or --profile).",
    )
