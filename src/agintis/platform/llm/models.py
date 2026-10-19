"""Chat models per role, built from primitives (no Settings here).

The port is `langchain_core.language_models.BaseChatModel` itself: nothing wraps it. This module
only decides WHICH model each role gets and fails early, with an actionable message, when a
provider integration is missing or the analyzer model cannot produce structured output.

A role is served by an `LlmProfile` (provider + model + client options). The platform defaults
(`LlmConfig`, one implicit profile named `default` per role) serve every role an agent does not
bind; an agent may bind a role to a NAMED profile of the catalog (`ModelCatalog.profiles`, see
`profiles.py`). Clients are cached per profile, so roles and agents sharing one share the client.
"""

import logging
import os
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Provider -> uv extra that installs its langchain integration.
PROVIDER_EXTRAS: Mapping[str, str] = MappingProxyType(
    {"openai": "openai", "anthropic": "anthropic", "ollama": "ollama", "deepseek": "deepseek"}
)

ChatModelBuilder = Callable[..., BaseChatModel]

# Providers whose integration falls back to PROCESS environment variables for the server and the
# key (OPENAI_BASE_URL / OPENAI_API_KEY, ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY, DEEPSEEK_API_BASE
# / DEEPSEEK_API_KEY): with a redirected server and no explicit key the process's own key would be
# shipped to that host. A NAMED profile therefore always pins both (`ModelCatalog` fills the
# official base URL and demands a key resolved from its `api_key_env`); the implicit `default`
# profile keeps the integrations' env behaviour (see `LlmConfig`).
DEFAULT_BASE_URLS: Mapping[str, str] = MappingProxyType(
    {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "deepseek": "https://api.deepseek.com/v1",
    }
)


class ChatRole(StrEnum):
    """What a model is used for; each role may be served by a different model."""

    ROUTER = "router"
    ANALYZER = "analyzer"
    SYNTHESIS = "synthesis"


# The analyzer extracts structured data from the question, so it must support structured output.
STRUCTURED_OUTPUT_ROLES: frozenset[ChatRole] = frozenset({ChatRole.ANALYZER})


class LlmError(RuntimeError):
    """The LLM layer is misconfigured."""


class LlmNotConfiguredError(LlmError):
    """The LLM capability is disabled (`LLM_PROVIDER` unset)."""


class ProviderNotInstalledError(LlmError):
    """The langchain integration of the configured provider is not installed."""


class StructuredOutputUnsupportedError(LlmError):
    """A role that needs structured output got a model that cannot provide it."""


class _StructuredProbe(BaseModel):
    """Throwaway schema used only to check that `with_structured_output` binds."""

    value: str


DEFAULT_PROFILE = "default"


@dataclass(frozen=True)
class LlmProfile:
    """Everything needed to build ONE chat client (primitives only).

    `api_key_env` is the NAME of the environment variable holding the key; the catalog resolves it
    into `api_key` when the profile is bound (secrets never live in a committable file).
    """

    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    stream_usage: bool | None = None


@dataclass(frozen=True)
class LlmConfig:
    """The platform defaults: one provider, a model per role (the `default` profile).

    Backward compatible on purpose: an option left unset is NOT pinned, so the provider
    integration's own environment fallback applies (OPENAI_API_KEY, OPENAI_BASE_URL,
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL). The operator who sets OPENAI_BASE_URL in the process
    environment redirects the default profile (and sends it the key it resolves) by design;
    named profiles are immune to that (see `DEFAULT_BASE_URLS`). Set `LLM_BASE_URL` and
    `LLM_API_KEY` explicitly to pin the default as well. `deepseek` is the exception: it has no
    legacy to keep, so it always needs an explicit key.
    """

    provider: str
    models: Mapping[ChatRole, str]
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    temperature: float | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    stream_usage: bool | None = None

    def __post_init__(self) -> None:
        missing = set(ChatRole) - set(self.models)
        if missing:
            raise LlmError(f"No model configured for roles: {sorted(missing)}")
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))

    def profile_for(self, role: ChatRole) -> LlmProfile:
        return LlmProfile(
            provider=self.provider,
            model=self.models[role],
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            stream_usage=self.stream_usage,
        )


@dataclass(frozen=True)
class ResolvedModel:
    """What a role resolves to (for reports): the profile name and the model behind it."""

    role: ChatRole
    profile: str
    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.profile} ({self.provider}:{self.model})"


def _openai_kwargs(cfg: LlmProfile) -> dict[str, Any]:
    kwargs = _generic_kwargs(cfg)
    if cfg.stream_usage is not None:
        kwargs["stream_usage"] = cfg.stream_usage
    return kwargs


def _generic_kwargs(cfg: LlmProfile) -> dict[str, Any]:
    optional = {
        "api_key": cfg.api_key,
        "base_url": cfg.base_url,
        "temperature": cfg.temperature,
        "timeout": cfg.timeout_seconds,
        "max_retries": cfg.max_retries,
    }
    return {key: value for key, value in optional.items() if value is not None}


def _ollama_kwargs(cfg: LlmProfile) -> dict[str, Any]:
    # ChatOllama has no api_key / max_retries and takes the timeout through its httpx client.
    unsupported = [
        name
        for name, value in (("api_key", cfg.api_key), ("max_retries", cfg.max_retries))
        if value is not None
    ]
    if unsupported:
        raise LlmError(
            f"Provider 'ollama' does not support {', '.join(unsupported)}: unset them "
            "(LLM_API_KEY / LLM_MAX_RETRIES for the platform default, the profile entry otherwise)"
        )
    kwargs: dict[str, Any] = {}
    if cfg.base_url is not None:
        kwargs["base_url"] = cfg.base_url
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if cfg.timeout_seconds is not None:
        kwargs["client_kwargs"] = {"timeout": cfg.timeout_seconds}
    return kwargs


def _deepseek_kwargs(cfg: LlmProfile) -> dict[str, Any]:
    # ChatDeepSeek reads DEEPSEEK_API_BASE / DEEPSEEK_API_KEY from the environment when they are
    # not passed, and its client would send the process key wherever DEEPSEEK_API_BASE points:
    # pin both here.
    if not cfg.api_key:
        raise LlmError(
            "Provider 'deepseek' needs an explicit API key (api_key_env of the profile, or "
            "LLM_API_KEY for the platform default): the environment is never consulted"
        )
    kwargs = _generic_kwargs(cfg)
    kwargs["base_url"] = cfg.base_url or DEFAULT_BASE_URLS["deepseek"]
    return kwargs


KwargsMapper = Callable[[LlmProfile], dict[str, Any]]

# Provider -> mapping of the generic options to that integration's constructor kwargs.
# openai and anthropic take the generic names as they are; deepseek pins its server and key;
# unknown providers pass through. `deepseek` (langchain-deepseek) is NOT `openai` + a base_url on
# purpose: DeepSeek's API has no `json_schema` response format, which is what
# `ChatOpenAI.with_structured_output` uses by default; `ChatDeepSeek` falls back to function
# calling there.
PROVIDER_KWARGS: Mapping[str, KwargsMapper] = MappingProxyType(
    {
        "openai": _openai_kwargs,
        "anthropic": _generic_kwargs,
        "ollama": _ollama_kwargs,
        "deepseek": _deepseek_kwargs,
    }
)


def _where(profile: str, agent: str | None) -> str:
    return f"profile {profile!r}" + (f" of agent {agent!r}" if agent else "")


def verify_structured_output(
    model: BaseChatModel,
    role: ChatRole,
    provider: str | None = None,
    *,
    profile: str | None = None,
    agent: str | None = None,
    model_id: str | None = None,
    hint: str | None = None,
) -> None:
    """Offline check (no network, no cost) that `model` can serve a structured-output role.

    Only `model.profile` is an offline signal: an explicit `structured_output: false` without
    `tool_calling` fails fast. `with_structured_output` binds even for models that cannot do it
    (OpenAI, Ollama), so a failed bind is just an extra sanity check. A model without profile
    data (local OpenAI-compatible servers, new ids) passes with a WARNING: the real guarantee is
    an eval or an optional online smoke test, not this check.

    `profile` / `agent` / `model_id` only name the culprit in the messages (the LLM profile, not
    the model profile above); `hint` replaces the default "what to change" sentence for callers
    that are not an agent spec (the answer judge).
    """
    name = model_id or getattr(model, "model_name", None) or getattr(model, "model", None)
    label = f"{provider}:{name}" if provider else str(name or type(model).__name__)
    named = profile is not None and profile != DEFAULT_PROFILE
    where = f" [{_where(profile, agent)}]" if profile is not None else ""
    hint = hint or (
        f"Bind another LLM profile to the {role.value!r} role (`models:` of the agent spec)."
        if named
        else f"Configure another model for LLM_{role.name}_MODEL."
    )
    model_profile = model.profile or {}
    structured = model_profile.get("structured_output")
    tool_calling = model_profile.get("tool_calling")
    if structured is False and tool_calling is not True:
        raise StructuredOutputUnsupportedError(
            f"Model {label!r} for role {role.value!r} does not support structured output "
            f"(model profile){where}. {hint}"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # langchain-openai method-default notice
            model.with_structured_output(_StructuredProbe)
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise StructuredOutputUnsupportedError(
            f"Model {label!r} for role {role.value!r} cannot bind structured output{where}: "
            f"{exc}. {hint}"
        ) from exc
    if structured is None and tool_calling is None:
        logger.warning(
            "Cannot verify structured output for %s offline (role %s%s); validated on first call",
            label,
            role.value,
            f", {_where(profile, agent)}" if profile is not None else "",
        )


class UnknownProfileError(LlmError):
    """A role is bound to a profile the catalog does not declare."""


class ModelCatalog:
    """The named profiles, the platform defaults and the client cache (one per process).

    Resolution is cheap and offline (`resolve` also checks the profile exists and its
    `api_key_env` variable is set); `client` builds lazily and caches PER PROFILE, so two roles
    or two agents that resolve to the same profile share one client.
    """

    def __init__(
        self,
        defaults: LlmConfig | None,
        profiles: Mapping[str, LlmProfile] | None = None,
        *,
        environ: Callable[[str], str | None] = lambda name: os.environ.get(name),
        builder: ChatModelBuilder = init_chat_model,
    ) -> None:
        self._defaults = defaults
        self._profiles = MappingProxyType(dict(profiles or {}))
        self._environ = environ
        self._builder = builder
        self._resolved: dict[str, LlmProfile] = {}
        self._clients: dict[LlmProfile, BaseChatModel] = {}
        # Per-agent views (see ChatModels.bind), deduplicated by their key.
        self._views: dict[tuple[Any, ...], ChatModels] = {}

    @property
    def views(self) -> Mapping[tuple[Any, ...], "ChatModels"]:
        """The registered per-agent views (read-only; `ChatModels.bind` registers them)."""
        return MappingProxyType(self._views)

    def register_view(self, key: tuple[Any, ...], view: "ChatModels") -> None:
        self._views[key] = view

    @property
    def enabled(self) -> bool:
        """The platform defaults are configured (`LLM_PROVIDER` set)."""
        return self._defaults is not None

    @property
    def profile_names(self) -> list[str]:
        return sorted(self._profiles)

    def resolve(
        self,
        role: ChatRole,
        bindings: Mapping[ChatRole, str] | None = None,
        agent: str | None = None,
    ) -> tuple[str, LlmProfile]:
        """The (profile name, profile) serving `role` for an agent with these `bindings`."""
        name = (bindings or {}).get(role, DEFAULT_PROFILE)
        if name == DEFAULT_PROFILE:
            if self._defaults is None:
                raise LlmNotConfiguredError(
                    "LLM not configured: set LLM_PROVIDER"
                    + (
                        f" (agent {agent!r} leaves role {role.value!r} on the default)"
                        if agent
                        else ""
                    )
                )
            return name, self._defaults.profile_for(role)
        return name, self._named(name, role, agent)

    def _named(self, name: str, role: ChatRole, agent: str | None) -> LlmProfile:
        cached = self._resolved.get(name)
        if cached is not None:
            return cached
        profile = self._profiles.get(name)
        if profile is None:
            known = ", ".join(self.profile_names) or "none declared (LLM_PROFILES_FILE)"
            raise UnknownProfileError(
                f"Unknown LLM profile {name!r} bound to role {role.value!r}"
                + (f" of agent {agent!r}" if agent else "")
                + f": known profiles: {known}"
            )
        if profile.api_key_env is not None:
            key = self._environ(profile.api_key_env)
            if not key:
                raise LlmError(
                    f"LLM {_where(name, agent)}: the environment variable "
                    f"{profile.api_key_env!r} (api_key_env) is not set"
                )
            profile = replace(profile, api_key=key)
        profile = self._pinned(name, profile, agent)
        self._resolved[name] = profile
        return profile

    @staticmethod
    def _pinned(name: str, profile: LlmProfile, agent: str | None) -> LlmProfile:
        """A named profile of a keyed provider never leans on the process environment: it needs
        its own key and always carries an explicit server (the official one when it names none)."""
        provider = profile.provider
        if provider not in DEFAULT_BASE_URLS:
            return profile
        if not profile.api_key:
            raise LlmError(
                f"LLM {_where(name, agent)}: provider {provider!r} needs `api_key_env` (a named "
                "profile never falls back to the provider's own environment variable)"
            )
        if profile.base_url is not None:
            return profile
        changes: dict[str, Any] = {"base_url": DEFAULT_BASE_URLS[provider]}
        if provider == "openai" and profile.stream_usage is None:
            # langchain-openai only turns stream_usage on by default WITHOUT a base_url.
            changes["stream_usage"] = True
        return replace(profile, **changes)

    def client(
        self, profile: LlmProfile, *, name: str, role: ChatRole, agent: str | None = None
    ) -> BaseChatModel:
        model = self._clients.get(profile)
        if model is None:
            model = self._clients.setdefault(profile, self._build(profile, name, role, agent))
        return model

    def _build(
        self, profile: LlmProfile, name: str, role: ChatRole, agent: str | None
    ) -> BaseChatModel:
        where = _where(name, agent)
        provider = profile.provider
        if profile.stream_usage is not None and provider != "openai":
            raise LlmError(
                f"Provider {provider!r} does not support stream_usage ({where}): "
                "unset it (LLM_STREAM_USAGE for the platform default)"
            )
        mapper = PROVIDER_KWARGS.get(provider, _generic_kwargs)
        kwargs = {"model_provider": provider, **mapper(profile)}
        try:
            return self._builder(profile.model, **kwargs)
        except ImportError as exc:
            extra = PROVIDER_EXTRAS.get(provider)
            hint = f"uv sync --extra {extra}" if extra else f"install the integration: {exc}"
            raise ProviderNotInstalledError(
                f"LLM provider {provider!r} is not installed ({where}): run `{hint}`"
            ) from exc
        except Exception as exc:
            # Provider SDKs raise their own errors (e.g. openai.OpenAIError for a missing key).
            raise LlmError(
                f"Cannot build the {role.value!r} chat model {profile.model!r} "
                f"(provider {provider!r}, {where}): {exc}"
            ) from exc


# Roles whose model also writes structured output when a view USES them (the llm router binds
# its schema like the analyzer does; the merge tie-break also uses ROUTER, but as plain text);
# the platform defaults only check the analyzer.
# shortcut: agents that only use merge_tiebreak (plain text) are also structured-checked for
# ROUTER, upgrade when a tie-break-only agent needs a non-structured router model
VIEW_STRUCTURED_ROLES: frozenset[ChatRole] = STRUCTURED_OUTPUT_ROLES | {ChatRole.ROUTER}


class ChatModels:
    """One chat model per `ChatRole`, built lazily and cached per profile.

    `config=None` means the LLM capability is disabled (`LLM_PROVIDER` unset): `verify()` is a
    no-op and `get()` raises `LlmError`, so processes that never call an LLM (ingest and eval
    CLIs also run the boot checks) need no provider, extra or API key. A `ModelCatalog` may be
    passed instead of a config (profiles declared next to the defaults); `builder` only applies
    to a config.

    `bind(bindings, agent, roles=)` returns the view of ONE agent: its bound roles are served by
    the named profiles, the others by the platform defaults; `roles` are the roles the agent
    really calls (bindings of other roles are ignored, never resolved or built). Nodes only ever
    call `get(role)`. Views are registered in the catalog, deduplicated by (agent, bindings,
    roles).

    `verify()` of the root view builds every role and also verifies every registered view (the
    roles it uses, with the structured-output check of the router too); a view verifies only
    itself. With the checks off, the first `get()` builds the client inside the running loop
    (accepted: construction does no network).
    """

    def __init__(
        self,
        config: LlmConfig | ModelCatalog | None,
        builder: ChatModelBuilder | None = None,
    ) -> None:
        if isinstance(config, ModelCatalog):
            if builder is not None:
                raise TypeError("pass the builder to the ModelCatalog, not to ChatModels")
            self._catalog = config
        else:
            self._catalog = ModelCatalog(config, builder=builder or init_chat_model)
        self._bindings: Mapping[ChatRole, str] = MappingProxyType({})
        self._agent: str | None = None
        self._scope: frozenset[ChatRole] | None = None  # None = the root view
        self._fix_hint: str | None = None

    @property
    def enabled(self) -> bool:
        return self._catalog.enabled

    @property
    def catalog(self) -> ModelCatalog:
        return self._catalog

    def bind(
        self,
        bindings: Mapping[ChatRole, str],
        agent: str | None = None,
        *,
        roles: Iterable[ChatRole] | None = None,
        fix_hint: str | None = None,
    ) -> "ChatModels":
        """The view of an agent whose `bindings` (role -> profile name) override the defaults.

        Fails now (at wiring time) when a bound profile is unknown or its `api_key_env` is not
        set. Without `roles` and without bindings this is the root view itself. `fix_hint`
        replaces the "bind another profile under `models:`" advice of the structured-output
        check, for views that no agent spec owns.
        """
        scope = None if roles is None else frozenset(roles)
        used = {r: n for r, n in bindings.items() if scope is None or r in scope}
        if not used and scope is None:
            return self
        key = (agent, frozenset(used.items()), scope, fix_hint)
        existing = self._catalog.views.get(key)
        if existing is not None:
            return existing
        for role, name in used.items():
            if name != DEFAULT_PROFILE:  # the defaults are checked by verify() (they may be off)
                self._catalog.resolve(role, used, agent)
        view = ChatModels(self._catalog)
        view._bindings = MappingProxyType(used)
        view._agent = agent
        view._scope = scope
        view._fix_hint = fix_hint
        self._catalog.register_view(key, view)
        return view

    def describe(self, role: ChatRole) -> ResolvedModel:
        """The profile and model `role` resolves to (no client is built)."""
        name, profile = self._catalog.resolve(role, self._bindings, self._agent)
        return ResolvedModel(role, name, profile.provider, profile.model)

    def get(self, role: ChatRole) -> BaseChatModel:
        name, profile = self._catalog.resolve(role, self._bindings, self._agent)
        return self._catalog.client(profile, name=name, role=role, agent=self._agent)

    def __getitem__(self, role: ChatRole) -> BaseChatModel:
        return self.get(role)

    def verify(self) -> None:
        """Build the roles and check the structured-output ones (no-op if disabled and unbound)."""
        self._verify_roles()
        if self._scope is None:
            for view in tuple(self._catalog.views.values()):
                view._verify_roles()

    def _verify_roles(self) -> None:
        scope = self._scope
        if self._catalog.enabled:
            roles: Iterable[ChatRole] = tuple(ChatRole) if scope is None else sorted(scope)
        else:
            roles = [r for r in self._bindings if scope is None or r in scope]
        structured = STRUCTURED_OUTPUT_ROLES if scope is None else VIEW_STRUCTURED_ROLES
        for role in roles:
            name, profile = self._catalog.resolve(role, self._bindings, self._agent)
            model = self._catalog.client(profile, name=name, role=role, agent=self._agent)
            if role in structured:
                verify_structured_output(
                    model,
                    role,
                    profile.provider,
                    profile=name,
                    agent=self._agent,
                    model_id=profile.model,
                    hint=self._fix_hint,
                )
