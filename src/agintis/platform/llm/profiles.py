"""The catalog of named LLM profiles, declared in a YAML file (primitives out, no Settings here).

```yaml
profiles:
  fast:
    provider: openai
    model: gpt-5.4-mini
    api_key_env: OPENAI_API_KEY      # the NAME of the variable, never the key itself
    temperature: 0
  thinker:
    provider: openai
    model: gpt-6-luna
    api_key_env: OPENAI_API_KEY
    reasoning_effort: none           # openai and deepseek only; models accept subsets
    # with any effort but `none`, OpenAI reasoning models reject `temperature`: leave it unset
  judge:
    provider: deepseek               # needs the `deepseek` extra; see the note below
    model: deepseek-flash
    reasoning_effort: none           # deepseek-flash thinks by default
    api_key_env: DEEPSEEK_API_KEY
    temperature: 0
    timeout_seconds: 60
  local:
    provider: openai                 # any other OpenAI-compatible server is `openai` + `base_url`
    model: some-model
    base_url: https://llm.internal.example/v1
    api_key_env: INTERNAL_LLM_KEY
```

DeepSeek is its own provider (not `openai` + `base_url`): its API accepts only the `text` and
`json_object` response formats, while `ChatOpenAI.with_structured_output` defaults to
`json_schema` (a 400 there); `ChatDeepSeek` uses function calling. Models that do not support tools
(`deepseek-reasoner`) cannot serve a structured-output role.

A profile of a keyed provider (openai, anthropic, deepseek: `DEFAULT_BASE_URLS`) MUST name its
`api_key_env`, checked when the file is parsed: without it the integration would fall back to the
process's own OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY (and OPENAI_BASE_URL and the
like), possibly shipping that key to another server. A named profile also always pins its server
(the official one when it names no `base_url`). A `base_url` of any other provider (ollama) needs no
key. `base_url` must be https (localhost excepted; local mode may allow plain http), and
`api_key_env` cannot name one of the application's own settings.

A profile carries exactly the options `PROVIDER_KWARGS` knows; an option left out is not passed
(the integration's own default applies, nothing is inherited from the `LLM_*` defaults). The name
`default` is reserved for the implicit platform profile (`LLM_*` settings).
"""

import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError

from agintis.platform.llm.models import DEFAULT_BASE_URLS, DEFAULT_PROFILE, LlmError, LlmProfile
from agintis.platform.yaml_config import YamlConfigError, load_yaml_file

PROFILE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
ENV_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
MAX_PROFILES = 64
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# Providers whose server may be local and keyless (a dummy key is still fine).
KEYLESS_BASE_URL_PROVIDERS = frozenset({"ollama"})
REASONING_EFFORT_PROVIDERS = frozenset({"openai", "deepseek"})
# OpenAI's `reasoning.effort` values (each model accepts a subset; unset = the model's default).
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

logger = logging.getLogger(__name__)

_HTTP_URL = TypeAdapter(AnyHttpUrl)


class InvalidLlmProfilesError(LlmError):
    """The profiles file is malformed (boot fails)."""


def _redacted(value: str) -> str:
    """The URL without credentials (`user:pass@`), safe to put in an error message."""
    return re.sub(r"(//)[^/@]*@", r"\1***@", value)


def _http_url(value: str) -> str:
    try:
        _HTTP_URL.validate_python(value)
    except ValueError:
        raise ValueError(f"must be a valid http(s) URL: {_redacted(value)!r}") from None
    return value.rstrip("/")


class LlmProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: str = Field(min_length=1, pattern=r"^[^:\s]+$")
    model: str = Field(min_length=1)
    base_url: Annotated[str, AfterValidator(_http_url)] | None = None
    api_key_env: str | None = Field(default=None, pattern=ENV_NAME_PATTERN)
    temperature: float | None = Field(default=None, ge=0, le=2)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, ge=0)
    stream_usage: bool | None = None
    reasoning_effort: ReasoningEffort | None = None

    @model_validator(mode="after")
    def _reasoning_effort_needs_a_mapper(self) -> Self:
        if self.reasoning_effort is not None and self.provider not in REASONING_EFFORT_PROVIDERS:
            raise ValueError(
                "reasoning_effort is only supported by providers "
                f"{sorted(REASONING_EFFORT_PROVIDERS)}, not {self.provider!r}: "
                "unset it (agintis only maps it for those)"
            )
        return self

    @model_validator(mode="after")
    def _keyed_providers_name_their_key(self) -> Self:
        if self.provider in DEFAULT_BASE_URLS and self.api_key_env is None:
            raise ValueError(
                f"api_key_env is required for provider {self.provider!r}: a named profile never "
                "falls back to the provider's own environment variable (OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, DEEPSEEK_API_KEY), which could be shipped to another server"
            )
        return self

    @model_validator(mode="after")
    def _base_url_needs_its_own_key(self) -> Self:
        if (
            self.base_url is not None
            and self.api_key_env is None
            and self.provider not in KEYLESS_BASE_URL_PROVIDERS
        ):
            raise ValueError(
                "api_key_env is required with base_url: otherwise the process's own provider key "
                "(e.g. OPENAI_API_KEY) would be sent to that server (a local server can name a "
                "variable holding a dummy key)"
            )
        return self

    def to_profile(self) -> LlmProfile:
        return LlmProfile(**self.model_dump())


class LlmProfilesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    profiles: Mapping[str, LlmProfileSpec] = Field(min_length=1, max_length=MAX_PROFILES)


def parse_llm_profiles(
    raw: object,
    *,
    origin: str,
    forbidden_env_prefixes: Iterable[str] = (),
    allow_insecure_urls: bool = False,
) -> dict[str, LlmProfile]:
    """`forbidden_env_prefixes`: `api_key_env` may not start with any of them (the application's
    own settings). `allow_insecure_urls`: accept plain-http `base_url`s of any host (local mode)."""
    if not isinstance(raw, dict):
        raise InvalidLlmProfilesError(f"{origin}: an LLM profiles file must be a YAML mapping")
    try:
        parsed = LlmProfilesFile.model_validate(raw)
    except PydanticValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False)
        )
        raise InvalidLlmProfilesError(f"{origin}: {details}") from exc
    for name in parsed.profiles:
        if name == DEFAULT_PROFILE:
            raise InvalidLlmProfilesError(
                f"{origin}: profile name {DEFAULT_PROFILE!r} is reserved for the LLM_* defaults"
            )
        if not PROFILE_NAME_PATTERN.fullmatch(name):
            raise InvalidLlmProfilesError(
                f"{origin}: invalid profile name {name!r} (lowercase letters, digits, '-' and '_', "
                "starting with a letter, at most 63 characters)"
            )
    prefixes = tuple(prefix.upper() for prefix in forbidden_env_prefixes)
    for name, spec in parsed.profiles.items():
        env = spec.api_key_env
        if env is not None and _is_reserved(env, prefixes):
            raise InvalidLlmProfilesError(
                f"{origin}: profile {name!r}: api_key_env {env!r} names one of the application's "
                "own settings; use a variable dedicated to this key (e.g. DEEPSEEK_API_KEY)"
            )
        if spec.base_url is not None:
            _check_transport(name, spec.base_url, origin, allow_insecure_urls)
    return {name: spec.to_profile() for name, spec in parsed.profiles.items()}


def _is_reserved(env: str, prefixes: tuple[str, ...]) -> bool:
    """`env` starts with a reserved prefix (`LLM_`) or is its bare word (`LLM`)."""
    name = env.upper()
    return any(name == prefix.rstrip("_") or name.startswith(prefix) for prefix in prefixes)


def _check_transport(name: str, url: str, origin: str, allow_insecure: bool) -> None:
    parts = urlsplit(url)
    if parts.scheme == "https" or parts.hostname in LOCAL_HOSTS:
        return
    if not allow_insecure:
        raise InvalidLlmProfilesError(
            f"{origin}: profile {name!r}: base_url must be https (the API key and the prompts "
            "travel over it); plain http is accepted only for localhost or with an explicit "
            "APP_ENV=local"
        )
    logger.warning("LLM profile %r: base_url is not https (allowed by APP_ENV=local)", name)


def load_llm_profiles(
    path: Path,
    *,
    forbidden_env_prefixes: Iterable[str] = (),
    allow_insecure_urls: bool = False,
) -> dict[str, LlmProfile]:
    """The profiles of a YAML file; `InvalidLlmProfilesError` names the file on any problem."""
    if not path.is_file():
        raise InvalidLlmProfilesError(f"LLM profiles file {str(path)!r} does not exist")
    try:
        raw = load_yaml_file(path)
    except YamlConfigError as exc:
        raise InvalidLlmProfilesError(str(exc)) from exc
    return parse_llm_profiles(
        raw,
        origin=path.name,
        forbidden_env_prefixes=forbidden_env_prefixes,
        allow_insecure_urls=allow_insecure_urls,
    )
