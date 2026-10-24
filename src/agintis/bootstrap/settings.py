from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    Field,
    PrivateAttr,
    SecretStr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from agintis.modules.knowledge.api import DEFAULT_RERANK_CANDIDATES
from agintis.modules.knowledge.infrastructure.parsers.limits import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_COMPRESSION_RATIO,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_UNCOMPRESSED_BYTES,
)
from agintis.platform.auth import DEFAULT_ALGORITHMS, disallowed_algorithms
from agintis.platform.http.tenant import TENANT_PATTERN
from agintis.platform.llm import ChatRole
from agintis.platform.llm.profiles import ReasoningEffort
from agintis.platform.observability import parse_headers

SQLALCHEMY_PSYCOPG_SCHEME = "postgresql+psycopg://"
LIBPQ_SCHEME = "postgresql://"

_HTTP_URL = TypeAdapter(AnyHttpUrl)


def _normalize_http_url(value: str) -> str:
    """Validate with pydantic's AnyHttpUrl (scheme, host, port) but keep the operator's spelling.

    `str(AnyHttpUrl(...))` rewrites the URL (drops the default port, adds a trailing slash);
    only the trailing slash is removed here so base URLs join cleanly with request paths.
    """
    try:
        _HTTP_URL.validate_python(value)
    except ValueError as exc:
        raise ValueError(f"must be a valid http(s) URL: {value!r}") from exc
    return value.rstrip("/")


HttpUrl = Annotated[str, AfterValidator(_normalize_http_url)]


def _verbatim_http_url(value: str) -> str:
    """Validate an http(s) URL and keep it exactly as written (signed or exact-match endpoints)."""
    try:
        _HTTP_URL.validate_python(value)
    except ValueError as exc:
        raise ValueError(f"must be a valid http(s) URL: {value!r}") from exc
    return value


VerbatimHttpUrl = Annotated[str, AfterValidator(_verbatim_http_url)]


class EmbeddingsSettings(BaseSettings):
    """TEI embeddings service. Env: EMBEDDINGS_URL, EMBEDDINGS_MODEL, EMBEDDINGS_TOKENIZER, ..."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDINGS_", env_file=".env", extra="ignore")

    url: HttpUrl = "http://localhost:8081"
    # Model the server is expected to serve (compared with TEI GET /info at startup) and the
    # default tokenizer source. The model actually served is set in docker-compose.
    model: str = "BAAI/bge-m3"
    # HF hub id or local tokenizer.json used to count tokens when chunking (default: `model`).
    # The Docker image bakes the bge-m3 tokenizer and points this at it (offline boot).
    tokenizer: str | None = None
    # Must not exceed the server's --max-client-batch-size (TEI_MAX_CLIENT_BATCH_SIZE).
    batch_size: int = Field(default=16, gt=0)
    # Per-attempt READ timeout; connect/write/pool timeouts are short fixed constants.
    timeout_seconds: float = Field(default=120.0, gt=0)
    # Total budget of one embed call (all batches' retries included for a single request).
    deadline_seconds: float = Field(default=300.0, gt=0)
    # Instruction prefixes some models require (e5: "query: " / "passage: "). Empty for bge-m3.
    query_prefix: str = ""
    document_prefix: str = ""
    # Every returned vector must have exactly this dimension. It is also the width of the
    # `chunks.embedding vector(N)` column created by the first Alembic migration: changing the
    # embedding model to another dimension needs a NEW migration (alter column + re-embed all).
    expected_dim: int = Field(default=1024, gt=0)
    api_key: SecretStr | None = None

    @property
    def tokenizer_source(self) -> str:
        return self.tokenizer or self.model


class RerankerSettings(BaseSettings):
    """Optional TEI reranker. `url` unset = no "tei" reranker registered (profiles use "none")."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_", env_file=".env", extra="ignore")

    url: HttpUrl | None = None
    # Max vector hits scored by the cross-encoder per query (CPU: ~45 s for 16 texts).
    max_candidates: int = Field(default=DEFAULT_RERANK_CANDIDATES, gt=0)
    # Per-attempt READ timeout.
    timeout_seconds: float = Field(default=60.0, gt=0)
    # Total budget of one rerank call; past it retrieval degrades to the vector order.
    deadline_seconds: float = Field(default=90.0, gt=0)
    api_key: SecretStr | None = None

    @field_validator("url", mode="before")
    @classmethod
    def _empty_means_disabled(cls, value: object) -> object:
        return None if value == "" else value


class ParserSettings(BaseSettings):
    """Resource limits of the lightweight parser. Env: PARSER_MAX_BYTES, PARSER_MAX_PAGES, ..."""

    model_config = SettingsConfigDict(env_prefix="PARSER_", env_file=".env", extra="ignore")

    # Largest accepted source file.
    max_bytes: int = Field(default=DEFAULT_MAX_BYTES, gt=0)
    # Largest accepted PDF (pages).
    max_pages: int = Field(default=DEFAULT_MAX_PAGES, gt=0)
    # DOCX zip pre-check: declared uncompressed size and compression ratio (zip bombs).
    max_uncompressed_bytes: int = Field(default=DEFAULT_MAX_UNCOMPRESSED_BYTES, gt=0)
    max_compression_ratio: float = Field(default=DEFAULT_MAX_COMPRESSION_RATIO, gt=1)


class KnowledgeSettings(BaseSettings):
    """Knowledge module configuration. Env: KNOWLEDGE_PROFILES_DIR."""

    model_config = SettingsConfigDict(env_prefix="KNOWLEDGE_", env_file=".env", extra="ignore")

    # Directory with one YAML RagProfile per file, loaded next to the code-defined default
    # profile at boot (None = only the default). Relative paths resolve against the process cwd.
    profiles_dir: Path | None = None
    # Text search configuration used to parse the QUERY of the lexical branch of hybrid search
    # (KNOWLEDGE_TEXT_SEARCH_CONFIG). The `tsv` column itself is built by the migration with
    # `public.es_unaccent`: change this only for a deployment whose migration used another one.
    text_search_config: str = Field(default="public.es_unaccent", min_length=1)
    # Hybrid and bm25_hybrid retrieval: candidates taken from EACH branch, and the RRF constant.
    hybrid_branch_depth: int = Field(default=50, ge=1, le=1000)
    hybrid_rrf_k: int = Field(default=60, ge=1)
    # Longest accepted retrieval query in characters (KNOWLEDGE_MAX_QUERY_CHARS); longer ones are
    # rejected with QueryTooLongError instead of being embedded and parsed.
    max_query_chars: int = Field(default=2000, ge=1)
    # Dense searches of ALL profile probes (evidence router) that may run at once in this process
    # (KNOWLEDGE_PROBE_CONCURRENCY). Each one holds a pooled connection, shared with thread
    # leases and retrieval, so it must leave room: validated <= DATABASE_POOL_SIZE - 1 (min 1).
    probe_concurrency: int = Field(default=2, ge=1)

    @field_validator("profiles_dir", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        return None if value == "" else value


class AgentsSettings(BaseSettings):
    """Configurable agents. Env: AGENTS_DIR."""

    model_config = SettingsConfigDict(env_prefix="AGENTS_", env_file=".env", extra="ignore")

    # Directory with one YAML AgentSpec per file (generic rag-chat agents); each file registers
    # one agent at boot next to direct-chat (None = none). Specs naming an unknown profile or an
    # invalid filter field abort the boot. Relative paths resolve against the process cwd.
    dir: Path | None = None

    @field_validator("dir", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        return None if value == "" else value


DEFAULT_LLM_MODEL = "gpt-6-luna"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0


class LlmSettings(BaseSettings):
    """Chat models. Env: LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_ROUTER_MODEL, ...

    Optional capability: `provider` unset = LLM disabled (boot checks no-op, `get` raises).
    One provider for every role; each role may use its own model id (`*_model`, unset = `model`).
    These are the PLATFORM DEFAULTS (profile `default`): an agent spec may bind roles to named
    profiles of `profiles_file` (other providers/models/keys); unbound roles use the defaults.
    OpenAI-compatible servers (vLLM, LM Studio) are provider `openai` plus `base_url` (and a
    dummy `api_key`: ChatOpenAI refuses to build without one). Anthropic builds without a key
    and fails on the first call. Options a provider cannot honour (api_key/max_retries with
    ollama) are rejected at build.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    # Any `init_chat_model` provider whose uv extra is installed (openai, anthropic, ollama).
    provider: str | None = Field(default=None, min_length=1, pattern=r"^[^:\s]+$")
    model: str = Field(default=DEFAULT_LLM_MODEL, min_length=1)
    api_key: SecretStr | None = None
    base_url: HttpUrl | None = None
    # None = provider default (reasoning models reject a custom temperature).
    temperature: float | None = Field(default=None, ge=0, le=2)
    # Per-request timeout; max_retries None = the SDK's own default.
    timeout_seconds: float = Field(default=DEFAULT_LLM_TIMEOUT_SECONDS, gt=0)
    max_retries: int | None = Field(default=None, ge=0)
    # Ask the provider to report token usage while streaming (openai only; None = integration
    # default). langchain-openai defaults it to True only without a base_url / OPENAI_BASE_URL,
    # so OpenAI-compatible servers need LLM_STREAM_USAGE=true to report usage.
    stream_usage: bool | None = None
    # openai only; "none" = reasoning off (Luna's API default, medium, bills reasoning tokens and
    # failed the spensiones floors). Empty = the API default. The default is dropped for other
    # providers; an explicit value with one is refused below.
    reasoning_effort: ReasoningEffort | None = "none"
    router_model: str | None = Field(default=None, min_length=1)
    # The analyzer needs structured output (checked at boot).
    analyzer_model: str | None = Field(default=None, min_length=1)
    synthesis_model: str | None = Field(default=None, min_length=1)
    # YAML file declaring NAMED profiles an agent spec may bind per role (`models:`); see
    # `agintis.platform.llm.profiles`. None = only the defaults above. Relative paths resolve
    # against the process cwd.
    profiles_file: Path | None = None
    # YAML price catalog of the cost report (`agintis.platform.usage.prices`). None = the file
    # shipped with the package.
    prices_file: Path | None = None

    @field_validator(
        "provider",
        "api_key",
        "base_url",
        "temperature",
        "max_retries",
        "stream_usage",
        "reasoning_effort",
        "router_model",
        "analyzer_model",
        "synthesis_model",
        "profiles_file",
        "prices_file",
        mode="before",
    )
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def _reasoning_effort_is_openai_only(self) -> Self:
        if (
            self.reasoning_effort is not None
            and "reasoning_effort" in self.model_fields_set
            and self.provider not in (None, "openai")
        ):
            raise ValueError(
                f"LLM_REASONING_EFFORT is only supported by LLM_PROVIDER 'openai', not "
                f"{self.provider!r}: set it empty (agintis only maps it for openai)"
            )
        return self

    def model_for(self, role: ChatRole) -> str:
        override: str | None = getattr(self, f"{role.value}_model")
        return override or self.model


class EvalSettings(BaseSettings):
    """Offline answer-quality evaluation (`cli.judge`). Env: EVAL_JUDGE_PROFILE, ...

    The judge model is a NAMED LLM profile (`LLM_PROFILES_FILE`), so it can live on another
    provider than the generator. There is NO default on purpose: a judge that silently falls back
    to the generator's own model would grade its own homework (`cli.judge` also refuses a judge
    whose provider:model equals the answering model unless `--allow-self-judge`). `default`
    may be named explicitly: it means the `LLM_*` defaults' analyzer model.
    """

    model_config = SettingsConfigDict(env_prefix="EVAL_", env_file=".env", extra="ignore")

    judge_profile: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,62}$")
    # Judge calls in flight at once (one structured call each).
    judge_concurrency: int = Field(default=4, ge=1, le=64)

    @field_validator("judge_profile", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        return None if value == "" else value


class ObservabilitySettings(BaseSettings):
    """Tracing. Env: OBSERVABILITY_TRACING, OBSERVABILITY_OTLP_ENDPOINT, ...

    `tracing=none` (default) imports and runs nothing. `otlp` exports OpenTelemetry spans (LangChain
    and LangGraph runs, through the OpenInference instrumentation) to ANY OTLP/HTTP backend
    (Phoenix, Langfuse, LangSmith, a collector) and needs the `otel` uv extra. LangChain's own
    tracing (`LANGSMITH_TRACING`, `LANGCHAIN_TRACING_V2`) must stay off: the boot refuses it.

    `hide_content` redacts, in every span: prompts, completions, retrieved texts and tool I/O
    (span attributes, OpenInference `hide_inputs`/`hide_outputs`) AND the text of errors (status
    description and `exception.message` / `exception.stacktrace` of exception events, redacted by
    our exporter wrapper; `exception.type` is kept). The trace keeps structure, timings, span
    names, model names, token counts and tenant/agent/run ids, but loses what you need to debug
    answer quality and failures; leave it off only for backends you trust with user content.
    """

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", env_file=".env", extra="ignore")

    tracing: Literal["none", "otlp"] = "none"
    # Full OTLP/HTTP traces URL, POSTed to verbatim (Phoenix: http://localhost:6006/v1/traces).
    otlp_endpoint: HttpUrl | None = None
    # `name=value,name2=value2` sent with every export (e.g. Langfuse/LangSmith auth).
    otlp_headers: SecretStr | None = None
    # Per-request timeout of each OTLP export; the shutdown flush is bounded by it plus a grace.
    otlp_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    service_name: str = Field(default="agintis", min_length=1)
    hide_content: bool = False

    @field_validator("otlp_endpoint", "otlp_headers", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def _otlp_needs_an_endpoint(self) -> Self:
        if self.tracing == "otlp" and self.otlp_endpoint is None:
            raise ValueError("OBSERVABILITY_OTLP_ENDPOINT is required when tracing is 'otlp'")
        if self.otlp_headers is not None:
            parse_headers(self.otlp_headers.get_secret_value())  # same error at boot, not mid-run
        return self


DEFAULT_MAX_MESSAGE_CHARS = 4000


class HttpSettings(BaseSettings):
    """HTTP edge. Env: HTTP_DEFAULT_TENANT, HTTP_MAX_MESSAGE_CHARS.

    Request BODY size limits (bytes) belong to the gateway / reverse proxy; the app only caps
    the message length in characters after the body is parsed.
    """

    model_config = SettingsConfigDict(env_prefix="HTTP_", env_file=".env", extra="ignore")

    # Tenant used when a request carries no tenant header. Honoured ONLY when APP_ENV=local was
    # set EXPLICITLY (`Settings.default_tenant_enabled`); otherwise the header is required.
    default_tenant: str | None = Field(default=None, pattern=TENANT_PATTERN.pattern)
    # Longest accepted chat message in characters; longer ones are rejected with 422.
    max_message_chars: int = Field(default=DEFAULT_MAX_MESSAGE_CHARS, ge=1)

    @field_validator("default_tenant", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        return None if value == "" else value


class AuthSettings(BaseSettings):
    """Who the caller is. Env: AUTH_TENANT_RESOLVER, AUTH_ISSUER, AUTH_AUDIENCE, ...

    `header` (default, unchanged behaviour) trusts `X-Tenant-Id`: only behind a trusted gateway.
    `jwt` verifies `Authorization: Bearer <jwt>` against a JWKS and takes the tenant from
    `tenant_claim` (a dotted path reaches nested claims, e.g. `org.tenant`). Any OIDC-style IdP
    works by configuration. The JWKS comes from exactly one of `jwks_url` (production; cached,
    refreshed on TTL expiry or on an unknown `kid`, rate-limited) or `jwks_file` (dev, air-gapped).
    Only asymmetric algorithms are accepted (`none` and HS* are refused here and at verification).

    Notes: there is no `typ` check, so `audience` must NOT be the OAuth client_id (ID tokens carry
    it as `aud` and would pass as access tokens). `GET /v1/agents` and `/health` stay public in
    every mode. FastAPI parses a JSON body before the auth dependency runs: a malformed body
    without a token answers 422, not 401. After the TTL, a FAILING refresh keeps the cached keys
    for `jwks_stale_grace_seconds` (0 = off); a successful refresh never does. A non-https
    `jwks_url` (except localhost) is refused at boot unless `jwks_allow_insecure` or an explicit
    APP_ENV=local; a JWKS FILE with `dev-*` kids (dev_token) is refused outside APP_ENV=local.
    """

    model_config = SettingsConfigDict(env_prefix="AUTH_", env_file=".env", extra="ignore")

    tenant_resolver: Literal["header", "jwt"] = "header"
    issuer: str | None = Field(default=None, min_length=1)
    audience: str | None = Field(default=None, min_length=1)
    jwks_url: VerbatimHttpUrl | None = None
    jwks_file: Path | None = None
    tenant_claim: str = Field(default="tenant_id", min_length=1)
    # Comma-separated in the environment (AUTH_ALGORITHMS=RS256,ES256).
    algorithms: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [*DEFAULT_ALGORITHMS]
    )
    leeway_seconds: int = Field(default=10, ge=0, le=300)
    jwks_cache_ttl_seconds: float = Field(default=600.0, gt=0, le=86400)
    # Minimum time between two JWKS fetch attempts (successful or not).
    jwks_min_refresh_seconds: float = Field(default=30.0, ge=5)
    # Total deadline of one JWKS fetch (also bounds the wait for the single-flight lock).
    jwks_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    # Keep serving cached keys this long past the TTL WHILE the refresh fails (0 = disabled).
    jwks_stale_grace_seconds: float = Field(default=3600.0, ge=0, le=86400)
    # Accept a non-https jwks_url (otherwise only with an explicit APP_ENV=local).
    jwks_allow_insecure: bool = False
    # Longest accepted token lifetime (exp - iat, or exp - now without iat).
    max_token_lifetime_seconds: int = Field(default=86400, ge=60, le=2_592_000)

    @field_validator("issuer", "audience", "jwks_url", "jwks_file", mode="before")
    @classmethod
    def _empty_means_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("algorithms", mode="before")
    @classmethod
    def _split_algorithms(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("algorithms")
    @classmethod
    def _asymmetric_only(cls, value: list[str]) -> list[str]:
        bad = disallowed_algorithms(value)
        if bad or not value:
            raise ValueError(
                f"AUTH_ALGORITHMS must be a non-empty list of asymmetric algorithms "
                f"(RS*, PS*, ES*, EdDSA); 'none' and HS* are never accepted: {bad}"
            )
        return value

    @model_validator(mode="after")
    def _jwt_needs_its_config(self) -> Self:
        if self.jwks_min_refresh_seconds > self.jwks_cache_ttl_seconds:
            raise ValueError(
                "AUTH_JWKS_MIN_REFRESH_SECONDS must not exceed AUTH_JWKS_CACHE_TTL_SECONDS"
            )
        if self.tenant_resolver != "jwt":
            return self
        missing = [
            name
            for name, value in (("AUTH_ISSUER", self.issuer), ("AUTH_AUDIENCE", self.audience))
            if value is None
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} required when AUTH_TENANT_RESOLVER is 'jwt'")
        if (self.jwks_url is None) == (self.jwks_file is None):
            raise ValueError(
                "exactly one of AUTH_JWKS_URL / AUTH_JWKS_FILE is required when "
                "AUTH_TENANT_RESOLVER is 'jwt'"
            )
        return self


class DatabaseSettings(BaseSettings):
    """SQLAlchemy pool knobs (small defaults for low-resource hosts). Env: DATABASE_POOL_SIZE, ...

    The URL itself stays flat in `Settings.database_url` (DATABASE_URL).
    """

    model_config = SettingsConfigDict(env_prefix="DATABASE_", env_file=".env", extra="ignore")

    pool_size: int = Field(default=3, gt=0)
    max_overflow: int = Field(default=2, ge=0)
    # Seconds to wait for a free pooled connection before failing.
    pool_timeout_seconds: float = Field(default=10.0, gt=0)
    # Recycle connections older than this (guards against idle-killed connections).
    pool_recycle_seconds: int = Field(default=1800, gt=0)
    echo: bool = False
    # libpq connect timeout (whole seconds, rounded up) for every connection the app opens.
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    # Upper bound of the whole boot-time database check (connect + queries).
    startup_check_timeout_seconds: float = Field(default=15.0, gt=0)
    # Server-side cap per statement of the pooled engine (None = server default; the ingest-lock
    # connections never get one). Leave it unset unless every pooled statement, ingest writes
    # included, should share one cap: searches have their own (`search_timeout_ms`).
    statement_timeout_ms: int | None = Field(default=None, gt=0)
    # Server-side cap per retrieval statement (each search transaction sets it with SET LOCAL, so
    # the ingest paths sharing the pool are unaffected). DATABASE_SEARCH_TIMEOUT_MS.
    search_timeout_ms: int = Field(default=10_000, gt=0)
    # Open every pooled connection with default_transaction_read_only=on (read-only consumers such
    # as the eval CLI). Writes then fail in the database itself.
    read_only: bool = False
    # Ingest locks: each running ingest holds ONE dedicated (non-pooled) connection, so this also
    # caps concurrent ingests per process. Size it below the server's max_connections budget.
    # Needs a direct or session-pooled connection: PgBouncer transaction pooling breaks the
    # session-level advisory locks.
    max_ingest_locks: int = Field(default=2, gt=0)
    # Overall wait for an ingest slot + the per-source lock before ConcurrentIngestError.
    ingest_lock_timeout_seconds: float = Field(default=300.0, gt=0)
    # pgvector HNSW search knobs, applied per search with SET LOCAL. ef_search: candidate list
    # size (recall vs latency; pgvector allows 1..1000). max_scan_tuples: how far an iterative
    # (filtered) scan may walk the graph before giving up with fewer than k rows.
    hnsw_ef_search: int = Field(default=40, ge=1, le=1000)
    hnsw_max_scan_tuples: int = Field(default=20_000, gt=0)

    @field_validator("statement_timeout_ms", mode="before")
    @classmethod
    def _empty_means_no_timeout(cls, value: object) -> object:
        return None if value == "" else value


class CheckpointerSettings(BaseSettings):
    """LangGraph Postgres checkpointer (conversation state of threads). Env: CHECKPOINTER_*.

    Own psycopg pool (the saver needs autocommit + dict_row), separate from the SQLAlchemy one:
    size it within the server's connection budget. The pool opens at `Container.start()`; the
    tables are created by `python -m agintis.cli.checkpointer_setup`, never at boot, and the boot
    check refuses to start while they are missing. `enabled=false` turns threads off (thread runs
    answer 503, no pool, no check): the ingest/eval CLIs run that way.

    Sizing: AsyncPostgresSaver serializes ALL its I/O behind one asyncio.Lock, so a bigger pool
    adds NO concurrency between runs (extra connections only absorb reconnects): keep the
    defaults small.
    """

    model_config = SettingsConfigDict(env_prefix="CHECKPOINTER_", env_file=".env", extra="ignore")

    enabled: bool = True
    pool_min_size: int = Field(default=1, ge=1)
    pool_max_size: int = Field(default=4, ge=1)
    # Seconds a checkpoint operation waits for a free pooled connection before failing.
    pool_timeout_seconds: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def _min_not_above_max(self) -> Self:
        if self.pool_min_size > self.pool_max_size:
            raise ValueError("pool_min_size must not exceed pool_max_size")
        return self


DEFAULT_LEASE_TTL_SECONDS = 900.0
DEFAULT_RUN_DEADLINE_SECONDS = 600.0
# A run must end at least this long before its lease expires (acquire/close/release overhead).
LEASE_MARGIN_SECONDS = 30.0


class ThreadsSettings(BaseSettings):
    """Conversation threads. Env: THREADS_LEASE_TTL_SECONDS, THREADS_RUN_DEADLINE_SECONDS."""

    model_config = SettingsConfigDict(env_prefix="THREADS_", env_file=".env", extra="ignore")

    # How long a run owns its thread (no heartbeat: the lease is never extended).
    lease_ttl_seconds: float = Field(default=DEFAULT_LEASE_TTL_SECONDS, gt=0)
    # A thread run is stopped (error `run_timeout`, retryable) after this long, so it always ends
    # before its lease: must be below `lease_ttl_seconds` minus LEASE_MARGIN_SECONDS.
    run_deadline_seconds: float = Field(default=DEFAULT_RUN_DEADLINE_SECONDS, gt=0)

    @model_validator(mode="after")
    def _deadline_below_lease(self) -> Self:
        limit = self.lease_ttl_seconds - LEASE_MARGIN_SECONDS
        if self.run_deadline_seconds >= limit:
            raise ValueError(
                f"run_deadline_seconds must be below lease_ttl_seconds minus {LEASE_MARGIN_SECONDS}"
                f" (< {limit})"
            )
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["local", "test", "prod"] = "local"
    database_url: SecretStr
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # Boot-time checks: tokenizer warm-up (fail fast when it cannot be loaded) and a best-effort
    # comparison of the served TEI model with `embeddings.model`.
    startup_checks: bool = True
    # Nested groups keep FLAT env names (EMBEDDINGS_URL, not EMBEDDINGS__URL): each group is
    # its own BaseSettings with an env_prefix and its own `.env` loading. Because of that
    # `Settings(_env_file=None)` is NOT isolated: use `Settings.isolated(...)` in tests.
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    agents: AgentsSettings = Field(default_factory=AgentsSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    http: HttpSettings = Field(default_factory=HttpSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    checkpointer: CheckpointerSettings = Field(default_factory=CheckpointerSettings)
    threads: ThreadsSettings = Field(default_factory=ThreadsSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)

    @model_validator(mode="after")
    def _probe_concurrency_fits_the_pool(self) -> Self:
        limit = max(1, self.database.pool_size - 1)
        if self.knowledge.probe_concurrency > limit:
            raise ValueError(
                f"KNOWLEDGE_PROBE_CONCURRENCY={self.knowledge.probe_concurrency} must be at most "
                f"DATABASE_POOL_SIZE - 1 = {limit} (probes share the pool with thread leases)"
            )
        return self

    # False for `Settings.isolated()`: nothing derived from these settings may read `.env`.
    _reads_dotenv: bool = PrivateAttr(default=True)

    @property
    def reads_dotenv(self) -> bool:
        return self._reads_dotenv

    @field_validator("database_url")
    @classmethod
    def _require_psycopg_scheme(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith(SQLALCHEMY_PSYCOPG_SCHEME):
            raise ValueError(f"database_url must start with {SQLALCHEMY_PSYCOPG_SCHEME}")
        return value

    @property
    def default_tenant_enabled(self) -> bool:
        """The default tenant is fail-open, so it needs an explicit APP_ENV=local.

        `app_env` defaults to "local": a deploy that forgot APP_ENV must not inherit it.
        """
        return self.explicit_local_env

    @property
    def explicit_local_env(self) -> bool:
        """APP_ENV=local set EXPLICITLY (the default is also "local" and must not count)."""
        return self.app_env == "local" and "app_env" in self.model_fields_set

    @property
    def libpq_url(self) -> SecretStr:
        """Plain libpq DSN for raw psycopg / AsyncPostgresSaver (no SQLAlchemy driver suffix)."""
        raw = self.database_url.get_secret_value()
        return SecretStr(LIBPQ_SCHEME + raw.removeprefix(SQLALCHEMY_PSYCOPG_SCHEME))

    @classmethod
    def isolated(cls, **overrides: Any) -> Self:
        """Settings that never read a `.env` file, nested groups included.

        Group overrides may be a mapping (`embeddings={"batch_size": 4}`) or a ready instance.
        Process environment variables still apply (tests monkeypatch them explicitly).
        """
        for name, model in _GROUPS.items():
            value = overrides.get(name)
            if value is None:
                overrides[name] = model(_env_file=None)
            elif isinstance(value, Mapping):
                overrides[name] = model(_env_file=None, **value)
        settings = cls(_env_file=None, **overrides)
        settings._reads_dotenv = False
        return settings


_GROUPS: dict[str, type[BaseSettings]] = {
    "embeddings": EmbeddingsSettings,
    "reranker": RerankerSettings,
    "parser": ParserSettings,
    "knowledge": KnowledgeSettings,
    "agents": AgentsSettings,
    "database": DatabaseSettings,
    "llm": LlmSettings,
    "http": HttpSettings,
    "auth": AuthSettings,
    "checkpointer": CheckpointerSettings,
    "threads": ThreadsSettings,
    "observability": ObservabilitySettings,
    "eval": EvalSettings,
}


# Environment variable prefixes a profile's `api_key_env` must not use: every nested group's
# `env_prefix`, the first word of every FLAT `Settings` field (APP_, DATABASE_, LOG_, STARTUP_...,
# derived, so a new field is covered) and the compose/test secrets that live next to them in `.env`.
# The bare word (`LLM`, `DATABASE`) is refused too (see `platform.llm.profiles`).
EXTRA_SECRET_PREFIXES: tuple[str, ...] = ("POSTGRES_", "TEST_DATABASE_")
ENV_PREFIXES: tuple[str, ...] = tuple(
    sorted(
        {
            *(str(model.model_config["env_prefix"]) for model in _GROUPS.values()),
            *(
                name.split("_")[0].upper() + "_"
                for name in Settings.model_fields
                if name not in _GROUPS
            ),
            *EXTRA_SECRET_PREFIXES,
        }
    )
)


@lru_cache
def get_settings() -> Settings:
    """Production path only; tests build Settings explicitly."""
    return Settings()
