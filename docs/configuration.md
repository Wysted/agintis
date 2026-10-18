# Configuration

**English** | [Español](configuration.es.md)

Everything is configured with environment variables (a `.env` file is read too; real environment
variables win). The source of truth is `src/agintis/bootstrap/settings.py`: nested groups keep
flat names (`EMBEDDINGS_URL`, never `EMBEDDINGS__URL`). An empty value means "unset" for optional
variables. `tests/unit/test_configuration_docs.py` fails when a variable accepted by `Settings` has no row here or a row names a variable that no longer exists.

Columns: **Req.** = the app does not boot without it; **Secret** = never log it or commit it.
Under `docker compose`, the `app` and `migrate` services override `DATABASE_URL`, `EMBEDDINGS_URL`,
`EMBEDDINGS_TOKENIZER`, `RERANKER_URL`, `OBSERVABILITY_OTLP_ENDPOINT`, `LLM_PROFILES_FILE` and
`AUTH_JWKS_FILE`; `app` also `KNOWLEDGE_PROFILES_DIR` and `AGENTS_DIR`. The values in `.env` for
these are ignored there (they are host paths/URLs); use the `APP_*` variants described in
[Compose-only variables](#compose-only-variables).

## Core

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `APP_ENV` | `local` | no | no | `local`, `test` or `prod`. The default tenant (`HTTP_DEFAULT_TENANT`) is honoured only with an EXPLICIT `APP_ENV=local`. |
| `DATABASE_URL` | none | yes | yes | SQLAlchemy URL, must start with `postgresql+psycopg://`. |
| `LOG_LEVEL` | `INFO` | no | no | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |
| `STARTUP_CHECKS` | `true` | no | no | `false` skips EVERY boot check (pgvector, Alembic head, text search config, BM25 index when a profile uses `bm25_hybrid`, checkpoint tables, LLM profiles, tokenizer warm-up, served TEI model). Tests and tooling only; never in production. |

## Database (`DATABASE_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `DATABASE_POOL_SIZE` | `3` | no | no | SQLAlchemy pool size. |
| `DATABASE_MAX_OVERFLOW` | `2` | no | no | Extra connections allowed above the pool size. |
| `DATABASE_POOL_TIMEOUT_SECONDS` | `10.0` | no | no | Wait for a free pooled connection before failing. |
| `DATABASE_POOL_RECYCLE_SECONDS` | `1800` | no | no | Recycle connections older than this. |
| `DATABASE_ECHO` | `false` | no | no | Log every SQL statement. |
| `DATABASE_CONNECT_TIMEOUT_SECONDS` | `5.0` | no | no | libpq connect timeout (rounded up to whole seconds). |
| `DATABASE_STARTUP_CHECK_TIMEOUT_SECONDS` | `15.0` | no | no | Upper bound of the whole boot-time database check. |
| `DATABASE_STATEMENT_TIMEOUT_MS` | unset | no | no | Server-side cap per statement of the pooled engine (ingest-lock connections never get one). |
| `DATABASE_SEARCH_TIMEOUT_MS` | `10000` | no | no | Server-side cap per retrieval statement. |
| `DATABASE_READ_ONLY` | `false` | no | no | Open pooled connections read-only (eval CLIs). |
| `DATABASE_MAX_INGEST_LOCKS` | `2` | no | no | Dedicated connections for ingest locks; caps concurrent ingests per process. Needs a direct or session-pooled connection (not PgBouncer transaction pooling). |
| `DATABASE_INGEST_LOCK_TIMEOUT_SECONDS` | `300.0` | no | no | Wait for an ingest slot plus the per-source lock before `ConcurrentIngestError`. |
| `DATABASE_HNSW_EF_SEARCH` | `40` | no | no | pgvector HNSW candidate list size per search (1..1000). |
| `DATABASE_HNSW_MAX_SCAN_TUPLES` | `20000` | no | no | How far an iterative (filtered) HNSW scan may walk. |

## Embeddings (`EMBEDDINGS_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `EMBEDDINGS_URL` | `http://localhost:8081` | no | no | TEI embeddings server. |
| `EMBEDDINGS_MODEL` | `BAAI/bge-m3` | no | no | Model the server is expected to serve; default tokenizer source. Compose also uses it to start TEI. |
| `EMBEDDINGS_TOKENIZER` | unset (uses the model) | no | no | HF hub id or local `tokenizer.json` used to count tokens when chunking. |
| `EMBEDDINGS_BATCH_SIZE` | `16` | no | no | Texts per request; must not exceed TEI `--max-client-batch-size`. |
| `EMBEDDINGS_TIMEOUT_SECONDS` | `120.0` | no | no | Per-attempt read timeout. |
| `EMBEDDINGS_DEADLINE_SECONDS` | `300.0` | no | no | Total budget of one embed call, retries included. |
| `EMBEDDINGS_QUERY_PREFIX` | empty | no | no | Instruction prefix some models need for queries (e5: `query: `). |
| `EMBEDDINGS_DOCUMENT_PREFIX` | empty | no | no | Same, for documents (e5: `passage: `). |
| `EMBEDDINGS_EXPECTED_DIM` | `1024` | no | no | Required vector size; also the `chunks.embedding` column width. Another dimension needs a new migration and a re-embed. |
| `EMBEDDINGS_API_KEY` | unset | no | yes | Bearer key for the embeddings server. |

## Reranker (`RERANKER_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `RERANKER_URL` | unset | no | no | TEI reranker. Unset = no `tei` reranker (profiles use `none`); if it is down, retrieval degrades to the vector order. |
| `RERANKER_MAX_CANDIDATES` | `8` | no | no | Max vector hits scored by the cross-encoder per query. |
| `RERANKER_TIMEOUT_SECONDS` | `60.0` | no | no | Per-attempt read timeout. |
| `RERANKER_DEADLINE_SECONDS` | `90.0` | no | no | Total budget of one rerank call. |
| `RERANKER_API_KEY` | unset | no | yes | Bearer key for the reranker server. |

## Parser (`PARSER_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `PARSER_MAX_BYTES` | `52428800` | no | no | Largest accepted source file. |
| `PARSER_MAX_PAGES` | `2000` | no | no | Largest accepted PDF, in pages. |
| `PARSER_MAX_UNCOMPRESSED_BYTES` | `209715200` | no | no | DOCX zip pre-check: declared uncompressed size. |
| `PARSER_MAX_COMPRESSION_RATIO` | `100.0` | no | no | DOCX zip pre-check: compression ratio (zip bombs). Must be above 1. |

## Knowledge (`KNOWLEDGE_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `KNOWLEDGE_PROFILES_DIR` | unset | no | no | Directory with one YAML RAG profile per file, loaded at boot (unset = only the built-in default profile). |
| `KNOWLEDGE_TEXT_SEARCH_CONFIG` | `public.es_unaccent` | no | no | Text search configuration for the lexical branch of hybrid search. Change it only if the migration used another. |
| `KNOWLEDGE_HYBRID_BRANCH_DEPTH` | `50` | no | no | Candidates taken from EACH branch of `hybrid` and `bm25_hybrid` search (1..1000). |
| `KNOWLEDGE_HYBRID_RRF_K` | `60` | no | no | Reciprocal rank fusion constant. |
| `KNOWLEDGE_MAX_QUERY_CHARS` | `2000` | no | no | Longest accepted retrieval query; longer ones are rejected. |
| `KNOWLEDGE_PROBE_CONCURRENCY` | `2` | no | no | Evidence-router probe searches in flight; at most `DATABASE_POOL_SIZE - 1` (min 1). |

## Agents (`AGENTS_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `AGENTS_DIR` | unset | no | no | Directory with one YAML agent spec per file; each registers one agent next to the built-in `direct-chat`. A spec naming an unknown profile aborts the boot. |

## LLM (`LLM_*`)

`LLM_PROVIDER` unset disables the LLM (agents that need it answer an error). These are the platform defaults (profile `default`); agent specs may bind roles to named profiles from `LLM_PROFILES_FILE`.

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `LLM_PROVIDER` | unset | no | no | `openai`, `anthropic`, `ollama` or any `init_chat_model` provider whose uv extra is installed. OpenAI-compatible servers: `openai` plus `LLM_BASE_URL`. |
| `LLM_MODEL` | `gpt-5.4-mini` | no | no | Default model id for every role. |
| `LLM_API_KEY` | unset | no | yes | Provider key (use a dummy value for OpenAI-compatible local servers). |
| `LLM_BASE_URL` | unset | no | no | Override the provider endpoint. |
| `LLM_TEMPERATURE` | unset | no | no | 0..2; unset = provider default (reasoning models reject a custom value). |
| `LLM_TIMEOUT_SECONDS` | `60.0` | no | no | Per-request timeout. |
| `LLM_MAX_RETRIES` | unset | no | no | SDK retries; unset = the SDK default. |
| `LLM_STREAM_USAGE` | unset | no | no | Ask for token usage while streaming (openai only). Set `true` for OpenAI-compatible servers. |
| `LLM_ROUTER_MODEL` | unset | no | no | Model for the LLM router role (unset = `LLM_MODEL`). |
| `LLM_ANALYZER_MODEL` | unset | no | no | Model for the query analyzer role; needs structured output (unset = `LLM_MODEL`). |
| `LLM_SYNTHESIS_MODEL` | unset | no | no | Model for answer synthesis (unset = `LLM_MODEL`). |
| `LLM_PROFILES_FILE` | unset | no | no | YAML with NAMED LLM profiles an agent spec can bind per role; also used for the judge. Each profile names its key variable (`api_key_env`). |

## HTTP edge (`HTTP_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `HTTP_DEFAULT_TENANT` | unset | no | no | Tenant for requests without a tenant header. Honoured ONLY with an explicit `APP_ENV=local`. |
| `HTTP_MAX_MESSAGE_CHARS` | `4000` | no | no | Longest accepted chat message; longer ones answer 422. Body size limits belong to the gateway. |

## Auth (`AUTH_*`)

`jwt` mode requires `AUTH_ISSUER`, `AUTH_AUDIENCE` and exactly one of `AUTH_JWKS_URL` / `AUTH_JWKS_FILE`.

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `AUTH_TENANT_RESOLVER` | `header` | no | no | `header` trusts `X-Tenant-Id` (only behind a trusted gateway); `jwt` verifies `Authorization: Bearer`. |
| `AUTH_ISSUER` | unset | jwt | no | Expected `iss`. |
| `AUTH_AUDIENCE` | unset | jwt | no | Expected `aud`. Must NOT be the OAuth client id. |
| `AUTH_JWKS_URL` | unset | jwt (or file) | no | JWKS endpoint (production; cached). Non-https only with `AUTH_JWKS_ALLOW_INSECURE` or explicit `APP_ENV=local`. |
| `AUTH_JWKS_FILE` | unset | jwt (or url) | no | Local JWKS file (dev, air-gapped). `dev-*` kids are refused outside `APP_ENV=local`. |
| `AUTH_TENANT_CLAIM` | `tenant_id` | no | no | Claim holding the tenant; a dotted path reaches nested claims. |
| `AUTH_ALGORITHMS` | `RS256,ES256` | no | no | Comma-separated asymmetric algorithms; `none` and HS* are refused. |
| `AUTH_LEEWAY_SECONDS` | `10` | no | no | Clock skew tolerance (0..300). |
| `AUTH_JWKS_CACHE_TTL_SECONDS` | `600.0` | no | no | How long fetched keys are cached. |
| `AUTH_JWKS_MIN_REFRESH_SECONDS` | `30.0` | no | no | Minimum time between JWKS fetches; at least 5 and not above the TTL. |
| `AUTH_JWKS_TIMEOUT_SECONDS` | `5.0` | no | no | Deadline of one JWKS fetch. |
| `AUTH_JWKS_STALE_GRACE_SECONDS` | `3600.0` | no | no | Keep serving cached keys this long past the TTL while refresh fails (0 = off). |
| `AUTH_JWKS_ALLOW_INSECURE` | `false` | no | no | Accept a non-https `AUTH_JWKS_URL`. |
| `AUTH_MAX_TOKEN_LIFETIME_SECONDS` | `86400` | no | no | Longest accepted token lifetime (60..2592000). |

## Checkpointer (`CHECKPOINTER_*`)

The LangGraph checkpoint tables are created by `python -m agintis.cli.checkpointer_setup`, never at boot; the boot check refuses to start while they are missing.

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `CHECKPOINTER_ENABLED` | `true` | no | no | `false` turns threads off (thread runs answer 503; no pool). |
| `CHECKPOINTER_POOL_MIN_SIZE` | `1` | no | no | Minimum psycopg pool size. |
| `CHECKPOINTER_POOL_MAX_SIZE` | `4` | no | no | Maximum psycopg pool size; a bigger pool adds no concurrency. |
| `CHECKPOINTER_POOL_TIMEOUT_SECONDS` | `10.0` | no | no | Wait for a free connection before failing. |

## Threads (`THREADS_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `THREADS_LEASE_TTL_SECONDS` | `900.0` | no | no | How long a run owns its thread (never extended). |
| `THREADS_RUN_DEADLINE_SECONDS` | `600.0` | no | no | A run is stopped (`run_timeout`) after this long; must be below the lease TTL minus 30. |

## Observability (`OBSERVABILITY_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `OBSERVABILITY_TRACING` | `none` | no | no | `none` or `otlp` (OpenTelemetry spans; needs the `otel` uv extra, included in the Docker image). |
| `OBSERVABILITY_OTLP_ENDPOINT` | unset | with `otlp` | no | Full OTLP/HTTP traces URL (Phoenix: `http://localhost:6006/v1/traces`). |
| `OBSERVABILITY_OTLP_HEADERS` | unset | no | yes | `name=value,name2=value2` sent with every export (e.g. backend auth). |
| `OBSERVABILITY_OTLP_TIMEOUT_SECONDS` | `5.0` | no | no | Per-request export timeout (max 60). |
| `OBSERVABILITY_SERVICE_NAME` | `agintis` | no | no | `service.name` resource attribute. |
| `OBSERVABILITY_HIDE_CONTENT` | `false` | no | no | Redact prompts, completions, retrieved texts, tool I/O and error text in spans. |

## Evaluation (`EVAL_*`)

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `EVAL_JUDGE_PROFILE` | unset | for `cli.judge` | no | Named LLM profile (from `LLM_PROFILES_FILE`) of the answer judge; `cli.judge --profile` overrides it. No default on purpose: it must not be the generator's model. |
| `EVAL_JUDGE_CONCURRENCY` | `4` | no | no | Judge calls in flight (1..64). |

## Compose-only variables

Read by `docker-compose.yml` (not by `Settings`). Secrets live in `.env`, which is gitignored.

| Variable | Default | Req. | Secret | Meaning |
|---|---|---|---|---|
| `POSTGRES_USER` | none | yes | no | Database role; compose builds the app `DATABASE_URL` from it. |
| `POSTGRES_PASSWORD` | none | yes | yes | Database password. |
| `POSTGRES_DB` | none | yes | no | Database name. |
| `POSTGRES_PORT` | `5432` | no | no | Host port (bound to 127.0.0.1) of `db`. |
| `CASE` | `fastapi_es` | no | no | Case directory under `case/` mounted read-only at `/app/case` in `app` only (`migrate` does not need it). It must exist: a missing directory fails at `up`. |
| `APP_IMAGE` | `agintis-app` | no | no | Image of `migrate` and `app`. Compose builds it locally unless it was pulled first (e.g. `ghcr.io/wysted/agintis:0.2.0`). |
| `APP_EMBEDDINGS_TOKENIZER` | `/app/models/bge-m3/tokenizer.json` | no | no | Becomes `EMBEDDINGS_TOKENIZER` inside the containers. Change it with another `EMBEDDINGS_MODEL`. |
| `APP_OTLP_ENDPOINT` | `http://phoenix:6006/v1/traces` | no | no | Becomes `OBSERVABILITY_OTLP_ENDPOINT` inside the containers. |
| `APP_LLM_PROFILES_FILE` | empty (unset) | no | no | Becomes `LLM_PROFILES_FILE` inside the containers: a CONTAINER path (e.g. `/app/case/llm_profiles.yaml`). |
| `APP_AUTH_JWKS_FILE` | empty (unset) | no | no | Becomes `AUTH_JWKS_FILE` inside the containers: a CONTAINER path. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | no | no | Model served by the optional `reranker` service (`--profile rerank`). |
| `EMBEDDINGS_PORT` | `8081` | no | no | Host port of `embeddings`. |
| `RERANKER_PORT` | `8082` | no | no | Host port of `reranker`. |
| `PHOENIX_PORT` | `6006` | no | no | Host port of the optional `phoenix` tracing UI (`--profile observability`). |
| `TEI_THREADS` | `2` | no | no | CPU threads and tokenization workers of the TEI services. |
| `TEI_MAX_CLIENT_BATCH_SIZE` | `16` | no | no | TEI `--max-client-batch-size`; keep `EMBEDDINGS_BATCH_SIZE` at or below it. |
| `TEI_MAX_BATCH_TOKENS` | `1024` | no | no | TEI `--max-batch-tokens`. |
| `TEI_MAX_CONCURRENT_REQUESTS` | `32` | no | no | TEI `--max-concurrent-requests`. |
| `EMBEDDINGS_MEM_LIMIT` | `2g` | no | no | Memory limit of `embeddings`. |
| `RERANKER_MEM_LIMIT` | `2g` | no | no | Memory limit of `reranker`. |
| `PHOENIX_MEM_LIMIT` | `1g` | no | no | Memory limit of `phoenix`. |

Also read by tests only: `TEST_DATABASE_URL` (disposable `*_test` databases for the integration suite).
