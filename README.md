# agintis

**English** | [Español](README.es.md)

**agintis** — a LangGraph template for building agents: a hexagonal base (threads, streaming HTTP, LLM profiles, tracing, offline evaluation) with a configurable RAG agent that answers only from your sources, refuses what it can't answer, and measures both.

It is a Python modular monolith (one
deployable) with a hexagonal layout: domain / application / infrastructure / api per module,
Repository + Criteria for persistence, wiring only in `bootstrap/`. Agents are generic code plus
YAML: a "case" (corpus, retrieval profiles, agent specs, calibration and eval sets) is
configuration, so a new domain needs no new Python. The reusable graph nodes live in `agent_kit`
(query analysis, evidence or LLM router, multi-topic retrieval, relevance sufficiency with rewrite
loop, scope check with evidence thresholds, optionally cited synthesis). Answer quality is measured offline
with retrieval metrics and an answer judge.

| Module | What it owns |
|---|---|
| `modules/knowledge` | Ingestion (parse, clean, chunk, embed), Postgres + pgvector storage, dense/hybrid/BM25 search, rerank, YAML retrieval profiles |
| `modules/conversations` | Tenant-scoped threads, leases, LangGraph checkpointer state |
| `modules/agents_catalog` | Agent registry and the HTTP/SSE run API |
| `modules/agents` | Concrete agents: `rag_chat` (built from YAML specs) and `direct_chat` |
| `agent_kit` | Generic LangGraph nodes shared by agents |
| `platform` | Cross-cutting: db, http, auth (header / JWT), llm, observability |
| `case/` | Worked examples: `fastapi_es` (small), `spensiones` (large) |

## Quickstart (Docker Compose)

Needs Docker, [uv](https://docs.astral.sh/uv/) (only for the corpus fetch) and an LLM key.

```bash
cp .env.example .env        # fill POSTGRES_* and LLM_* (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY)
docker compose up -d db embeddings          # first start builds the DB image, downloads the model

# 1. Fetch the corpus of a case (fastapi_es: FastAPI docs in Spanish, pinned commit)
uv run --no-project --python 3.12 python -I case/fastapi_es/fetch/fetch.py

# 2. Ingest it (the case dir is mounted at /app/case; CASE defaults to fastapi_es)
docker compose --profile app run --rm app python -m agintis.cli.ingest \
    --profiles /app/case/profiles --index /app/case/index/documents.jsonl --samples /app/case/samples

# 3. Start the app. A one-shot `migrate` service first runs `alembic upgrade head` and
#    `python -m agintis.cli.checkpointer_setup`; the app never migrates at boot.
docker compose --profile app up -d
curl -s localhost:8000/health
```

`CASE=<name>` selects the case directory (`./case/<name>` is mounted read-only at `/app/case` in
`app`, and `KNOWLEDGE_PROFILES_DIR` / `AGENTS_DIR` point into it; it must exist). The image itself
is case-agnostic and `migrate` does not need the case. Inside the containers the host-path
variables of `.env` (`LLM_PROFILES_FILE`, `AUTH_JWKS_FILE`, `OBSERVABILITY_OTLP_ENDPOINT`) are
overridden: pass container paths with `APP_LLM_PROFILES_FILE`, `APP_AUTH_JWKS_FILE` and
`APP_OTLP_ENDPOINT` (see [docs/configuration.md](docs/configuration.md)).
Ingest and the app must use the same tenant (the ingest CLI defaults to `default`).
Tagged releases publish the image as `ghcr.io/wysted/agintis:<version>`: set
`APP_IMAGE=ghcr.io/wysted/agintis:0.2.0` and run `docker compose --profile app pull` before `up` to use
it instead of building locally.

Ask the agent (SSE stream; with `APP_ENV=local` and `HTTP_DEFAULT_TENANT=default` the header can be
dropped):

```bash
curl -N localhost:8000/v1/runs/stream \
  -H 'Content-Type: application/json' -H 'X-Tenant-Id: default' \
  -d '{"agent_id": "fastapi-es-rag", "message": "¿Cómo declaro un parámetro de query opcional?"}'
```

Events: `run_started`, `step`, `token`, `sources`, `scope`, `clarification`, `done`, `error`. `GET /v1/agents`
lists the agent ids. For multi-turn conversations create a thread and stream into it:

```bash
curl -s localhost:8000/v1/threads -H 'Content-Type: application/json' -H 'X-Tenant-Id: default' \
  -d '{"agent_id": "fastapi-es-rag"}'             # -> {"id": "<thread_id>", ...}
curl -N localhost:8000/v1/threads/<thread_id>/runs/stream \
  -H 'Content-Type: application/json' -H 'X-Tenant-Id: default' -d '{"message": "..."}'
```

With `AUTH_TENANT_RESOLVER=jwt` send `Authorization: Bearer <token>` instead of `X-Tenant-Id`. For
local testing mint one with a throwaway keypair:
`uv run python -m agintis.cli.dev_token --tenant default` (see its docstring for the `AUTH_*`
values to set; the dev keys are refused outside `APP_ENV=local`).

All settings are documented in [docs/configuration.md](docs/configuration.md).

## Local development (no app container)

```bash
docker compose up -d db embeddings
uv sync --all-extras                 # or pick extras: openai, anthropic, ollama, deepseek, otel, ocr
uv run alembic upgrade head
uv run python -m agintis.cli.checkpointer_setup
uv run uvicorn agintis.main:create_app --factory --reload
```

With `.env` as copied from `.env.example` (DB and TEI on localhost) add
`KNOWLEDGE_PROFILES_DIR=case/fastapi_es/profiles` and `AGENTS_DIR=case/fastapi_es/agents`.
Tests: `uv run pytest` runs everything except the `integration` marker; `uv run pytest -m integration`
needs the real services (see the comment in `pyproject.toml`). Checks: `uv run ruff check`,
`uv run ruff format --check .`, `uv run mypy`, `uv run lint-imports`.

### Checks / CI

`scripts/check.sh` runs the whole gate (ruff check, ruff format --check, mypy, lint-imports, unit
tests) from any directory; run it before pushing. `.github/workflows/ci.yml` runs it on every pull
request and on pushes to `main`. The `integration` job (Postgres + TEI) runs only
on `main` and manual dispatch, and `docker` builds the image without pushing it.
Pushing a tag `vX.Y.Z` that matches the `pyproject.toml` version runs
`.github/workflows/release.yml`, which publishes the image to GHCR as `X.Y.Z`, `X.Y` and `latest`.

## Add a case

A case is a directory `case/<name>/`. `case/fastapi_es` is the worked example (Markdown corpus,
two profiles); `case/spensiones` is the larger one (PDF/HTML, seven profiles, metadata filters).
Its scanned resolutions are ingested as index cards; OCR for scanned PDFs is opt-in (parser
`ocr_index_card`, RapidOCR on CPU, ~4 s per page) and needs the `ocr` extra: `uv sync --extra ocr`
plus the other extras you use (`--all-extras` includes it); Docker:
`docker compose --profile app build --build-arg UV_EXTRAS="openai otel ocr"`.

1. `profiles/*.yaml`: one retrieval profile per document family (parser, cleaners, chunker,
   embedding/search settings) plus routing rules. No registry to edit. `retrieval_mode`: `dense`
   (default), `hybrid` (dense + Postgres full-text, any Postgres) or the opt-in `bm25_hybrid`
   (dense + BM25 from the pg_textsearch extension, weighted RRF). `bm25_hybrid` needs the
   project's Postgres image (`docker/postgres`, used by compose and CI); managed Postgres such as
   RDS or Cloud SQL lacks the extension, so the app refuses to boot with such a profile there.
   A compose DB created before that image must be rebuilt (`docker compose up -d --build db`)
   before `alembic upgrade head`, otherwise revision 0005 skips the BM25 index (logged warning).
   The shipped cases do not use it: measured as MRR@10 vs dense, a fresh spensiones dev set
   gained (.7304 -> .7708) but the older one dropped past the -0.01 limit (.7937 -> .7780;
   fastapi_es .7917 -> .7900). The BM25 index is shared by every tenant, so term statistics
   (IDF) mix tenants; an offline run with per-tenant indexes had stayed within it (.7879, fastapi_es .7861).
2. Corpus: `index/documents.jsonl` (one row per document, titles from `document_metadata.title`
   or the parser's H1) and the files under `samples/` with `samples/_manifest.json`. Write a
   `fetch/` script that produces both. After a re-fetch that changed or dropped sources, ingest
   with `--prune` (full index, no `--limit`).
3. `agents/*.yaml`: the agent spec (see below).
4. `calibration/in_dev.jsonl` and `ood.jsonl`: in-domain and out-of-domain dev questions, with
   near-domain traps checked against the corpus. Calibrate the evidence thresholds:
   `uv run python -m agintis.cli.calibrate_evidence --profiles case/<name>/profiles --agents case/<name>/agents --agent <id> --analyze --report case/<name>/calibration/report.json case/<name>/calibration/in_dev.jsonl case/<name>/calibration/ood.jsonl`
   and copy the thresholds into the agent's `clarification` block.
5. `profiles/eval_questions.jsonl`: held-out questions; `profile` is the profile KEY.
6. `tests/unit/test_case_configs.py` boots every `case/*` automatically; no edit needed.

## Add an agent

Two ways: a YAML spec of the generic RAG agent (no Python), or your own graph in code.

### A RAG agent from YAML

An agent is one YAML file in the directory `AGENTS_DIR` points to (the registry is that
directory), parsed by `modules/agents/rag_chat/spec.py` (unknown keys are errors). The fields that
matter:

- `id`, `description`: what `GET /v1/agents` shows.
- `topics`: list of `{name, profile, description, filters}`; each topic maps to a knowledge
  profile and a metadata-filter allowlist (`filters: {}` for none).
- `router`: `auto` (single topic, else the LLM router), `single` (exactly one topic), `llm` or `evidence` (no LLM: probes every
  profile with the question embedding; tune `evidence_margin`, `probe_k`, `max_topics`).
- `clarification`: `policy: none | evidence`. With `evidence`, `low_confidence_below` (answer plus
  a `low` scope event) and `out_of_scope_below` (fixed refusal, no retrieval or synthesis) come
  from calibration, plus the refusal and small-talk messages.
- `sufficiency`, `min_sources`, `max_rewrites`: when the retrieved evidence is enough, otherwise
  rewrite and retry.
- `prompts`: per-node system prompt overrides (analyze, extract_filters, synthesize, route,
  rewrite); defaults are generic English.
- `citations: inline | none` (default `inline`): `inline` asks for `[n]` markers in the answer;
  `none` keeps every grounding rule (answer only from the sources, say what is missing, "I do
  not know" otherwise) without markers, e.g. for a customer-service agent. It picks the default
  synthesize prompt; a `prompts.synthesize` override replaces it in both modes (boot warning with
  `none`: make sure your prompt does not ask for markers). The `sources` SSE event is sent in
  both modes, so a UI can still list the references.
- `models`: bind LLM roles to named profiles of `LLM_PROFILES_FILE`.

### An agent in code

Any LangGraph graph can be an agent. Copy `modules/agents/direct_chat.py` (the built-in
`direct-chat`, plain LLM chat with no retrieval: a one-node graph wrapped in `LangGraphRunner`)
to a new module under `modules/agents/`, change its id, description and nodes, and register its
runner in `bootstrap/container.py` next to `direct_chat_runner(...)`. The import contracts
(`uv run lint-imports`) keep it off HTTP and bootstrap: it reaches business modules through their
`api.py` and reusable nodes through `agent_kit`.

## Evaluation

All offline; use a separate model family for the judge than for the generator.

- Retrieval (no LLM): `uv run python -m agintis.cli.eval --profiles ... --questions case/<name>/profiles/eval_questions.jsonl [--report out.json]` (hit@k, MRR; `--route --agents DIR` adds router accuracy).
- Answers: `cli.dump_answers --profiles ... --agents ... --agent <id> --questions <jsonl> --out answers.jsonl`, then `cli.judge --answers answers.jsonl [--report r.json]` (answer judge v12, which grades inline-citation agents exactly like v11: faithfulness, citation support, relevance, refusal correctness, script mix; citation support is not applicable to a `citations: none` agent unless it emits [n] markers anyway). Set `LLM_PROFILES_FILE` (named profiles, e.g. `case/spensiones/llm_profiles.yaml`) and `EVAL_JUDGE_PROFILE` (or `--profile`); the judge refuses to grade with the answering model unless `--allow-self-judge`.
- Judge quality: `cli.agreement --labels <silver labels> --judgments <judge report>`.

Rules: tune prompts, retrieval and thresholds on the calibration sets only. `eval_questions` are
held out: report on them, never tune on them.

## Production notes

- Auth: set `AUTH_TENANT_RESOLVER=jwt` (+ issuer, audience, JWKS URL). The default `header` mode
  trusts `X-Tenant-Id` and is only safe behind a gateway that sets it.
- Set `APP_ENV=prod`. The default tenant works only with an explicit `APP_ENV=local`.
- Compose binds ports to `127.0.0.1`: put a reverse proxy with TLS in front. Request body size
  limits belong there too.
- Scale with replicas, not uvicorn workers. Run migrations as a separate step (the `migrate`
  service or your pipeline), never at app boot: concurrent replicas would race.
- No CORS middleware exists until a browser client does; add it with an explicit origin list then.
- Tracing: `OBSERVABILITY_TRACING=otlp` + `OBSERVABILITY_OTLP_ENDPOINT` (the image includes the
  `otel` extra); consider `OBSERVABILITY_HIDE_CONTENT=true` for backends you do not fully trust.
- Re-calibrate the evidence thresholds after a re-ingest or an embedding model change: they are
  valid only for the corpus snapshot, tenant and model they were measured on.

## License and third-party content

agintis is released under the [MIT License](LICENSE).

The example cases do not redistribute their corpora; each fetcher downloads them:

- `case/fastapi_es`: the Spanish documentation of [FastAPI](https://github.com/fastapi/fastapi),
  MIT License, Copyright (c) 2018 Sebastián Ramírez (copy in `case/fastapi_es/samples/LICENSE`).
- `case/spensiones`: public regulation published by the Chilean Superintendencia de Pensiones
  (spensiones.cl). The documents are not versioned: only the crawler, URLs, document metadata and
  evaluation files (questions, labels and judge examples quoting short excerpts).
  Respect the site's terms and `robots.txt`, and crawl politely if you run the crawler.
- The bge-m3 tokenizer (MIT) is downloaded when the image is built.
