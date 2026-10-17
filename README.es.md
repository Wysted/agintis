# agintis

[English](README.md) | **Español**

> La versión en inglés es la canónica: si difieren, vale la inglesa.

**agintis** — una plantilla de LangGraph para agentes RAG que citan sus fuentes, rechazan lo que no pueden responder y miden ambas cosas.

Es un monolito modular en Python (un solo desplegable) con una estructura hexagonal: domain / application / infrastructure / api por módulo,
Repository + Criteria para la persistencia y el wiring solo en `bootstrap/`. Los agentes son código genérico más YAML: un "caso" (corpus, perfiles de recuperación, especificaciones de agentes, conjuntos de calibración y de evaluación) es
configuración, así que un dominio nuevo no necesita Python nuevo. Los nodos reutilizables del grafo viven en `agent_kit`
(análisis de la consulta, router por evidencia o por LLM, recuperación multitema, suficiencia de relevancia con bucle de reescritura, verificación de alcance con umbrales de evidencia, síntesis con citas). La calidad de las respuestas se mide offline
con métricas de recuperación y un juez de respuestas.

| Módulo | Qué contiene |
|---|---|
| `modules/knowledge` | Ingesta (parseo, limpieza, chunking, embeddings), almacenamiento en Postgres + pgvector, búsqueda densa/híbrida, rerank, perfiles de recuperación en YAML |
| `modules/conversations` | Hilos por tenant, leases, estado del checkpointer de LangGraph |
| `modules/agents_catalog` | Registro de agentes y la API de ejecución HTTP/SSE |
| `modules/agents` | Agentes concretos: `rag_chat` (construido desde especificaciones YAML) y `direct_chat` |
| `agent_kit` | Nodos genéricos de LangGraph compartidos por los agentes |
| `platform` | Transversal: db, http, auth (header / JWT), llm, observabilidad |
| `case/` | Ejemplos completos: `fastapi_es` (pequeño), `spensiones` (grande) |

## Inicio rápido (Docker Compose)

Necesitas Docker, [uv](https://docs.astral.sh/uv/) (solo para descargar el corpus) y una clave de LLM.

```bash
cp .env.example .env        # fill POSTGRES_* and LLM_* (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY)
docker compose up -d db embeddings          # first start downloads the embedding model

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

`CASE=<name>` selecciona el directorio del caso (`./case/<name>` se monta en solo lectura en `/app/case` en
`app`, y `KNOWLEDGE_PROFILES_DIR` / `AGENTS_DIR` apuntan hacia él; debe existir). La imagen en sí
es agnóstica al caso y `migrate` no necesita el caso. Dentro de los contenedores las variables
de rutas del host de `.env` (`LLM_PROFILES_FILE`, `AUTH_JWKS_FILE`, `OBSERVABILITY_OTLP_ENDPOINT`) se
sobrescriben: pasa las rutas del contenedor con `APP_LLM_PROFILES_FILE`, `APP_AUTH_JWKS_FILE` y
`APP_OTLP_ENDPOINT` (ver [docs/configuration.es.md](docs/configuration.es.md)).
La ingesta y la app deben usar el mismo tenant (la CLI de ingesta usa `default` por defecto).
Las versiones etiquetadas publican la imagen como `ghcr.io/wysted/agintis:<version>`: define
`APP_IMAGE=ghcr.io/wysted/agintis:0.1.0` y ejecuta `docker compose --profile app pull` antes de `up` para
usarla en lugar de construirla localmente.

Consulta al agente (stream SSE; con `APP_ENV=local` y `HTTP_DEFAULT_TENANT=default` se puede
omitir el header):

```bash
curl -N localhost:8000/v1/runs/stream \
  -H 'Content-Type: application/json' -H 'X-Tenant-Id: default' \
  -d '{"agent_id": "fastapi-es-rag", "message": "¿Cómo declaro un parámetro de query opcional?"}'
```

Eventos: `run_started`, `step`, `token`, `sources`, `scope`, `clarification`, `done`, `error`. `GET /v1/agents`
lista los ids de los agentes. Para conversaciones de varios turnos, crea un hilo y haz stream sobre él:

```bash
curl -s localhost:8000/v1/threads -H 'Content-Type: application/json' -H 'X-Tenant-Id: default' \
  -d '{"agent_id": "fastapi-es-rag"}'             # -> {"id": "<thread_id>", ...}
curl -N localhost:8000/v1/threads/<thread_id>/runs/stream \
  -H 'Content-Type: application/json' -H 'X-Tenant-Id: default' -d '{"message": "..."}'
```

Con `AUTH_TENANT_RESOLVER=jwt` envía `Authorization: Bearer <token>` en lugar de `X-Tenant-Id`. Para
pruebas locales, genera uno con un par de claves desechable:
`uv run python -m agintis.cli.dev_token --tenant default` (mira su docstring para ver los valores de `AUTH_*`
que debes configurar; las claves de desarrollo se rechazan fuera de `APP_ENV=local`).

Todos los ajustes están documentados en [docs/configuration.es.md](docs/configuration.es.md).

## Desarrollo local (sin contenedor de la app)

```bash
docker compose up -d db embeddings
uv sync --all-extras                 # or pick extras: openai, anthropic, ollama, deepseek, otel
uv run alembic upgrade head
uv run python -m agintis.cli.checkpointer_setup
uv run uvicorn agintis.main:create_app --factory --reload
```

Con el `.env` tal como se copió de `.env.example` (DB y TEI en localhost) agrega
`KNOWLEDGE_PROFILES_DIR=case/fastapi_es/profiles` y `AGENTS_DIR=case/fastapi_es/agents`.
Tests: `uv run pytest` ejecuta todo excepto el marcador `integration`; `uv run pytest -m integration`
necesita los servicios reales (mira el comentario en `pyproject.toml`). Verificaciones: `uv run ruff check`,
`uv run ruff format --check .`, `uv run mypy`, `uv run lint-imports`.

### Verificaciones / CI

`scripts/check.sh` ejecuta toda la compuerta (ruff check, ruff format --check, mypy, lint-imports, tests
unitarios) desde cualquier directorio; ejecútalo antes de hacer push. `.github/workflows/ci.yml` lo ejecuta en cada pull
request y en los push a `main`. El job `integration` (Postgres + TEI) se ejecuta solo
en `main` y por disparo manual, y `docker` construye la imagen sin publicarla.
Al hacer push de un tag `vX.Y.Z` que coincide con la versión de `pyproject.toml` se ejecuta
`.github/workflows/release.yml`, que publica la imagen en GHCR como `X.Y.Z`, `X.Y` y `latest`.

## Agregar un caso

Un caso es un directorio `case/<name>/`. `case/fastapi_es` es el ejemplo completo (corpus en Markdown,
dos perfiles); `case/spensiones` es el más grande (PDF/HTML, siete perfiles, filtros de metadatos).

1. `profiles/*.yaml`: un perfil de recuperación por familia de documentos (parser, limpiadores, chunker,
   ajustes de embeddings/búsqueda) más reglas de enrutamiento. No hay registro que editar.
2. Corpus: `index/documents.jsonl` (una fila por documento, títulos desde `document_metadata.title`
   o el H1 del parser) y los archivos bajo `samples/` con `samples/_manifest.json`. Escribe un
   script en `fetch/` que produzca ambos. Tras una nueva descarga que cambió o eliminó fuentes, ingesta
   con `--prune` (índice completo, sin `--limit`).
3. `agents/*.yaml`: la especificación del agente (ver abajo).
4. `calibration/in_dev.jsonl` y `ood.jsonl`: preguntas de desarrollo dentro y fuera del dominio, con
   trampas de dominio cercano verificadas contra el corpus. Calibra los umbrales de evidencia:
   `uv run python -m agintis.cli.calibrate_evidence --profiles case/<name>/profiles --agents case/<name>/agents --agent <id> --analyze --report case/<name>/calibration/report.json case/<name>/calibration/in_dev.jsonl case/<name>/calibration/ood.jsonl`
   y copia los umbrales en el bloque `clarification` del agente.
5. `profiles/eval_questions.jsonl`: preguntas reservadas (held-out); `profile` es la KEY del perfil.
6. `tests/unit/test_case_configs.py` levanta automáticamente cada `case/*`; no requiere edición.

## Agregar un agente

Un agente es un archivo YAML en el directorio al que apunta `AGENTS_DIR` (el registro es ese
directorio), parseado por `modules/agents/rag_chat/spec.py` (las claves desconocidas son errores). Los campos que
importan:

- `id`, `description`: lo que muestra `GET /v1/agents`.
- `topics`: lista de `{name, profile, description, filters}`; cada tema se asocia a un perfil de
  conocimiento y a una lista permitida de filtros de metadatos (`filters: {}` para ninguno).
- `router`: `auto` (un solo tema; si no, el router LLM), `single` (exactamente un tema), `llm` o `evidence` (sin LLM: sondea cada
  perfil con el embedding de la pregunta; ajusta `evidence_margin`, `probe_k`, `max_topics`).
- `clarification`: `policy: none | evidence`. Con `evidence`, `low_confidence_below` (responde y emite
  un evento de alcance `low`) y `out_of_scope_below` (rechazo fijo, sin recuperación ni síntesis) provienen
  de la calibración, además de los mensajes de rechazo y de charla casual.
- `sufficiency`, `min_sources`, `max_rewrites`: cuándo la evidencia recuperada es suficiente; si no,
  se reescribe y se reintenta.
- `prompts`: sobrescrituras del system prompt por nodo (analyze, extract_filters, synthesize, route,
  rewrite); los valores por defecto son genéricos en inglés.
- `models`: asocia roles de LLM a perfiles con nombre de `LLM_PROFILES_FILE`.

El agente integrado `direct-chat` (chat LLM simple, sin recuperación) siempre está registrado.

## Evaluación

Todo es offline; usa para el juez una familia de modelos distinta a la del generador.

- Recuperación (sin LLM): `uv run python -m agintis.cli.eval --profiles ... --questions case/<name>/profiles/eval_questions.jsonl [--report out.json]` (hit@k, MRR; `--route --agents DIR` agrega la precisión del router).
- Respuestas: `cli.dump_answers --profiles ... --agents ... --agent <id> --questions <jsonl> --out answers.jsonl`, luego `cli.judge --answers answers.jsonl [--report r.json]` (juez de respuestas v11: fidelidad, soporte de citas, relevancia, corrección del rechazo, mezcla de escrituras). Define `LLM_PROFILES_FILE` (perfiles con nombre, p. ej. `case/spensiones/llm_profiles.yaml`) y `EVAL_JUDGE_PROFILE` (o `--profile`); el juez se niega a calificar con el modelo que respondió salvo que uses `--allow-self-judge`.
- Calidad del juez: `cli.agreement --labels <silver labels> --judgments <judge report>`.

Reglas: ajusta prompts, recuperación y umbrales solo con los conjuntos de calibración. Las `eval_questions` están
reservadas: reporta sobre ellas, nunca ajustes con ellas.

## Notas de producción

- Auth: define `AUTH_TENANT_RESOLVER=jwt` (+ issuer, audience, URL de JWKS). El modo `header` por defecto
  confía en `X-Tenant-Id` y solo es seguro detrás de un gateway que lo establezca.
- Define `APP_ENV=prod`. El tenant por defecto solo funciona con un `APP_ENV=local` explícito.
- Compose enlaza los puertos a `127.0.0.1`: pon por delante un reverse proxy con TLS. Los límites de
  tamaño del cuerpo de las peticiones también corresponden ahí.
- Escala con réplicas, no con workers de uvicorn. Ejecuta las migraciones como un paso aparte (el servicio
  `migrate` o tu pipeline), nunca al arrancar la app: réplicas concurrentes competirían entre sí.
- No existe middleware CORS hasta que haya un cliente de navegador; agrégalo entonces con una lista explícita de orígenes.
- Trazas: `OBSERVABILITY_TRACING=otlp` + `OBSERVABILITY_OTLP_ENDPOINT` (la imagen incluye el
  extra `otel`); considera `OBSERVABILITY_HIDE_CONTENT=true` para backends en los que no confíes del todo.
- Vuelve a calibrar los umbrales de evidencia tras una reingesta o un cambio de modelo de embeddings: son
  válidos solo para el snapshot del corpus, el tenant y el modelo con los que se midieron.

## Licencia y contenido de terceros

agintis se publica bajo la [Licencia MIT](LICENSE).

Los casos de ejemplo no redistribuyen sus corpus; cada fetcher los descarga:

- `case/fastapi_es`: la documentación en español de [FastAPI](https://github.com/fastapi/fastapi),
  Licencia MIT, Copyright (c) 2018 Sebastián Ramírez (copia en `case/fastapi_es/samples/LICENSE`).
- `case/spensiones`: normativa pública publicada por la Superintendencia de Pensiones de Chile
  (spensiones.cl). Los documentos no están versionados: solo el crawler, las URL, los metadatos de los documentos y
  los archivos de evaluación (preguntas, etiquetas y ejemplos del juez que citan extractos breves).
  Respeta los términos del sitio y su `robots.txt`, y haz el crawling con cortesía si ejecutas el crawler.
- El tokenizer de bge-m3 (MIT) se descarga al construir la imagen.
