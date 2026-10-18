# Configuración

[English](configuration.md) | **Español**

> La versión en inglés es la canónica: si difieren, vale la inglesa.

Todo se configura con variables de entorno (también se lee un archivo `.env`; las variables de entorno reales tienen prioridad). La fuente de verdad es `src/agintis/bootstrap/settings.py`: los grupos anidados mantienen nombres planos (`EMBEDDINGS_URL`, nunca `EMBEDDINGS__URL`). Un valor vacío significa "sin definir" en las variables opcionales. `tests/unit/test_configuration_docs.py` falla cuando una variable aceptada por `Settings` no tiene fila aquí o cuando una fila nombra una variable que ya no existe.

Columnas: **Req.** = la app no arranca sin ella; **Secret** = nunca la registres en logs ni la subas al repositorio.
Con `docker compose`, los servicios `app` y `migrate` sobrescriben `DATABASE_URL`, `EMBEDDINGS_URL`,
`EMBEDDINGS_TOKENIZER`, `RERANKER_URL`, `OBSERVABILITY_OTLP_ENDPOINT`, `LLM_PROFILES_FILE` y
`AUTH_JWKS_FILE`; `app` además `KNOWLEDGE_PROFILES_DIR` y `AGENTS_DIR`. Los valores en `.env` de
estas variables se ignoran allí (son rutas/URL del host); usa las variantes `APP_*` descritas en
[Variables solo de Compose](#variables-solo-de-compose).

## Core

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `APP_ENV` | `local` | no | no | `local`, `test` o `prod`. El tenant por defecto (`HTTP_DEFAULT_TENANT`) se respeta solo con un `APP_ENV=local` EXPLÍCITO. |
| `DATABASE_URL` | none | yes | yes | URL de SQLAlchemy, debe comenzar con `postgresql+psycopg://`. |
| `LOG_LEVEL` | `INFO` | no | no | `DEBUG`, `INFO`, `WARNING`, `ERROR` o `CRITICAL`. |
| `STARTUP_CHECKS` | `true` | no | no | `false` omite TODAS las verificaciones de arranque (pgvector, head de Alembic, configuración de búsqueda de texto, índice BM25 si un perfil usa `bm25_hybrid`, tablas del checkpointer, perfiles LLM, precalentamiento del tokenizer, modelo TEI servido). Solo para tests y herramientas; nunca en producción. |

## Base de datos (`DATABASE_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `DATABASE_POOL_SIZE` | `3` | no | no | Tamaño del pool de SQLAlchemy. |
| `DATABASE_MAX_OVERFLOW` | `2` | no | no | Conexiones extra permitidas por encima del tamaño del pool. |
| `DATABASE_POOL_TIMEOUT_SECONDS` | `10.0` | no | no | Espera por una conexión libre del pool antes de fallar. |
| `DATABASE_POOL_RECYCLE_SECONDS` | `1800` | no | no | Recicla las conexiones más antiguas que este valor. |
| `DATABASE_ECHO` | `false` | no | no | Registra cada sentencia SQL. |
| `DATABASE_CONNECT_TIMEOUT_SECONDS` | `5.0` | no | no | Timeout de conexión de libpq (se redondea hacia arriba a segundos enteros). |
| `DATABASE_STARTUP_CHECK_TIMEOUT_SECONDS` | `15.0` | no | no | Límite superior de toda la verificación de la base de datos al arrancar. |
| `DATABASE_STATEMENT_TIMEOUT_MS` | unset | no | no | Tope del lado del servidor por sentencia del engine del pool (las conexiones de locks de ingesta nunca lo reciben). |
| `DATABASE_SEARCH_TIMEOUT_MS` | `10000` | no | no | Tope del lado del servidor por sentencia de recuperación. |
| `DATABASE_READ_ONLY` | `false` | no | no | Abre las conexiones del pool en solo lectura (CLIs de evaluación). |
| `DATABASE_MAX_INGEST_LOCKS` | `2` | no | no | Conexiones dedicadas a los locks de ingesta; limita las ingestas concurrentes por proceso. Requiere una conexión directa o con pool por sesión (no pooling por transacción de PgBouncer). |
| `DATABASE_INGEST_LOCK_TIMEOUT_SECONDS` | `300.0` | no | no | Espera por un cupo de ingesta más el lock por fuente antes de `ConcurrentIngestError`. |
| `DATABASE_HNSW_EF_SEARCH` | `40` | no | no | Tamaño de la lista de candidatos HNSW de pgvector por búsqueda (1..1000). |
| `DATABASE_HNSW_MAX_SCAN_TUPLES` | `20000` | no | no | Cuánto puede avanzar un escaneo HNSW iterativo (con filtros). |

## Embeddings (`EMBEDDINGS_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `EMBEDDINGS_URL` | `http://localhost:8081` | no | no | Servidor de embeddings TEI. |
| `EMBEDDINGS_MODEL` | `BAAI/bge-m3` | no | no | Modelo que se espera que sirva el servidor; fuente por defecto del tokenizer. Compose también lo usa para iniciar TEI. |
| `EMBEDDINGS_TOKENIZER` | unset (uses the model) | no | no | Id del hub de HF o `tokenizer.json` local usado para contar tokens al hacer chunking. |
| `EMBEDDINGS_BATCH_SIZE` | `16` | no | no | Textos por petición; no debe superar el `--max-client-batch-size` de TEI. |
| `EMBEDDINGS_TIMEOUT_SECONDS` | `120.0` | no | no | Timeout de lectura por intento. |
| `EMBEDDINGS_DEADLINE_SECONDS` | `300.0` | no | no | Presupuesto total de una llamada de embed, reintentos incluidos. |
| `EMBEDDINGS_QUERY_PREFIX` | empty | no | no | Prefijo de instrucción que algunos modelos necesitan para las consultas (e5: `query: `). |
| `EMBEDDINGS_DOCUMENT_PREFIX` | empty | no | no | Lo mismo, para documentos (e5: `passage: `). |
| `EMBEDDINGS_EXPECTED_DIM` | `1024` | no | no | Tamaño de vector requerido; también el ancho de la columna `chunks.embedding`. Otra dimensión requiere una migración nueva y volver a generar los embeddings. |
| `EMBEDDINGS_API_KEY` | unset | no | yes | Clave Bearer para el servidor de embeddings. |

## Reranker (`RERANKER_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `RERANKER_URL` | unset | no | no | Reranker TEI. Sin definir = no hay reranker `tei` (los perfiles usan `none`); si está caído, la recuperación degrada al orden vectorial. |
| `RERANKER_MAX_CANDIDATES` | `8` | no | no | Máximo de resultados vectoriales que puntúa el cross-encoder por consulta. |
| `RERANKER_TIMEOUT_SECONDS` | `60.0` | no | no | Timeout de lectura por intento. |
| `RERANKER_DEADLINE_SECONDS` | `90.0` | no | no | Presupuesto total de una llamada de rerank. |
| `RERANKER_API_KEY` | unset | no | yes | Clave Bearer para el servidor del reranker. |

## Parser (`PARSER_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `PARSER_MAX_BYTES` | `52428800` | no | no | Archivo fuente más grande aceptado. |
| `PARSER_MAX_PAGES` | `2000` | no | no | PDF más grande aceptado, en páginas. |
| `PARSER_MAX_UNCOMPRESSED_BYTES` | `209715200` | no | no | Pre-verificación del zip DOCX: tamaño descomprimido declarado. |
| `PARSER_MAX_COMPRESSION_RATIO` | `100.0` | no | no | Pre-verificación del zip DOCX: razón de compresión (zip bombs). Debe ser mayor que 1. |

## Knowledge (`KNOWLEDGE_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `KNOWLEDGE_PROFILES_DIR` | unset | no | no | Directorio con un perfil RAG YAML por archivo, cargado al arrancar (sin definir = solo el perfil por defecto integrado). |
| `KNOWLEDGE_TEXT_SEARCH_CONFIG` | `public.es_unaccent` | no | no | Configuración de búsqueda de texto para la rama léxica de la búsqueda híbrida. Cámbiala solo si la migración usó otra. |
| `KNOWLEDGE_HYBRID_BRANCH_DEPTH` | `50` | no | no | Candidatos tomados de CADA rama de la búsqueda `hybrid` y `bm25_hybrid` (1..1000). |
| `KNOWLEDGE_HYBRID_RRF_K` | `60` | no | no | Constante de la fusión por rango recíproco (RRF). |
| `KNOWLEDGE_MAX_QUERY_CHARS` | `2000` | no | no | Consulta de recuperación más larga aceptada; las más largas se rechazan. |
| `KNOWLEDGE_PROBE_CONCURRENCY` | `2` | no | no | Búsquedas de sondeo del router por evidencia en vuelo; como máximo `DATABASE_POOL_SIZE - 1` (mín. 1). |

## Agents (`AGENTS_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `AGENTS_DIR` | unset | no | no | Directorio con una especificación YAML de agente por archivo; cada una registra un agente junto al `direct-chat` integrado. Una especificación que nombre un perfil desconocido aborta el arranque. |

## LLM (`LLM_*`)

`LLM_PROVIDER` sin definir desactiva el LLM (los agentes que lo necesitan responden con un error). Estos son los valores por defecto de la plataforma (perfil `default`); las especificaciones de agentes pueden asociar roles a perfiles con nombre de `LLM_PROFILES_FILE`.

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `LLM_PROVIDER` | unset | no | no | `openai`, `anthropic`, `ollama` o cualquier proveedor de `init_chat_model` cuyo extra de uv esté instalado. Servidores compatibles con OpenAI: `openai` más `LLM_BASE_URL`. |
| `LLM_MODEL` | `gpt-5.4-mini` | no | no | Id del modelo por defecto para todos los roles. |
| `LLM_API_KEY` | unset | no | yes | Clave del proveedor (usa un valor ficticio para servidores locales compatibles con OpenAI). |
| `LLM_BASE_URL` | unset | no | no | Sobrescribe el endpoint del proveedor. |
| `LLM_TEMPERATURE` | unset | no | no | 0..2; sin definir = el valor por defecto del proveedor (los modelos de razonamiento rechazan un valor personalizado). |
| `LLM_TIMEOUT_SECONDS` | `60.0` | no | no | Timeout por petición. |
| `LLM_MAX_RETRIES` | unset | no | no | Reintentos del SDK; sin definir = el valor por defecto del SDK. |
| `LLM_STREAM_USAGE` | unset | no | no | Pide el uso de tokens durante el streaming (solo openai). Define `true` para servidores compatibles con OpenAI. |
| `LLM_ROUTER_MODEL` | unset | no | no | Modelo para el rol de router LLM (sin definir = `LLM_MODEL`). |
| `LLM_ANALYZER_MODEL` | unset | no | no | Modelo para el rol de analizador de consultas; requiere salida estructurada (sin definir = `LLM_MODEL`). |
| `LLM_SYNTHESIS_MODEL` | unset | no | no | Modelo para la síntesis de respuestas (sin definir = `LLM_MODEL`). |
| `LLM_PROFILES_FILE` | unset | no | no | YAML con perfiles LLM CON NOMBRE que una especificación de agente puede asociar por rol; también se usa para el juez. Cada perfil nombra su variable de clave (`api_key_env`). |

## Borde HTTP (`HTTP_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `HTTP_DEFAULT_TENANT` | unset | no | no | Tenant para las peticiones sin header de tenant. Se respeta SOLO con un `APP_ENV=local` explícito. |
| `HTTP_MAX_MESSAGE_CHARS` | `4000` | no | no | Mensaje de chat más largo aceptado; los más largos responden 422. Los límites de tamaño del cuerpo corresponden al gateway. |

## Auth (`AUTH_*`)

El modo `jwt` requiere `AUTH_ISSUER`, `AUTH_AUDIENCE` y exactamente uno entre `AUTH_JWKS_URL` / `AUTH_JWKS_FILE`.

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `AUTH_TENANT_RESOLVER` | `header` | no | no | `header` confía en `X-Tenant-Id` (solo detrás de un gateway de confianza); `jwt` verifica `Authorization: Bearer`. |
| `AUTH_ISSUER` | unset | jwt | no | `iss` esperado. |
| `AUTH_AUDIENCE` | unset | jwt | no | `aud` esperado. NO debe ser el client id de OAuth. |
| `AUTH_JWKS_URL` | unset | jwt (or file) | no | Endpoint JWKS (producción; con caché). Sin https solo con `AUTH_JWKS_ALLOW_INSECURE` o un `APP_ENV=local` explícito. |
| `AUTH_JWKS_FILE` | unset | jwt (or url) | no | Archivo JWKS local (desarrollo, entornos aislados). Los kids `dev-*` se rechazan fuera de `APP_ENV=local`. |
| `AUTH_TENANT_CLAIM` | `tenant_id` | no | no | Claim que contiene el tenant; una ruta con puntos alcanza claims anidados. |
| `AUTH_ALGORITHMS` | `RS256,ES256` | no | no | Algoritmos asimétricos separados por coma; `none` y HS* se rechazan. |
| `AUTH_LEEWAY_SECONDS` | `10` | no | no | Tolerancia a desfase de reloj (0..300). |
| `AUTH_JWKS_CACHE_TTL_SECONDS` | `600.0` | no | no | Cuánto tiempo se cachean las claves obtenidas. |
| `AUTH_JWKS_MIN_REFRESH_SECONDS` | `30.0` | no | no | Tiempo mínimo entre descargas de JWKS; al menos 5 y no mayor que el TTL. |
| `AUTH_JWKS_TIMEOUT_SECONDS` | `5.0` | no | no | Plazo de una descarga de JWKS. |
| `AUTH_JWKS_STALE_GRACE_SECONDS` | `3600.0` | no | no | Sigue sirviendo las claves en caché durante este tiempo pasado el TTL mientras falla la actualización (0 = desactivado). |
| `AUTH_JWKS_ALLOW_INSECURE` | `false` | no | no | Acepta un `AUTH_JWKS_URL` sin https. |
| `AUTH_MAX_TOKEN_LIFETIME_SECONDS` | `86400` | no | no | Vida máxima aceptada de un token (60..2592000). |

## Checkpointer (`CHECKPOINTER_*`)

Las tablas de checkpoints de LangGraph se crean con `python -m agintis.cli.checkpointer_setup`, nunca al arrancar; la verificación de arranque se niega a iniciar mientras falten.

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `CHECKPOINTER_ENABLED` | `true` | no | no | `false` desactiva los hilos (las ejecuciones sobre hilos responden 503; sin pool). |
| `CHECKPOINTER_POOL_MIN_SIZE` | `1` | no | no | Tamaño mínimo del pool de psycopg. |
| `CHECKPOINTER_POOL_MAX_SIZE` | `4` | no | no | Tamaño máximo del pool de psycopg; un pool mayor no agrega concurrencia. |
| `CHECKPOINTER_POOL_TIMEOUT_SECONDS` | `10.0` | no | no | Espera por una conexión libre antes de fallar. |

## Hilos (`THREADS_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `THREADS_LEASE_TTL_SECONDS` | `900.0` | no | no | Cuánto tiempo una ejecución es dueña de su hilo (nunca se extiende). |
| `THREADS_RUN_DEADLINE_SECONDS` | `600.0` | no | no | Una ejecución se detiene (`run_timeout`) pasado este tiempo; debe ser menor que el TTL del lease menos 30. |

## Observabilidad (`OBSERVABILITY_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `OBSERVABILITY_TRACING` | `none` | no | no | `none` u `otlp` (spans de OpenTelemetry; requiere el extra de uv `otel`, incluido en la imagen Docker). |
| `OBSERVABILITY_OTLP_ENDPOINT` | unset | with `otlp` | no | URL completa de trazas OTLP/HTTP (Phoenix: `http://localhost:6006/v1/traces`). |
| `OBSERVABILITY_OTLP_HEADERS` | unset | no | yes | `name=value,name2=value2` enviado en cada exportación (p. ej. autenticación del backend). |
| `OBSERVABILITY_OTLP_TIMEOUT_SECONDS` | `5.0` | no | no | Timeout de exportación por petición (máx. 60). |
| `OBSERVABILITY_SERVICE_NAME` | `agintis` | no | no | Atributo de recurso `service.name`. |
| `OBSERVABILITY_HIDE_CONTENT` | `false` | no | no | Oculta prompts, completions, textos recuperados, E/S de herramientas y texto de errores en los spans. |

## Evaluación (`EVAL_*`)

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `EVAL_JUDGE_PROFILE` | unset | for `cli.judge` | no | Perfil LLM con nombre (de `LLM_PROFILES_FILE`) del juez de respuestas; `cli.judge --profile` lo sobrescribe. Sin valor por defecto a propósito: no debe ser el modelo del generador. |
| `EVAL_JUDGE_CONCURRENCY` | `4` | no | no | Llamadas al juez en vuelo (1..64). |

## Variables solo de Compose

Las lee `docker-compose.yml` (no `Settings`). Los secretos viven en `.env`, que está en gitignore.

| Variable | Default | Req. | Secret | Significado |
|---|---|---|---|---|
| `POSTGRES_USER` | none | yes | no | Rol de la base de datos; compose construye con él el `DATABASE_URL` de la app. |
| `POSTGRES_PASSWORD` | none | yes | yes | Contraseña de la base de datos. |
| `POSTGRES_DB` | none | yes | no | Nombre de la base de datos. |
| `POSTGRES_PORT` | `5432` | no | no | Puerto del host (enlazado a 127.0.0.1) de `db`. |
| `CASE` | `fastapi_es` | no | no | Directorio del caso bajo `case/` montado en solo lectura en `/app/case` solo en `app` (`migrate` no lo necesita). Debe existir: un directorio faltante falla en `up`. |
| `APP_IMAGE` | `agintis-app` | no | no | Imagen de `migrate` y `app`. Compose la construye localmente salvo que se haya descargado antes (p. ej. `ghcr.io/wysted/agintis:0.2.0`). |
| `APP_EMBEDDINGS_TOKENIZER` | `/app/models/bge-m3/tokenizer.json` | no | no | Se convierte en `EMBEDDINGS_TOKENIZER` dentro de los contenedores. Cámbialo al usar otro `EMBEDDINGS_MODEL`. |
| `APP_OTLP_ENDPOINT` | `http://phoenix:6006/v1/traces` | no | no | Se convierte en `OBSERVABILITY_OTLP_ENDPOINT` dentro de los contenedores. |
| `APP_LLM_PROFILES_FILE` | empty (unset) | no | no | Se convierte en `LLM_PROFILES_FILE` dentro de los contenedores: una ruta DEL CONTENEDOR (p. ej. `/app/case/llm_profiles.yaml`). |
| `APP_AUTH_JWKS_FILE` | empty (unset) | no | no | Se convierte en `AUTH_JWKS_FILE` dentro de los contenedores: una ruta DEL CONTENEDOR. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | no | no | Modelo servido por el servicio opcional `reranker` (`--profile rerank`). |
| `EMBEDDINGS_PORT` | `8081` | no | no | Puerto del host de `embeddings`. |
| `RERANKER_PORT` | `8082` | no | no | Puerto del host de `reranker`. |
| `PHOENIX_PORT` | `6006` | no | no | Puerto del host de la UI de trazas opcional `phoenix` (`--profile observability`). |
| `TEI_THREADS` | `2` | no | no | Hilos de CPU y workers de tokenización de los servicios TEI. |
| `TEI_MAX_CLIENT_BATCH_SIZE` | `16` | no | no | `--max-client-batch-size` de TEI; mantén `EMBEDDINGS_BATCH_SIZE` igual o por debajo. |
| `TEI_MAX_BATCH_TOKENS` | `1024` | no | no | `--max-batch-tokens` de TEI. |
| `TEI_MAX_CONCURRENT_REQUESTS` | `32` | no | no | `--max-concurrent-requests` de TEI. |
| `EMBEDDINGS_MEM_LIMIT` | `2g` | no | no | Límite de memoria de `embeddings`. |
| `RERANKER_MEM_LIMIT` | `2g` | no | no | Límite de memoria de `reranker`. |
| `PHOENIX_MEM_LIMIT` | `1g` | no | no | Límite de memoria de `phoenix`. |

Solo las leen los tests: `TEST_DATABASE_URL` (bases de datos desechables `*_test` para la suite de integración).
