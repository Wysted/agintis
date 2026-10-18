# Caso: documentación de FastAPI en español

[English](README.md) | **Español**

> La versión en inglés es la canónica: si difieren, vale la inglesa.

Segundo caso de ejemplo de la plantilla. Existe para mostrar que la plantilla es genérica: un corpus en Markdown
(no PDF/HTML), un índice sin títulos (salen del H1 de cada página, con los atributos de encabezado
`{ #id }` descartados por el parser de Markdown), un dominio distinto y dos perfiles/temas.
Todo es configuración: `profiles/` (conocimiento) + `agents/` (el agente `fastapi-es-rag`).

Estructura: `fetch/` (adquisición en un commit fijo), `samples/` (ver su README), `profiles/`,
`agents/`, `calibration/` (conjuntos de desarrollo `in_dev.jsonl` / `ood.jsonl` para los umbrales de evidencia, ver
su README). `index/` se genera y está en gitignore. La política de clarificación es `none` hasta que se calibren sus
umbrales. `profiles/eval_questions.jsonl` está RESERVADO (held-out): nunca ajustes prompts,
recuperación ni umbrales con él (ni calibres con él).

## Ejecución

Necesita los servicios `db` y `embeddings` de compose (ver el README principal).

```bash
uv run --no-project --python 3.12 python -I case/fastapi_es/fetch/fetch.py
uv run python -m agintis.cli.ingest --profiles case/fastapi_es/profiles \
    --index case/fastapi_es/index/documents.jsonl
```

Tras una nueva descarga que cambió o eliminó URI de origen, agrega `--prune` a la ingesta (índice completo, sin
`--limit`): elimina los documentos READY de estos perfiles cuya fuente ya no está en el
lote (solo cuando nada falló, se ingestó algo y cada fila del índice tiene su archivo en disco;
cada eliminación queda registrada). "En el lote" significa en ESTE archivo de índice: mantén un índice por conjunto de perfiles
y nunca ejecutes dos ingestas concurrentes sobre los mismos perfiles, o se podarán entre sí.

El id del agente es `fastapi-es-rag`. Calibra los umbrales de evidencia con los conjuntos de desarrollo (solo lectura;
`--analyze` puntúa la consulta independiente del analizador igual que en tiempo de ejecución y necesita las claves `LLM_*`):

```bash
uv run python -m agintis.cli.calibrate_evidence --profiles case/fastapi_es/profiles \
    --agents case/fastapi_es/agents --agent fastapi-es-rag --analyze \
    --report case/fastapi_es/calibration/report.json \
    case/fastapi_es/calibration/in_dev.jsonl case/fastapi_es/calibration/ood.jsonl
```

Sirve el agente con `KNOWLEDGE_PROFILES_DIR=case/fastapi_es/profiles`,
`AGENTS_DIR=case/fastapi_es/agents` y los ajustes `LLM_*`. Para `cli.dump_answers` / `cli.judge`
se pueden reutilizar los perfiles LLM con nombre de `case/spensiones/llm_profiles.yaml` mediante
`LLM_PROFILES_FILE` (este caso no incluye una copia).

## Resultados (parte 4.2)

Corpus: documentación de FastAPI en español (MIT) en el tag 0.143.3 / commit bb7f314: 64 documentos (13 `guia` +
51 `tutorial`), 735 chunks. El fetcher inserta en línea los includes de `docs_src` (se respetan los rangos `ln[]`;
falla ante un archivo faltante o un include sin correspondencia).

Umbrales de evidencia (`calibrate_evidence --analyze`, solo conjuntos de desarrollo: `in_dev` 30 / `ood` 40, 12
trampas de dominio cercano verificadas contra el corpus con `rg`; escritos aparte del
`eval_questions.jsonl` reservado, 30 preguntas sobre 30 documentos): `low_confidence_below` 0.6572,
`out_of_scope_below` 0.4922. AUROC 0.987, recall de `in` por encima del umbral `low` 0.90. Fuera de dominio
por encima de `low`: las preguntas sobre Dockerfile y rate limiting (el corpus no tiene página para ellas). `in`
por debajo de `low`: in-fa-08, in-fa-21, in-fa-24.

Recuperación en el conjunto reservado (`cli.eval`, sin LLM; denso == híbrido, ver hallazgo a; medido antes del cambio a `bm25_hybrid`). Solo se reporta, nunca se usa
para ajustar nada:

| hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 |
|-------|-------|-------|--------|--------|
| .467  | .700  | .767  | .800   | .586   |

Por perfil: `guia` hit@5 1.0; `tutorial` hit@10 .727.

Jueces en vivo (v9 para la tabla, `deepseek-chat`; generador `gpt-5.4-mini`; 2 volcados por conjunto):

| set    | metric              | dump 1 | dump 2 |
|--------|---------------------|--------|--------|
| in_dev | faithfulness        | 28/30  | 29/30  |
| in_dev | citation_support    | 22/30  | 22/30  |
| in_dev | answer_relevance    | 30/30  | 30/30  |
| in_dev | refusal_correctness | 30/30  | 30/30  |
| in_dev | script_mix          | 30/30  | 30/30  |
| ood    | refusal_correctness | 40/40  | 40/40  |
| ood    | script_mix          | 40/40  | 40/40  |

El 22/30 de `citation_support` es un artefacto de v9 (bloques de código divididos por línea). Con el juez v10 (parte 4.2c)
pasa a 27/30 (volcado a) y 26/30 (volcado b); las demás filas no cambian.

Meta-evaluación del juez (parte 4.2c): nuevas etiquetas silver ciegas de `citation_support` para `in_dev` (opus + sonnet,
59/60 de acuerdo entre etiquetadores; `calibration/answer_judge/silver_labels_in_dev_{a,b}.jsonl`). Acuerdo con el
consenso unánime, idéntico en 2 ejecuciones:

| judge | dump a (n=29) | dump b (n=30) |
|---|---|---|
| v9  | 75.9% | 70% |
| v10 | 93.1% | 90% |
| v11 | 93.1% | 96.7% / 93.3% (2 runs) |

Volcado a: 93.1% en ambas ejecuciones de v11. Los fallos restantes (in-fa-06, in-fa-15) son `evidence_not_verbatim`
causados por HTML en línea en las fuentes, no por tablas. Se probó una variante de fuente sin etiquetas y se descartó
(quitar las etiquetas en línea une los dígitos: 10<sup>6</sup> -> 106 deja pasar como exacto un número cambiado); costo
aceptado. Para reproducir:
`case/fastapi_es/judge/work/run_42c.sh` (en gitignore, solo del juez), luego `cli.agreement --labels <silver>
--judgments <report>`.

Reproducir: `case/fastapi_es/judge/work/run_42.sh calibrate|answers`. El script y los volcados viven
bajo `judge/`, que está en gitignore (no está en el repo); pide las claves con `read -rs`, así que ejecútalo
en tu propia terminal (sin TTY, no hay prompt).

### Hallazgos diferidos (no corregidos en 4.2; b corregido en 4.2c)

- a. Híbrido == denso aquí y en spensiones: `search_lexical` (`postgres/chunks.py`) usa
  `plainto_tsquery`, un AND sobre toda la pregunta, que da 0 resultados léxicos en 30/30 de estas
  preguntas de evaluación (y 33/40 de `in_dev` de spensiones), así que RRF conserva el orden denso. Los ejemplos
  publicados (Supabase, LangChain) también usan AND. La parte 4.2b probó un OR sobre los lexemas de la pregunta con frecuencia
  de documento <= 10/20/30% del alcance (normalización 1 de `ts_rank_cd`, top-N 20/50) en los conjuntos de
  desarrollo de recuperación (`calibration/retrieval_dev.jsonl` aquí y en spensiones) y lo RECHAZÓ según la
  regla preregistrada: toda configuración perdió MRR@10 frente al denso y al híbrido anterior en ambos conjuntos
  (fastapi .689-.722 vs .792; spensiones .650-.706 vs .794) y 4-6 ítems de hit@1. Sin IDF, la
  rama léxica agrega ruido a un ranking sólido de bge-m3. Rama sin fusionar `part-4.2b` (109025e). Lo resuelve
  `bm25_hybrid` (IDF de BM25 real, índice por tenant, RRF ponderado), que los perfiles usan ahora: ver el README raíz.
- b. CORREGIDO en 4.2c (juez v10) y 4.2d (v11): `split_sentences` dividía los bloques de código delimitados por línea, de modo que
  los marcadores quedaban pegados a fragmentos ("``` [1]"). v10 mantiene el bloque completo; v11 ignora los marcadores en código (delimitado y
  en línea), ver Resultados. Aceptado: `evidence_not_verbatim` en fuentes con HTML en línea (in-fa-06, in-fa-15;
  rediagnosticado, no son tablas). Sigue abierto: bloques delimitados con sangría de 4+ (comentario shortcut en `citations.py`).
- c. Los ejemplos de "dónde está" de `REFUSAL_DEFINITION` tienen sabor spensiones (entrada de índice /
  número de documento); ampliarlos a "página, sección" en la próxima meta-evaluación del juez, no antes.
- d. Fuera de alcance: un agente no RAG, OCR (in-res-30), el ajuste del encabezado de in-cir-26 (ajustaría con
  un ítem de calibración).
