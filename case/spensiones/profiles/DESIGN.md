# RagProfiles para spensiones.cl: diseño y validación

Caso: documentos de la Superintendencia de Pensiones (índice del crawler: 8.174 filas, 29 campos). Los identificadores van en inglés y el texto en español.

## 1. Contenido de esta carpeta

| archivo | qué es |
|---|---|
| `compendio.yaml`, `circular.yaml`, `ncg.yaml`, `articulo_faq.yaml`, `guia_jurisprudencia.yaml`, `resolucion.yaml` | un perfil por familia, validados con las muestras |
| `guia_jurisprudencia_nota.yaml` | borrador para las 117 "Nota Interna" (sin muestra, `status: draft_unvalidated`) |
| `validation/validate_profiles.py` | prototipo de referencia de cleaners, chunkers y extractores. Lee los YAML y genera `validation_report.md` y `chunks_preview.jsonl` |
| `validation/check_eval.py` | comprueba que cada pregunta de eval cae en un solo chunk cuyo metadata coincide con su localizador (`eval_locator_check.md`) |
| `validation/index_estimates.py` | estimaciones de volumen y tiempo de embedding a partir del índice (`index_estimates.md`) |
| `eval_questions.jsonl` | 29 preguntas en español con respuesta esperada, snippet, muestra y localizador |

Para correrlo: `uv run --no-project --python 3.12 --with tokenizers --with pyyaml python -I validation/validate_profiles.py --case <case> --out validation`. Los tokens se cuentan con el tokenizer de bge-m3 (`add_special_tokens=False`) y se reserva un margen de 8 tokens, como hace `StructuredChunker`.

> **Nota (2c-ii): forma canónica.** Los YAML se migraron al formato que carga el template (`load_profiles_dir`) y ya no usan la forma del prototipo que se describe más abajo:
>
> | prototipo (este documento, `validation/*.py`) | canónico (template) |
> |---|---|
> | `parser_params.cleaners[].name` | `cleaners[].key` en el nivel superior; `join_pages` / `keep_page_offsets` son siempre así |
> | `routing: {index_filter, title_regex, ...}` | `routing: [{match: {campo: [valores] \| {regex: ...}}, exclude: {...}}]`; `status: draft` = nunca se enruta |
> | `str_list` / `int_list` | `list[str]` / `list[int]` |
> | `transform: {value_map: ..., date_floor: true}` | `transforms: [{key: value_map, params: {...}}, {key: date_floor}]` |
> | extractores del caso (`compendio_path`, `compendio_target`, `ncg_role`, `title_is_question`, `resolution_topic`) | `template`, `layered_regex` (instancia nombrada bajo `extractors:`) y `classify`; `regex_all`, `legal_refs` y `vigencia_date` son genéricos |
> | `page_end` como extractor | `page` / `page_end` son campos built-in |
> | `max_tokens` / `fallback_overlap` en `chunker_params` | el presupuesto es `chunk_max_tokens` del perfil |
> | `heading_prefix_template` con `{system}`, `{libro}`, `{title_short}` | variables = campos de la fila del índice (`{area}`, `{number}`, `{date}`, `{title:70}`) |
> | `on_scanned: index_card` | `parser_key: simple_index_card` (OCR disponible opt-in con `ocr_index_card` + extra `ocr`; medido y no adoptado, ver 4.3) |
> | `spensiones_articulo_faq`: chunker `structured` con unión de secciones | chunker `whole_document` (un chunk si cabe, si no `structured`) |
>
> `validation/validate_profiles.py` y `check_eval.py` leen la forma del prototipo: son la referencia histórica de comportamiento y ya no corren contra estos YAML. La aceptación vive ahora en `tests/evals/test_case_spensiones.py`.

## 2. Perfiles

| perfil (`key`) | routing | parser / chunker | unidad (`split_level`) | `chunk_max_tokens` | metadata clave |
|---|---|---|---|---:|---|
| `spensiones_compendio` | `doc_type=compendio_libro` (10 PDF). Los 113 `compendio_titulo` HTML **no se ingieren**: son hubs de navegación (glosario + links, sin cuerpo normativo) | `simple` / `pattern_hierarchy` | número (`1.`) dentro de Capítulo/Anexo. El breadcrumb Título/Letra se usa como contexto | 700 | `libro`, `titulo`, `letra`, `capitulo`, `anexo`, `numeros[]`, `compendio_path`, `amended_by_ncg[]`, `cited_laws[]` |
| `spensiones_ncg` | `doc_type=ncg` (485) | `simple` / `pattern_hierarchy` con `ignore_inside` (comillas anidadas) | enmienda (`A.`, `B.`…). Si excede el presupuesto se corta en operaciones `a)` o en los numerales del texto insertado | 700 | `ncg_number`, `op_verb` (replace/add/delete/modify…), `target_*`, `compendio_path`, `effective_from`, `doc_role` |
| `spensiones_circular` | `doc_type=circular` (2.520) | `simple` / `pattern_hierarchy` | número (`1.` / `2.-`) y secciones VIGENCIA/ANEXO al mismo nivel; el preámbulo (MATERIA/VISTOS/REF.) va en un chunk propio | 600 | `circular_number`, `series`, `issued_date`, `effective_from`, `listing_tab`, `modifies_circulars[]` |
| `spensiones_articulo_faq` | `doc_type=articulo_html`, excepto los títulos "Resolución Exenta…" | `simple` + cleaner HTML / `structured` (ya existe) | documento completo (pregunta + respuesta). Las páginas guía se cortan por subtítulos en negrita | 512 | `aid`, `audiences[]`, `keywords[]`, `related_aids[]`, `is_question` |
| `spensiones_jurisprudencia_indice` | `guia_faq` con título "Libro de Jurisprudencia" | `simple` / `records` | tema (`7.`) con sus registros `7.12.4. tema  Of. Ord. N° …` | 400 | `materia`, `tema`, `oficio_refs[]` |
| `spensiones_resolucion` | `doc_type=resolucion` + 6 `articulo_html` "Resolución Exenta…" | `simple_index_card` (OCR opt-in `ocr_index_card`, no adoptado) / `pattern_hierarchy` (VISTOS/CONSIDERANDO/RESUELVO, sin validar) | ficha del índice (1 chunk) | 600 | `resolution_number`, `issued_date`, `content_mode`, `topic_class` |

Fuentes de metadata usadas:

- `document:<campo>` lee la fila del índice (número, fecha, subtype, area, section, aid). Es una fuente nueva y evita re-parsear lo que el crawler ya sabe.
- `group:<nombre>` sale de los grupos nombrados de la regex.
- `parsed:<clave>` lo producen los cleaners, por ejemplo keywords o links del HTML, o `content_mode`.
- `const:<valor>` es un valor fijo.
- `extractor:<clave>[.<sub>]` es código, con `params` opcionales.

Los campos admiten `transform`: `value_map`, `date_floor`, `reject_after: ingest_date`, `path_segment` y `verb_lemma`.

## 3. Cleaners (lado parser, genéricos, declarados en `parser_params.cleaners`)

| cleaner | params | uso |
|---|---|---|
| `drop_toc_pages` | `leader_pattern`, `min_leader_ratio`, `max_scan_pages` | Compendio: descarta las 5 páginas de índice con líneas punteadas |
| `strip_running_headers` | `patterns[]`, `zone` (top/bottom/any), `max_lines` | Compendio: elimina la cabecera "Compendio de Normas … - Libro I <n>" de las 453 páginas |
| `strip_repeated_page_lines` | `zone_lines_top/bottom`, `min_page_ratio`, `min_pages`, `normalize_digits` | circulares, NCG, notas y resoluciones: detecta solo el pie o membrete que se repite en ≥ max(2, ratio·páginas) páginas |
| `strip_lines` | `patterns[]` | números de página sueltos, "Verifique documento en …", ids internos `CI-DDN-26-3`, rayas |
| `dedupe_heading_echo` | `pattern` (grupo `key`), `key_normalize`, `window_lines` | Compendio: FOP imprime cada encabezado dos veces (marcador corto, a veces partido en varias líneas, y luego el título completo) |
| `normalize_quotes` | `map`, `ascii_double: alternate` | NCG: deja comillas tipográficas balanceadas para proteger el texto insertado |
| `normalize_whitespace` | `collapse_inner_spaces`, `strip_lines`, `max_blank_lines` | todos (el texto justificado del PDF trae dobles espacios) |
| `drop_pages_before` | `pattern`, `keep_matching_page` | jurisprudencia: descarta portada y prólogo |
| `join_continuation_lines` | `record_start`, `continuation` | jurisprudencia: une los registros y listas de oficios que ocupan varias líneas |
| `html_main_content` | `title_selector`, `include_selectors[]`, `exclude_selectors[]`, `bold_paragraph_as_heading`, `heading_selectors`, `output: markdown` | CMS Newtenberg: conserva solo `#article_…_resumen_y_cuerpo_1` y descarta menús, glosario, footer y "Otras preguntas" |
| `html_collect` | `fields: {name: {selector, attr?, regex?, split?}}` | expone `parsed:keywords` (div oculto con tags) y `parsed:related_aids` |

Política de documentos escaneados (no es un cleaner): `on_scanned: index_card | route_to_ocr_profile`. Hoy `SimpleParser` lanza `ScannedDocumentError`; el perfil decide qué hacer.

## 4. Qué necesita el template y qué es configuración del caso

**Features genéricas del template:**

1. `chunker_params` anidado y validado por un modelo Pydantic propio de cada `chunker_key`. Hoy `ParamValue = str | int | float`, así que `levels` no se puede expresar.
2. `parser_params.cleaners`: pipeline de cleaners resueltos desde un registro en lista blanca, que entra en el fingerprint.
3. Páginas unidas con offsets (`join_pages`, `keep_page_offsets`) para que los chunks crucen páginas y aun así tengan `page`/`page_end`. El `StructuredChunker` actual nunca cruza una página, y en el Compendio un número suele continuar en la página siguiente.
4. Chunker `pattern_hierarchy` con estas capacidades:
   - match **por línea** (no `re.M` sobre el texto completo, ver §5);
   - `depth` para niveles hermanos (Capítulo y Anexo);
   - `role: context` + `reset_on_change` para breadcrumbs repetidos;
   - `keep_heading_line: false`;
   - `ignore_inside` con anidamiento;
   - `oversize` (niveles más profundos, `extra_split_patterns` y recursivo con preferencia por el fin de oración al final de línea);
   - `merge_below_tokens` (solo mismo nivel y padre, conserva la lista de grupos), `fold_intro_below_tokens`, `preamble: keep|drop`;
   - `heading_template` con segmentos condicionales `{x?…}` y truncado `{x:50}`, y `heading_prefix_template`.
5. Chunker `records` (`group_levels`, `record_pattern`, `group_by`, `repeat_group_header`).
6. Nuevas fuentes de metadata `document:`, `parsed:`, `const:`; `extractor:` con `params` (incluido el genérico `regex_all`); `transform`.
7. Tipos `str_list` / `int_list` guardados como arrays JSONB (`metadata @> '{"numeros":["3"]}'` usa el GIN `jsonb_path_ops`). Hoy `MetadataValue` no admite listas. Además el tipo `bool` existe pero `freeze_metadata` rechaza valores bool, por eso los perfiles usan enums `str`.
8. Routing de ingesta declarativo (`routing`: `index_filter` + `title_regex`/`exclude_title_regex`) y el contrato `IngestDocument(..., document_fields=row)` para pasar la fila del índice.
9. Política `on_scanned` y fallback `index_card`.

**Configuración específica del caso:** todas las regex, selectores CSS y plantillas, más un módulo de extractores del caso registrados por clave: `compendio_path`, `compendio_target`, `vigencia_date`, `legal_refs`, `ncg_role`, `title_is_question` y `resolution_topic`.

## 5. Resultados de la validación (ver `validation/validation_report.md`)

| muestra | perfil | chunks | p50 / p90 / max tokens | sobre presupuesto | < 30 tokens |
|---|---|---:|---|---:|---:|
| Compendio Libro I (453 págs) | compendio | 1.413 | 170 / 448 / 690 | 0 | 0 |
| Circular 2425 | circular | 9 | 215 / 487 / 489 | 0 | 0 |
| Circular CAFP 100 (1982) | circular | 4 | 150 / 235 / 235 | 0 | 0 |
| NCG 372 | ncg | 17 | 437 / 567 / 676 | 0 | 0 |
| Libro de Jurisprudencia | jurisprudencia_indice | 57 | 89 / 266 / 390 | 0 | 3 |
| FAQ cms-aid-3590 | articulo_faq | 1 | 327 | 0 | 0 |
| Resolución 17157 (escaneada) | resolucion (ficha) | 1 | 74 | 0 | 0 |

Precisión y recall de los patrones contra ground truth:

- **Compendio, `capitulo`:** 83/83 contra el índice del propio PDF, en el mismo orden. P = R = 1,00.
- **Compendio, `anexo`:** 35/36, P = R = 0,97. El único faltante es "Anexo Información de Rentabilidad…": el encabezado se detecta pero su nombre partido en dos líneas se pierde.
- **Compendio, `numero`:** 1.493 en secuencia y 0 fuera de secuencia. Hay 5 saltos de numeración (posibles FN, sin verificar). Después de limpiar no queda ninguna cabecera corrida ni línea de índice.
- **Circulares:** `numero` 4/4 y 3/3; `seccion` (VIGENCIA, ANEXO) 2/2; literales a–g 7/7. El membrete legacy y el pie "CI-DDN" se eliminaron.
- **NCG 372:** `parte` 2/2; `enmienda` 6/6 (el original repite la letra "C."); `operacion` 9/9 fuera de comillas, con 22 marcadores enmascarados dentro del texto insertado (0 FP).
- **Jurisprudencia:** 219/219 registros y 278/278 referencias a oficios.
- **HTML:** no queda boilerplate (menú, glosario, "Otras preguntas" y footer en 0).
- **Eval:** 29/29 preguntas tienen su snippet dentro de un solo chunk con el metadata del localizador (`validation/eval_locator_check.md`).

Lo que corrigió la iteración (vale como gotcha del template):

1. `[^.;]*` y `\s*$` con `re.M` cruzan saltos de línea y se tragan el encabezado siguiente: el recall de `anexo` bajó a 0,81. **Los niveles deben evaluarse línea a línea.**
2. Las comillas del texto insertado se anidan (“… “Selección de Modalidad” …”). Sin conteo de profundidad hubo 3 operaciones falsas (P = 0,75).
3. Algunas referencias cruzadas quedan al inicio de línea por el ajuste del texto ("Capítulo V. Recaudación, de la presente Letra A, podrá…", "45. anterior."). Se resolvió con guardas de lookahead.
4. Con documentos de 2 páginas, el detector de pie o membrete repetido borraba contenido. Se exige que la línea aparezca en al menos 2 páginas.
5. El breadcrumb del Compendio se repite en cada página de sección y cerraba el Capítulo abierto. Se resolvió con `role: context` + `reset_on_change`.
6. Fusionar números pequeños perdía el número fusionado. Por eso `numeros` es una lista y `compendio_path` llega solo hasta el Capítulo.

Límites conocidos:

- Los subtítulos sin numeración del Compendio ("Determinación", "Cartolas Cuatrimestrales") quedan como texto del primer chunk que les sigue y no como metadata.
- Los anexos de formularios y tablas se extraen como texto plano, de bajo valor.
- En el NCG solo se extrae el destino al nivel de la enmienda; las operaciones con destinos distintos dentro de una misma enmienda no generan una lista.
- En PDFs antiguos quedan espacios dentro de palabras ("Ci rcular").
- Los ratios de chunks y tokens salen de una sola muestra por familia.

## 6. Respuestas a las preguntas de diseño

**4.1 Enmiendas NCG.** Recomiendo un chunk por enmienda (A., B., …), con las operaciones juntas, porque comparten el destino. Si excede el presupuesto se corta en operaciones y luego en los numerales del capítulo insertado.

- Metadata de enlace: `target_libro/titulo/letra/capitulo/numero/anexo` y `compendio_path` (`SP/LIII/TI/H/CVIII`). Se extraen fuera de comillas y se heredan en cascada desde el título de la NCG, luego la "parte" y luego la enmienda. `op_verb` queda normalizado.
- El Compendio usa la misma clave a nivel Capítulo, lo que permite el join directo y la búsqueda por prefijo cuando la enmienda apunta a una Letra entera.
- El Compendio también trae el enlace inverso: las 493 "Nota de actualización" se convierten en `amended_by_ncg[]` (410 de 1.413 chunks tienen valor).
- **Sí, conviene preferir el Compendio consolidado.** El Libro I cita hasta la NCG N° 368 (2026-07-20), y la última NCG del índice es la 372 (2026-08-18): el rezago es de alrededor de un mes. Las NCG pasan a ser historia, salvo las "pendientes de consolidar" (`ncg_number` mayor que el máximo `amended_by_ncg` del Libro destino), que se tratan como vigentes y se recuperan junto al Compendio.
- Esto encaja con el grafo: un sub-agente "normativa vigente" (Compendio + NCG pendientes) y otro "historia normativa" (`spensiones_ncg`) para preguntas del tipo "¿qué norma modificó…?" o "¿desde cuándo…?".

**4.2 Vigencia.**

- Qué guardar:
  - `issued_date` (`document:date` con `date_floor`) junto con `date_precision`.
  - `listing_tab` (renombrado a propósito: periódica/histórica es la pestaña del listado, **no** vigencia legal).
  - `effective_from`, extraído de la sección VIGENCIA. En las muestras: 2425 → 2026-10-01 y NCG 372 → 2026-09-01; la CAFP100 no tiene sección VIGENCIA y queda nula.
  - `validity: consolidated_current` para el Compendio.
- No hay filtro duro por defecto (`default_filter: null`), porque no existe un dato confiable de "derogada". Las fechas se usan para ordenar, para citar y para avisar "aún no rige" cuando `effective_from` es posterior a hoy. Los filtros explícitos se aplican solo si el usuario pide una fecha.
- Fechas futuras: son 3 filas, todas fuera de las familias del MVP y todas parseadas del título. Una es legítima (régimen "vigente a partir del 1 de abril de 2027") y dos son errores de parseo (2030 y 2027 en un documento de trabajo). Regla: `reject_after: ingest_date` anula `issued_date` cuando la fecha es futura. En cambio, un `effective_from` futuro sí es válido.
- Circulares periódicas (615): se reemplazan cada mes. Propongo una clave de serie (título normalizado) y recuperar por defecto solo la más reciente de cada serie.

**4.3 Resoluciones escaneadas.**

- El índice no trae ninguna señal de capa de texto. La única muestra es una impresión web a PDF con 4 páginas de imagen y 0 caracteres, y el parser `simple` lanza `ScannedDocumentError`.
- Recomendación:
  - Etapa 1: `index_card` (título + descripción del índice, que tienen 247 de 281 resoluciones). Es casi gratis y permite citarlas por número y fecha.
  - Etapa 2 (medida, no adoptada): OCR con RapidOCR (parser `ocr_index_card`, extra `ocr`, ~4 s por página en CPU). En la aceptación en vivo la pregunta cuyo snippet sale de la ficha bajó del rank 1 al 3 (MRR@10 denso 0,7937 -> 0,7766, más que el 0,01 permitido), aunque la respuesta de cuerpo de in-res-30 pasó a ser recuperable. El caso sigue con la ficha; el parser queda como capacidad opt-in de la plantilla.
- La proporción real de documentos que necesitan OCR se mide en la etapa 1 contando los `ScannedDocumentError` (la prueba con pypdf tarda milisegundos). Hoy el único dato es 1 de 1.

**4.4 Familias mal clasificadas.** Recomiendo enrutar por **campos del índice más overrides deterministas** de título y sección, y no por detección de contenido. Es barato, ocurre antes de descargar, es explicable y se puede testear.

- Evidencia, `guia_faq` (120 filas):
  - 117 son "Nota Interna" (dictámenes);
  - 1 es el índice del Libro de Jurisprudencia;
  - 2 son "Comisión Máxima por Asesoría Previsional";
  - ninguna es FAQ.
- Evidencia, `articulo_html`: 6 filas son "Resolución Exenta" y 74 de 204 son páginas guía, no preguntas.
- La detección por contenido queda como guarda posterior al parseo: si `pattern_hierarchy` no encuentra el `split_level` en un documento de más de N tokens, se marca `profile_mismatch` y se cae a `structured`.

**4.5 MVP en hardware modesto.** Estimación con TEI en CPU a ~8 s por lote de 16 chunks: la cota por chunks es pesimista porque nuestros chunks miden ~220 tokens y no 500; la cota por tokens supone ~1.000 tok/s. Detalle en `validation/index_estimates.md`.

| fase | contenido | docs | chunks est. | embedding |
|---|---|---:|---:|---|
| 1 (MVP) | Compendio SP+SC (10 Libros), FAQ (204), índice de jurisprudencia, fichas de resoluciones (287), NCG ≥ 2020 (159) | ~660 | ~17k (Compendio ±2×) | ~1,2–2,4 h |
| 2 | resto de las NCG (326), circulares ≥ 2015 (521) | ~850 | ~10–21k | ~1,5–3 h |
| 3 | Notas Internas (validar perfil antes), OCR de las 185 resoluciones normativas, circulares históricas < 2015 | ~2.300 | ~20–60k | horas; OCR aparte |

La razón del orden: el Compendio es la norma consolidada y vigente, y responde la mayoría de las preguntas de afiliados con ~14k chunks. Las FAQ aportan el lenguaje del ciudadano y las NCG recientes cubren el rezago de consolidación. Las circulares (estimadas en 23k–75k chunks porque el tamaño medio es de 640 KB) son caras y en su mayoría históricas o periódicas.

## 7. Próximos pasos

1. Descargar 3–5 Notas Internas y 2–3 resoluciones normativas para validar `guia_jurisprudencia_nota` y el OCR.
2. Implementar las features 1–9 del §4 en el template, usando el prototipo como oráculo de comportamiento: sus casos de `validation_report.md` sirven como fixtures de test.
3. Medir el retrieval con `eval_questions.jsonl` comparando el perfil estructural contra `recursive` de 256–512 tokens, como recomienda el reporte.
