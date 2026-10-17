# Evidence calibration sets (spensiones)

Dev sets for `python -m agintis.cli.calibrate_evidence` (never the held-out `eval_questions.jsonl`).

- `ood.jsonl`: 80 `out` queries. `in_dev.jsonl`: 40 `in` queries.
- Label criterion for `out` (fixed): the ingested corpus (tenant `default`) does NOT contain an
  answer to the question. `in`: it does.
- The `in` queries were written looking at the documents, so recall and AUROC are optimistic.
- A threshold is valid only for the corpus snapshot, tenant, embedding model and profiles in the
  report `meta`. Recalibrate after a re-ingest.

## Routing dev set 2 (`routing_dev2.jsonl`)

49 `in` questions in the eval format (`profile`, `expected_snippet`, `locator`, `type`) over the
34-document corpus of `samples/_manifest.json` (fetch it with `case/spensiones/fetch_samples.py`).
Use it for routing and retrieval experiments; it is a dev set, never a held-out one.

- Corpus: the original 8 samples plus a date-ordered selection made without looking at any score:
  10 historical CAFP circulars (rate tables excluded), 6 NCGs (NP 366-371), 7 FAQ-style articles
  and 3 normativa resolutions. Five articles whose body is loaded by JavaScript were dropped.
- Writing: three blind writers worked from the extracted document text, without access to the
  router, the probes or any score. At most 30% of the questions name a document.
- Checks: every snippet is in a ready chunk of its gold profile (mechanical check); 4 questions too
  close to `eval_questions.jsonl`, `in_dev.jsonl`, `retrieval_dev.jsonl` or `ood.jsonl` (token
  Jaccard >= 0.5) were dropped without reading those files into a prompt.
- Gold profiles: compendio 11, articulo_faq 9, ncg 9, circular 9, jurisprudencia_indice 8,
  resolucion 3.

Pre-registered results (2026-10-17), evidence router, tenant with this corpus:

| Router | hit@1 | any selected | compendio hit@1 | topics/question | old `retrieval_dev` hit@1 |
|---|---|---|---|---|---|
| max, no bias (current) | 34/49 | 39 | 11/11 | 1.22 | 34/39 |
| per-topic bias from the `ood.jsonl` medians | 38/49 | 42 | 9/11 | 1.20 | 32/39 |

The bias variant failed two of the five criteria (compendio and the old set may drop by at most
one question), so the router keeps `max` without biases. The set confirms the size bias: 10 of the
15 misses of the current router go to compendio.

## Label verification (near-domain `out` items)

Method (content only, scores were NOT used or looked at to decide): for each of the 36
near-domain `out` items (`pens`, `labor`, `isapre`, `tax`, `bank`, `abroad`), (1) the top-3
retrieved chunks of every profile were read, and (2) the text of every READY chunk of the tenant
was searched (accent-insensitive regex) with the terms of the question. A verdict is `out` when no
chunk answers the question, even if it mentions the topic in passing.

Corpus at verification time (READY chunks): compendio 1413, ncg 17, circular 13, resolucion 1,
articulo_faq 1 (seguro de cesantia), jurisprudencia_indice 57, jurisprudencia_nota 0.

Result: 36/36 verified `out`, 0 relabelled. `indice` chunks are lists of oficio titles and
`resolucion` is a lobby-law resolution: none answers any of them.

| id | verdict | evidence |
|---|---|---|
| out-isapre-01 (change of Isapre) | out | only mentions ISAPRE/FONASA as the destination of the health contribution (compendio chunks 94d0e41b, 96b7dbd0); no switching procedure |
| out-isapre-02 (GES) | out | no mention of GES / explicit health guarantees |
| out-isapre-03 (plan price at 60) | out | no health-plan pricing text |
| out-isapre-04 (Superintendencia de Salud complaint) | out | "Superintendencia de Salud" absent; the complaint chapters of the compendio (Titulo VII Gestion de Reclamos, Titulo VIII Cap. V Reclamacion) concern AFP and pension-account claims |
| out-isapre-05 (Ley de Urgencia) | out | absent |
| out-isapre-06 (FONASA tramos) | out | FONASA appears only as a collection account (63fa0f66, d8aa3fc5); "tramo B/C" absent |
| out-tax-01 (Operacion Renta) | out | only "proceso anual de declaracion de impuesto a la renta" as the channel where independents pay contributions (compendio 00c82efb); no filing instructions |
| out-tax-02 (company deductions) | out | no text on deductible expenses |
| out-tax-03 (boleta de honorarios) | out | absent; SII appears only as a data source (058c02a9, 12ae3844) |
| out-tax-04 (IVA deadline) | out | IVA absent |
| out-tax-05 (inicio de actividades SII) | out | absent |
| out-tax-06 (impuesto territorial) | out | absent |
| out-labor-01 (aviso previo al renunciar) | out | no notice-period text |
| out-labor-02 (severance by years of service) | out | closest: compendio d87cafcf / 4505d90a, the "pacto de indemnizacion sustitutiva" (art. 164 CT replaced after year 6 by an employer contribution of at least 4.11%); it never explains how the legal severance is computed in a dismissal |
| out-labor-03 (vacation days) | out | only "bonos por vacaciones" as a taxable item (db866498) |
| out-labor-04 (weekly working hours) | out | only the employer's duty to report hours (798c7043); no legal maximum |
| out-labor-05 (overtime pay) | out | absent |
| out-labor-06 (fuero maternal) | out | absent |
| out-abroad-01 (Spain) | out | no foreign system text |
| out-abroad-02 (ANSES, Argentina) | out | absent |
| out-abroad-03 (US Social Security) | out | absent |
| out-abroad-04 (Mexico IMSS) | out | absent |
| out-abroad-05 (Germany) | out | absent |
| out-abroad-06 (AFORE, Mexico) | out | absent |
| out-bank-01 (mortgage) | out | "hipotecari", "credito", "prestamo" absent |
| out-bank-02 (credit vs debit card) | out | "tarjeta" absent |
| out-bank-03 (loan installment) | out | absent |
| out-bank-04 (CAE) | out | absent |
| out-bank-05 (open a checking account) | out | "cuenta corriente" appears only as AFP collection accounts (09633329) |
| out-bank-06 (DICOM) | out | absent |
| out-pens-01 (retirement age, men) | out | no chunk states the legal retirement age; "edad legal" is used without the number (3ded2f25, 69c7cc6b); 65 anos appears only for contribution-exemption rules (e59e9f9d, a3844eb0) and men 60 / women 55 for fund assignment (216ff5e0, a0ea91d5) |
| out-pens-02 (Pension Garantizada Universal) | out | "pension garantizada" / PGU absent |
| out-pens-03 (percentage withdrawal in an emergency) | out | closest: COVID withdrawals of the severance-savings account under Ley 21.227 / 21.247 (compendio 7445fa79, 4a8e41ae), not a withdrawal of pension funds |
| out-pens-04 (alimony amount) | out | alimony appears only as the procedure to collect debts from the member's funds (3ded2f25); no amount rule |
| out-pens-05 (AFP returns per analysts) | out | no ranking or analyst text; "rentabilidad" appears in definitions and in circular 2425's rate method |
| out-pens-06 (how to apply to the bono por hijo) | out | only the definition of the "bonificacion fiscal por hijo nacido vivo" (compendio 5ee9167b) and accrediting it in the account; no application procedure |

The other 44 `out` items (`gen`, `chat`, `code`, `other`) are unrelated to pensions, health,
labour, tax or banking and were not checked chunk by chunk.

## Edits to `in_dev.jsonl`

These are `in` items (they only report, they never set a threshold):

- `in-ncg-20`: rewritten as a standalone question (the original said "de este tipo", which needed
  the previous turn). Answer: NCG 372, "Procedimiento de Notificacion", within the third day of the
  sentence being final (chunk b681c5a4).
- `in-cir-26`: replaced by a different question. Answer: circular 100, the payment is credited to
  the individual account at the share value of the previous day's close (chunk e2645ced).
