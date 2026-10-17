# Case: FastAPI docs in Spanish

**English** | [Español](README.es.md)

Second example case of the template. It exists to show the template is generic: a Markdown corpus
(not PDF/HTML), an index without titles (they come from each page's H1, with the `{ #id }`
heading attributes dropped by the Markdown parser), a different domain and two profiles/topics.
Everything is configuration: `profiles/` (knowledge) + `agents/` (the `fastapi-es-rag` agent).

Layout: `fetch/` (acquisition at a pinned commit), `samples/` (see its README), `profiles/`,
`agents/`, `calibration/` (dev sets `in_dev.jsonl` / `ood.jsonl` for the evidence thresholds, see
its README). `index/` is generated and gitignored. Clarification policy is `none` until its
thresholds are calibrated. `profiles/eval_questions.jsonl` is HELD OUT: never tune prompts,
retrieval or thresholds on it (nor calibrate with it).

## Run

Needs the compose `db` and `embeddings` services (see the main README).

```bash
uv run --no-project --python 3.12 python -I case/fastapi_es/fetch/fetch.py
uv run python -m agintis.cli.ingest --profiles case/fastapi_es/profiles \
    --index case/fastapi_es/index/documents.jsonl
```

After a re-fetch that changed or dropped source URIs, add `--prune` to the ingest (full index, no
`--limit`): it removes the READY documents of these profiles whose source is no longer in the
batch (only when nothing failed, something was ingested and every index row has its file on disk;
every removal is logged). "In the batch" means in THIS index file: keep one index per profile set
and never run two ingests over the same profiles concurrently, or they prune each other.

The agent id is `fastapi-es-rag`. Calibrate the evidence thresholds on the dev sets (read-only;
`--analyze` scores the analyzer's standalone query like the runtime and needs the `LLM_*` keys):

```bash
uv run python -m agintis.cli.calibrate_evidence --profiles case/fastapi_es/profiles \
    --agents case/fastapi_es/agents --agent fastapi-es-rag --analyze \
    --report case/fastapi_es/calibration/report.json \
    case/fastapi_es/calibration/in_dev.jsonl case/fastapi_es/calibration/ood.jsonl
```

Serve the agent with `KNOWLEDGE_PROFILES_DIR=case/fastapi_es/profiles`,
`AGENTS_DIR=case/fastapi_es/agents` and the `LLM_*` settings. For `cli.dump_answers` / `cli.judge`
the named LLM profiles of `case/spensiones/llm_profiles.yaml` can be reused through
`LLM_PROFILES_FILE` (this case ships no copy).

## Results (part 4.2)

Corpus: FastAPI docs in Spanish (MIT) at tag 0.143.3 / commit bb7f314: 64 documents (13 `guia` +
51 `tutorial`), 735 chunks. The fetcher inlines the `docs_src` includes (`ln[]` ranges honored; it
fails on a missing file or an unmatched include).

Evidence thresholds (`calibrate_evidence --analyze`, dev sets only: `in_dev` 30 / `ood` 40, 12
near-domain traps checked against the corpus with `rg`; written apart from the held-out
`eval_questions.jsonl`, 30 questions over 30 documents): `low_confidence_below` 0.6572,
`out_of_scope_below` 0.4922. AUROC 0.987, `in` recall above the low threshold 0.90. Out of domain
above `low`: the Dockerfile and rate limiting questions (the corpus has no page for them). `in`
below `low`: in-fa-08, in-fa-21, in-fa-24.

Held-out retrieval (`cli.eval`, LLM-free; dense == hybrid, see finding a). Report only, never used
to tune anything:

| hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 |
|-------|-------|-------|--------|--------|
| .467  | .700  | .767  | .800   | .586   |

By profile: `guia` hit@5 1.0; `tutorial` hit@10 .727.

Live judges (v9 for the table, `deepseek-chat`; generator `gpt-5.4-mini`; 2 dumps per set):

| set    | metric              | dump 1 | dump 2 |
|--------|---------------------|--------|--------|
| in_dev | faithfulness        | 28/30  | 29/30  |
| in_dev | citation_support    | 22/30  | 22/30  |
| in_dev | answer_relevance    | 30/30  | 30/30  |
| in_dev | refusal_correctness | 30/30  | 30/30  |
| in_dev | script_mix          | 30/30  | 30/30  |
| ood    | refusal_correctness | 40/40  | 40/40  |
| ood    | script_mix          | 40/40  | 40/40  |

The `citation_support` 22/30 is a v9 artifact (code blocks split per line). With judge v10 (part 4.2c)
it is 27/30 (dump a) and 26/30 (dump b); the other rows are unchanged.

Judge meta-eval (part 4.2c): new blind `citation_support` silver labels for `in_dev` (opus + sonnet,
59/60 inter-labeler; `calibration/answer_judge/silver_labels_in_dev_{a,b}.jsonl`). Agreement with the
unanimous consensus, identical over 2 runs:

| judge | dump a (n=29) | dump b (n=30) |
|---|---|---|
| v9  | 75.9% | 70% |
| v10 | 93.1% | 90% |
| v11 | 93.1% | 96.7% / 93.3% (2 runs) |

Dump a: 93.1% in both v11 runs. The remaining fails (in-fa-06, in-fa-15) are `evidence_not_verbatim`
caused by inline HTML in the sources, not by tables. A tag-free source variant was tried and dropped
(removing inline tags joins digits: 10<sup>6</sup> -> 106 lets a changed number pass as exact); accepted
cost. Reproduce:
`case/fastapi_es/judge/work/run_42c.sh` (gitignored, judge-only), then `cli.agreement --labels <silver>
--judgments <report>`.

Reproduce: `case/fastapi_es/judge/work/run_42.sh calibrate|answers`. The script and the dumps live
under the gitignored `judge/` (not in the repo); it prompts for the keys with `read -rs`, so run it
in your own terminal (no TTY, no prompt).

### Deferred findings (not fixed in 4.2; b fixed in 4.2c)

- a. Hybrid == dense here and in spensiones: `search_lexical` (`postgres/chunks.py`) uses
  `plainto_tsquery`, an AND over the whole question, which gives 0 lexical hits on 30/30 of these
  eval questions (and 33/40 spensiones `in_dev`), so RRF keeps the dense order. Published examples
  (Supabase, LangChain) AND too. Part 4.2b tried an OR over the question lexemes with document
  frequency <= 10/20/30% of the scope (`ts_rank_cd` normalization 1, top-N 20/50) on the retrieval
  dev sets (`calibration/retrieval_dev.jsonl` here and in spensiones) and REJECTED it by the
  pre-registered rule: every config lost MRR@10 against dense and the old hybrid on both sets
  (fastapi .689-.722 vs .792; spensiones .650-.706 vs .794) and 4-6 items of hit@1. Without IDF the
  lexical leg adds noise to a strong bge-m3 ranking. Unmerged branch `part-4.2b` (109025e).
- b. FIXED in 4.2c (judge v10) and 4.2d (v11): `split_sentences` split fenced code blocks per line, so
  markers stuck to fragments ("``` [1]"). v10 keeps the block whole; v11 ignores markers in code (fenced and
  inline), see Results. Accepted: `evidence_not_verbatim` on sources with inline HTML (in-fa-06, in-fa-15;
  re-diagnosed, not tables). Still open: fences indented 4+ (shortcut comment in `citations.py`).
- c. The `REFUSAL_DEFINITION` "where it is" examples are spensiones-flavoured (index entry /
  document number); widen to "page, section" at the next judge meta-eval, not before.
- d. Out of scope: a non-RAG agent, OCR (in-res-30), the in-cir-26 heading tweak (it would tune on a
  calibration item).
