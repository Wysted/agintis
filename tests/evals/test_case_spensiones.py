"""Acceptance of the configurable chunking on the spensiones case, over its LOCAL samples.

Runs the real pipeline (routing, parser, cleaners, pattern_hierarchy / records / whole_document
chunkers, metadata sources, `conform_metadata`) with the real bge-m3 tokenizer, without a
database or an embedding service. Ports `case/spensiones/profiles/validation/check_eval.py`
(the matching rule lives in `agintis.cli.eval_metrics`, shared with the retrieval eval CLI):
every eval question's snippet must sit in exactly one chunk whose metadata matches the
question's locator. Skipped where the raw samples or the tokenizer are not available.

The expected figures come from `validation/validation_report.md` (the reference prototype).
"""

import asyncio
import json
import statistics
from typing import Any

import pytest

from agintis.cli.eval_metrics import chunk_body, locator_matches, normalize
from agintis.modules.knowledge.domain import Chunk
from agintis.modules.knowledge.infrastructure.chunkers import HfTokenCounter
from agintis.modules.knowledge.infrastructure.chunkers.splitting import SPECIAL_TOKENS_RESERVE
from tests.support.case_spensiones import (
    PROFILES_DIR,
    SampleResult,
    ingest_samples,
    samples_available,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not samples_available(), reason="case samples are not downloaded"),
]

EVAL_FILE = PROFILES_DIR / "eval_questions.jsonl"
QUESTIONS: list[dict[str, Any]] = (
    [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line]
    if EVAL_FILE.is_file()
    else []
)

# sample id -> (profile, chunks in the reference report). The only divergence is documented in
# `test_resolution_card_carries_the_heading_prefix_the_reference_left_out`.
REFERENCE_CHUNK_COUNTS = {
    "compendio-sp-libro-i": ("spensiones_compendio", 1413),
    "circular-2425": ("spensiones_circular", 9),
    "circular-cafp-100": ("spensiones_circular", 4),
    "ncg-sp-372": ("spensiones_ncg", 17),
    "cms-aid-14843": ("spensiones_jurisprudencia_indice", 57),
    "cms-aid-3590": ("spensiones_articulo_faq", 1),
    "cms-aid-17157": ("spensiones_resolucion", 1),
}
CHUNK_BUDGETS = {
    "spensiones_compendio": 700,
    "spensiones_circular": 600,
    "spensiones_ncg": 700,
    "spensiones_jurisprudencia_indice": 400,
    "spensiones_articulo_faq": 512,
    "spensiones_resolucion": 600,
}


@pytest.fixture(scope="module")
def counter() -> HfTokenCounter:
    tokenizer = HfTokenCounter("BAAI/bge-m3")
    try:
        tokenizer.warm_up()
    except Exception as exc:
        pytest.skip(f"bge-m3 tokenizer unavailable: {exc.__class__.__name__}")
    return tokenizer


@pytest.fixture(scope="module")
def results(counter: HfTokenCounter) -> dict[str, SampleResult]:
    return asyncio.run(ingest_samples(counter))


def _sample_id(question: dict[str, Any]) -> str:
    return str(question["source_sample"]).rsplit("/", 1)[-1].rsplit(".", 1)[0]


def test_there_are_29_eval_questions() -> None:
    assert len(QUESTIONS) == 29


@pytest.mark.parametrize("question", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_each_eval_question_lands_in_exactly_one_chunk_with_its_locator(
    question: dict[str, Any], results: dict[str, SampleResult]
) -> None:
    result = results[_sample_id(question)]
    assert result.profile == question["profile"]
    snippet = normalize(question["expected_snippet"])
    # The first line of a chunk is its heading prefix (document title, path...): the snippet
    # must be found in the body, as in the reference check.
    hits: list[Chunk] = [c for c in result.chunks if snippet in normalize(chunk_body(c.text))]
    located = [
        c
        for c in hits
        if all(locator_matches(c.metadata, key, want) for key, want in question["locator"].items())
    ]
    # Some sentences recur verbatim elsewhere (an annex repeating a rate): what must be unique
    # is the chunk that has BOTH the snippet and the locator metadata.
    assert hits, f"{question['id']}: snippet not found in any chunk"
    assert len(located) == 1, (
        f"{question['id']}: {len(located)} chunks hold the snippet with locator "
        f"{question['locator']}; the {len(hits)} chunks with the snippet have "
        f"{[{k: c.metadata.get(k) for k in question['locator']} for c in hits[:3]]}"
    )


@pytest.mark.parametrize(("sample", "expected"), REFERENCE_CHUNK_COUNTS.items())
def test_chunk_counts_match_the_reference_report(
    sample: str, expected: tuple[str, int], results: dict[str, SampleResult]
) -> None:
    profile, count = expected
    assert results[sample].profile == profile
    assert len(results[sample].chunks) == count


def test_every_chunk_fits_its_profile_budget(
    results: dict[str, SampleResult], counter: HfTokenCounter
) -> None:
    for result in results.values():
        if result.profile is None:
            continue
        limit = CHUNK_BUDGETS[result.profile] - SPECIAL_TOKENS_RESERVE
        sizes = [counter.count(c.text) for c in result.chunks]
        assert max(sizes) <= limit, f"{result.sample_id}: {max(sizes)} > {limit}"
        assert all(c.text.strip() for c in result.chunks)


def test_compendio_token_distribution_is_close_to_the_reference(
    results: dict[str, SampleResult], counter: HfTokenCounter
) -> None:
    """Reference: p50 170 / p90 448 / max 690, 311,555 tokens. Ours only adds the longer
    heading prefix ("Compendio pensiones Libro I" instead of "Compendio SP Libro I")."""
    sizes = [counter.count(c.text) for c in results["compendio-sp-libro-i"].chunks]
    ordered = sorted(sizes)
    assert statistics.median(sizes) == pytest.approx(170, abs=3)
    assert ordered[round(0.9 * (len(ordered) - 1))] == pytest.approx(448, abs=5)
    assert max(sizes) == pytest.approx(690, abs=5)
    assert sum(sizes) == pytest.approx(311_555, rel=0.01)


def test_resolution_card_carries_the_heading_prefix_the_reference_left_out(
    results: dict[str, SampleResult], counter: HfTokenCounter
) -> None:
    """The scanned sample falls back to its index card (`content_mode: index_card`). The
    reference chunked the bare card (74 tokens); here it is a preamble chunk of the profile
    and so starts with the `heading_prefix_template` (83 tokens)."""
    (chunk,) = results["cms-aid-17157"].chunks
    assert chunk.metadata["content_mode"] == "index_card"
    assert chunk.text.startswith("Resolución N° 1266 (2026-09-15)\n")
    assert counter.count(chunk.text) == 83


def test_a_navigation_hub_is_not_routed(results: dict[str, SampleResult]) -> None:
    hub = results["compendio-sp-libro-i-titulo-i"]
    assert hub.profile is None
    assert hub.chunks == ()


def test_list_metadata_is_stored_as_tuples(results: dict[str, SampleResult]) -> None:
    compendio = results["compendio-sp-libro-i"].chunks
    assert any(isinstance(c.metadata.get("amended_by_ncg"), tuple) for c in compendio)
    assert all(
        isinstance(c.metadata["numeros"], tuple) for c in compendio if "numeros" in c.metadata
    )


def test_generic_extractors_reproduce_the_reference_legal_refs_and_effective_date(
    results: dict[str, SampleResult],
) -> None:
    """`regex_labeled` / `date_after_heading` (params in the YAML) replaced the Chilean code that
    used to live in the template; the reference preview shows what they must still produce."""
    first = results["ncg-sp-372"].chunks[0]
    assert first.metadata["effective_from"] == "2026-09-01"
    assert first.metadata["cited_laws"] == ("DL 3500", "Ley 20255", "Ley 21818")
