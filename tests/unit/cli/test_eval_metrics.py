"""Relevance judgement and retrieval metrics of the eval CLI (pure code, no database)."""

import json
from pathlib import Path
from typing import Any

import pytest

from agintis.cli.eval import build_parser, parse_ks
from agintis.cli.eval_metrics import (
    EvalQuestion,
    QuestionOutcome,
    first_relevant_rank,
    format_table,
    is_relevant,
    load_questions,
    locator_matches,
    summarize,
)

QUESTION = EvalQuestion(
    id="q1",
    question="¿Qué dice?",
    type="exact_reference",
    profile="p",
    expected_snippet="inicio de   labores",
    locator={"compendio_path": {"prefix": "SP/LI/TI/A"}, "numeros": "3"},
)
GOLDEN = (
    "Compendio\nEl INICIO de labores genera la afiliación",
    {"compendio_path": "SP/LI/TI/A/CII", "numeros": ["3"]},
)


def _outcome(qid: str, rank: int | None, profile: str = "p") -> QuestionOutcome:
    return QuestionOutcome(qid, profile, "semantic", rank, 10)


class TestRelevance:
    def test_snippet_and_locator_must_both_match(self) -> None:
        assert is_relevant(GOLDEN[0], GOLDEN[1], QUESTION)
        assert not is_relevant(GOLDEN[0], {**GOLDEN[1], "numeros": ["4"]}, QUESTION)
        assert not is_relevant("Compendio\nOtro texto", GOLDEN[1], QUESTION)

    def test_the_heading_line_is_not_searched(self) -> None:
        assert not is_relevant("inicio de labores\ncuerpo distinto", GOLDEN[1], QUESTION)

    def test_list_metadata_matches_as_tuple_or_list(self) -> None:
        assert locator_matches({"numeros": ("1", "3")}, "numeros", "3")
        assert locator_matches({"numeros": ["1", "3"]}, "numeros", "3")
        assert not locator_matches({"numeros": ["1"]}, "numeros", "3")

    def test_a_scalar_locator_is_plain_equality_whatever_the_key(self) -> None:
        """No key is special: `compendio_path` equality is exact unless the question says prefix."""
        assert locator_matches({"compendio_path": "SP/LI"}, "compendio_path", "SP/LI")
        assert not locator_matches({"compendio_path": "SP/LI/TI"}, "compendio_path", "SP/LI")

    def test_a_prefix_locator_matches_the_start_of_the_stored_string(self) -> None:
        wanted = {"prefix": "SP/LI/TI/A"}
        assert locator_matches({"compendio_path": "SP/LI/TI/A/CII"}, "compendio_path", wanted)
        assert locator_matches({"compendio_path": "SP/LI/TI/A"}, "compendio_path", wanted)
        assert not locator_matches({"compendio_path": "SP/LI/TII"}, "compendio_path", wanted)
        assert locator_matches({"paths": ["X", "SP/LI/TI/A/C1"]}, "paths", wanted)  # any item
        assert not locator_matches({"n": 5}, "n", {"prefix": "5"})  # strings only

    def test_a_malformed_locator_is_rejected_when_loading(self, tmp_path: Path) -> None:
        row = {"id": "q", "question": "?", "profile": "p", "expected_snippet": "s"}
        for bad in ({"k": {"suffix": "x"}}, {"k": {"prefix": 3}}, {"k": {"prefix": "a", "x": 1}}):
            path = tmp_path / "q.jsonl"
            path.write_text(json.dumps({**row, "locator": bad}) + "\n", encoding="utf-8")
            with pytest.raises(ValueError, match="q: locator 'k'"):
                load_questions(path)

    def test_missing_metadata_never_matches(self) -> None:
        assert not locator_matches({}, "numeros", "3")


class TestRanking:
    def test_first_relevant_rank_is_one_based(self) -> None:
        miss: tuple[str, dict[str, Any]] = ("Titulo\nnada", {})
        assert first_relevant_rank([miss, miss, GOLDEN], QUESTION) == 3
        assert first_relevant_rank([miss], QUESTION) is None
        assert first_relevant_rank([], QUESTION) is None


class TestMetrics:
    def test_hit_at_k_and_mrr(self) -> None:
        outcomes = [_outcome("a", 1), _outcome("b", 3), _outcome("c", None), _outcome("d", 10)]
        summary = summarize(outcomes, [1, 3, 5, 10])
        assert summary.questions == 4
        assert summary.hit_at == {1: 0.25, 3: 0.5, 5: 0.5, 10: 0.75}
        assert summary.mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 10) / 4)
        assert summary.mrr_depth == 10

    def test_no_questions(self) -> None:
        summary = summarize([], [1, 5])
        assert (summary.questions, summary.hit_at, summary.mrr) == (0, {1: 0.0, 5: 0.0}, 0.0)
        assert summary.mrr_depth == 5

    def test_the_table_has_a_row_per_profile_and_an_overall_row(self) -> None:
        table = format_table([_outcome("a", 1, "x"), _outcome("b", None, "y")], [1, 5])
        lines = table.splitlines()
        assert lines[0].split()[:2] == ["profile", "n"]
        assert [line.split()[0] for line in lines[1:]] == ["x", "y", "OVERALL"]
        assert lines[-1].split()[1] == "2"
        assert lines[0].split()[-1] == "MRR@5"  # the depth the MRR was measured at


class TestLoadingAndArguments:
    def test_questions_are_read_from_json_lines(self, tmp_path: Path) -> None:
        row = {
            "id": "q",
            "question": "?",
            "type": "semantic",
            "profile": "p",
            "expected_snippet": "s",
            "locator": {"a": 1},
            "extra": "ignored",
        }
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
        (question,) = load_questions(path)
        assert (question.id, question.profile, dict(question.locator)) == ("q", "p", {"a": 1})

    def test_a_question_without_a_gold_profile_groups_under_none(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        path.write_text(
            json.dumps({"id": "q", "question": "?", "expected_snippet": "s"}) + "\n",
            encoding="utf-8",
        )
        (question,) = load_questions(path)
        assert question.profile == "none"
        table = format_table([_outcome("q", 1, question.profile)], [1])
        assert [line.split()[0] for line in table.splitlines()[1:]] == ["none", "OVERALL"]

    def test_ks_are_sorted_and_deduplicated(self) -> None:
        assert parse_ks("10, 3,1,3") == (1, 3, 10)

    @pytest.mark.parametrize("raw", ["", "a", "0", "1,-2"])
    def test_invalid_ks(self, raw: str) -> None:
        with pytest.raises(Exception, match="--k"):
            parse_ks(raw)

    def test_defaults(self) -> None:
        args = build_parser().parse_args(["--profiles", "p", "--questions", "q"])
        assert (args.k, args.mode, args.rerank, args.report) == ((1, 3, 5, 10), "all", False, None)
