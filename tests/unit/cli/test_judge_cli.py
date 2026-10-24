import json
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agintis.agent_kit.judge import JudgeName, ScopeVerdict, prompts
from agintis.cli import judge as judge_cli
from agintis.cli._common import EXIT_ABORTED, EXIT_OK, EXIT_USAGE, UsageError
from agintis.cli.judge import EXIT_PARTIAL, load_dump, parse_judges
from agintis.platform.usage import Purpose
from tests.fakes.llm import RaisingModel, ScriptedChatModel, chat_models_for, rate_limit_error
from tests.support.settings import make_settings
from tests.unit.agent_kit.judge.support import checks, claims, make_record, relevance
from tests.unit.cli.conftest import FakeLedger


def _by_schema(schema: object, messages: list[Any]) -> object:
    refusal = "No lo sé" in messages[-1].text  # the judge model recognises a refusal itself
    return {
        "FaithfulnessOutput": claims(is_refusal=True) if refusal else claims(True, False),
        "CitationSupportOutput": checks(True, True),
        "RelevanceOutput": relevance(not refusal, is_refusal=refusal),
    }[getattr(schema, "__name__", "")]


def _dump(path: Path, *records: Any) -> Path:
    path.write_text("".join(json.dumps(r.to_json()) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def answers(tmp_path: Path) -> Path:
    return _dump(
        tmp_path / "answers.jsonl",
        make_record(id="ok"),
        make_record("No lo sé.", id="refused", expected_scope="out", sources=()),
        make_record("El plazo es de 24 meses.", id="nocite", expected_scope=None),
        make_record(
            "Mensaje fijo.", id="scope-out", expected_scope="in", scope=ScopeVerdict("out", 0.1)
        ),
    )


def _patch_model(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    monkeypatch.setattr(judge_cli, "build_chat_models", lambda _settings: chat_models_for(model))
    monkeypatch.setattr(
        judge_cli, "get_settings", lambda: make_settings(eval={"judge_profile": "default"})
    )


# --- loading and arguments ----------------------------------------------------------------------


def test_load_dump_reads_records_in_order(answers: Path) -> None:
    assert [r.id for r in load_dump(answers)] == ["ok", "refused", "nocite", "scope-out"]


def test_load_dump_rejects_duplicates_and_bad_lines(tmp_path: Path) -> None:
    dup = _dump(tmp_path / "dup.jsonl", make_record(id="a"), make_record(id="a"))
    with pytest.raises(UsageError, match="already used"):
        load_dump(dup)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"schema": 9}\n', encoding="utf-8")
    with pytest.raises(UsageError, match=r"bad\.jsonl:1: invalid record"):
        load_dump(bad)
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(UsageError, match="invalid record"):
        load_dump(bad)
    bad.write_text("\n\n", encoding="utf-8")
    with pytest.raises(UsageError, match="no records"):
        load_dump(bad)
    with pytest.raises(UsageError, match="cannot read"):
        load_dump(tmp_path / "missing.jsonl")


def test_parse_judges() -> None:
    assert parse_judges(None) == list(JudgeName)
    assert parse_judges("faithfulness, refusal_correctness,faithfulness") == [
        JudgeName.FAITHFULNESS,
        JudgeName.REFUSAL_CORRECTNESS,
    ]
    with pytest.raises(UsageError, match="unknown judge 'vibes'"):
        parse_judges("faithfulness,vibes")


# --- end to end over the dump -------------------------------------------------------------------


def test_an_abort_keeps_the_records_already_judged_in_a_private_partial(
    answers: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_model(monkeypatch, ScriptedChatModel(responder=_by_schema))
    run_judges = judge_cli.run_judges

    async def interrupted(records: Any, judges: Any, sink: Any = None) -> Any:
        await run_judges(records[:2], judges, sink)
        raise RuntimeError("boom")

    monkeypatch.setattr(judge_cli, "run_judges", interrupted)
    report_path = tmp_path / "report.json"

    code = judge_cli.main(["--answers", str(answers), "--report", str(report_path)])

    partial = tmp_path / "report.json.partial"
    assert code == EXIT_ABORTED
    assert not report_path.exists()
    assert stat.S_IMODE(partial.stat().st_mode) == 0o600
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["id"] for r in rows) == ["ok", "refused"]
    assert set(rows[0]["judgments"]) == {j.value for j in JudgeName}
    assert "report.json.partial" in capsys.readouterr().err


def test_main_judges_the_dump_and_writes_the_report(
    answers: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ledger: FakeLedger,
) -> None:
    _patch_model(monkeypatch, ScriptedChatModel(responder=_by_schema))
    report_path = tmp_path / "report.json"

    code = judge_cli.main(["--answers", str(answers), "--report", str(report_path)])

    assert code == EXIT_OK
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    assert not (tmp_path / "report.json.partial").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["meta"]["records"] == 4
    assert report["meta"]["judge_model"] == "default (openai:analyzer)"
    assert report["meta"]["judges"] == [j.value for j in JudgeName]
    assert {r.purpose for r in ledger.rows} == {Purpose.JUDGE}
    assert [r.model for r in ledger.rows] == list(report["meta"]["usage_by_model"])
    assert report["meta"]["answers_sha256"]
    assert report["meta"]["judge_version"] == prompts.JUDGE_VERSION
    assert len(report["meta"]["judge_prompts_sha256"]) == 64

    by_id = {r["id"]: r["judgments"] for r in report["records"]}
    # ok: 1 of 2 claims unsupported -> faithfulness fails; the rest pass
    assert by_id["ok"]["faithfulness"]["verdict"] is False
    assert by_id["ok"]["citation_support"]["verdict"] is True
    assert by_id["ok"]["answer_relevance"]["verdict"] is True
    assert by_id["ok"]["refusal_correctness"]["verdict"] is True
    # refused: nothing to ground, nothing relevant; the refusal is what was expected
    assert by_id["refused"]["faithfulness"]["status"] == "not_applicable"
    assert by_id["refused"]["answer_relevance"]["status"] == "not_applicable"
    assert by_id["refused"]["refusal_correctness"]["verdict"] is True
    # nocite: no citations, no expected scope
    assert by_id["nocite"]["citation_support"]["status"] == "not_applicable"
    assert by_id["nocite"]["refusal_correctness"]["status"] == "not_applicable"
    # scope-out but expected in: a blanket refusal of an answerable question
    assert by_id["scope-out"]["refusal_correctness"]["verdict"] is False

    faith = report["aggregate"]["faithfulness"]
    assert (faith["decided"], faith["passed"], faith["not_applicable"]) == (2, 0, 2)
    assert faith["pass_rate_interval"]["upper"] > 0
    # the scope-out refusal of an expected-in question is relevance False (in_scope_refused)
    assert by_id["scope-out"]["answer_relevance"]["verdict"] is False
    assert by_id["scope-out"]["answer_relevance"]["code"] == "in_scope_refused"
    assert report["aggregate"]["answer_relevance"]["pass_rate"] == pytest.approx(2 / 3)
    assert report["usage"]["calls"] == faith["usage"]["calls"] + sum(
        report["aggregate"][j]["usage"]["calls"]
        for j in ("citation_support", "answer_relevance", "refusal_correctness")
    )
    assert report["usage"]["total_tokens"] > 0
    out = capsys.readouterr().out
    assert "pass rates" in out
    assert "tokens:" in out


def test_a_judge_subset_only_runs_those(
    answers: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedChatModel(responder=_by_schema)
    _patch_model(monkeypatch, model)
    report_path = tmp_path / "r.json"

    code = judge_cli.main(
        ["--answers", str(answers), "--judges", "refusal_correctness", "--report", str(report_path)]
    )

    assert code == EXIT_OK
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert list(report["aggregate"]) == ["refusal_correctness"]
    assert model.structured_seen == []  # the deterministic judge never calls the model
    assert report["usage"]["calls"] == 0


def test_provider_errors_are_recorded_per_judgment_and_never_abort(
    answers: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model(monkeypatch, RaisingModel(error=rate_limit_error()))
    report_path = tmp_path / "r.json"

    code = judge_cli.main(["--answers", str(answers), "--report", str(report_path)])

    assert code == EXIT_PARTIAL
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ok = next(r for r in report["records"] if r["id"] == "ok")["judgments"]
    assert ok["faithfulness"]["status"] == "error"
    assert ok["faithfulness"]["rationale"] == "RateLimitError (status 429)"
    assert ok["refusal_correctness"]["verdict"] is True  # deterministic judge unaffected
    assert report["aggregate"]["faithfulness"]["errors"] == 3


def test_the_profile_flag_overrides_the_setting(
    answers: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_model(monkeypatch, ScriptedChatModel(responder=_by_schema))

    code = judge_cli.main(["--answers", str(answers), "--profile", "ghost"])

    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "judge profile 'ghost'" in err
    assert "Unknown LLM profile" in err


def test_the_profile_comes_from_eval_judge_profile_by_default(
    answers: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(judge_cli, "build_chat_models", lambda _s: chat_models_for())
    monkeypatch.setattr(
        judge_cli, "get_settings", lambda: make_settings(eval={"judge_profile": "from-settings"})
    )

    assert judge_cli.main(["--answers", str(answers)]) == EXIT_USAGE
    assert "judge profile 'from-settings'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--concurrency", "0"], "--concurrency"),
        (["--concurrency", "99"], "--concurrency"),
        (["--judges", "vibes"], "unknown judge"),
        (["--report", "{dir}"], "is a directory"),
        (["--report", "{dir}/gone/r.json"], "does not exist"),
    ],
)
def test_bad_flags_fail_before_any_call(
    answers: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    message: str,
) -> None:
    model = ScriptedChatModel(responder=_by_schema)
    _patch_model(monkeypatch, model)

    code = judge_cli.main(["--answers", str(answers), *[f.format(dir=tmp_path) for f in flags]])

    assert code == EXIT_USAGE
    assert message in capsys.readouterr().err
    assert model.structured_seen == []


def test_an_unexpected_failure_aborts_with_exit_3(
    answers: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom() -> Any:
        raise RuntimeError("no settings")

    monkeypatch.setattr(judge_cli, "get_settings", boom)
    assert judge_cli.main(["--answers", str(answers)]) == EXIT_ABORTED
    assert "RuntimeError" in capsys.readouterr().err


# --- review fixes -------------------------------------------------------------------------------


def test_there_is_no_default_judge_profile(
    answers: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = ScriptedChatModel(responder=_by_schema)
    monkeypatch.setattr(judge_cli, "build_chat_models", lambda _s: chat_models_for(model))
    monkeypatch.setattr(judge_cli, "get_settings", make_settings)  # EVAL_JUDGE_PROFILE unset

    assert judge_cli.main(["--answers", str(answers)]) == EXIT_USAGE
    assert "no judge profile" in capsys.readouterr().err
    assert model.structured_seen == []


def test_a_judge_that_wrote_the_answers_is_refused_unless_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # the judge resolves to "default (openai:analyzer)"; the answers were written by openai:analyzer
    record = replace(make_record(id="self"), models={"synthesis": "other (openai:analyzer)"})
    path = _dump(tmp_path / "self.jsonl", record)
    model = ScriptedChatModel(responder=_by_schema)
    _patch_model(monkeypatch, model)

    assert judge_cli.main(["--answers", str(path)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "openai:analyzer also wrote the answers" in err
    assert "--allow-self-judge" in err
    assert model.structured_seen == []

    assert judge_cli.main(["--answers", str(path), "--allow-self-judge"]) == EXIT_OK


def test_a_different_judge_model_is_not_self_judging(
    answers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model(monkeypatch, ScriptedChatModel(responder=_by_schema))
    assert judge_cli.main(["--answers", str(answers)]) == EXIT_OK


@pytest.mark.parametrize(
    ("label", "expected"),
    [("default (openai:m)", "openai:m"), ("p (anthropic:claude (x))", "anthropic:claude (x)"),
     ("nothing", None)],
)  # fmt: skip
def test_provider_model_of_a_label(label: str, expected: str | None) -> None:
    assert judge_cli.provider_model(label) == expected


def test_a_judge_that_cannot_do_structured_output_gets_a_judge_specific_hint(
    answers: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = ScriptedChatModel(profile={"structured_output": False})
    _patch_model(monkeypatch, model)

    assert judge_cli.main(["--answers", str(answers)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "Choose another judge profile (EVAL_JUDGE_PROFILE or --profile)" in err
    assert "Bind another LLM profile" not in err
    assert "agent spec" not in err


def test_failed_agent_runs_are_run_failed_not_judge_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _dump(
        tmp_path / "answers.jsonl",
        make_record(id="ok"),
        replace(make_record("", id="boom", error="RateLimitError"), sources=()),
    )
    _patch_model(monkeypatch, ScriptedChatModel(responder=_by_schema))
    report_path = tmp_path / "r.json"

    code = judge_cli.main(["--answers", str(path), "--report", str(report_path)])

    assert code == EXIT_OK  # a failed run in the dump is not the judge's failure
    report = json.loads(report_path.read_text(encoding="utf-8"))
    boom = next(r for r in report["records"] if r["id"] == "boom")["judgments"]
    assert {j["status"] for j in boom.values()} == {"run_failed"}
    for stats in report["aggregate"].values():
        assert (stats["run_failed"], stats["errors"]) == (1, 0)


def test_the_report_binds_each_record_to_its_answer_hash(
    answers: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model(monkeypatch, ScriptedChatModel(responder=_by_schema))
    report_path = tmp_path / "r.json"
    judge_cli.main(["--answers", str(answers), "--report", str(report_path)])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {r.id: r.answer_sha256 for r in load_dump(answers)}
    assert {r["id"]: r["answer_sha256"] for r in report["records"]} == expected


def test_a_model_refusal_is_a_counted_not_applicable_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def respond(schema: object, messages: list[object]) -> object:
        name = getattr(schema, "__name__", "")
        if name == "FaithfulnessOutput":
            return claims(is_refusal=True)
        return _by_schema(schema, messages)

    path = _dump(tmp_path / "a.jsonl", make_record("Eso queda fuera de mi alcance.", id="r"))
    _patch_model(monkeypatch, ScriptedChatModel(responder=respond))
    report_path = tmp_path / "r.json"
    judge_cli.main(["--answers", str(path), "--report", str(report_path)])

    stats = json.loads(report_path.read_text(encoding="utf-8"))["aggregate"]["faithfulness"]
    assert stats["not_applicable"] == 1
    assert stats["reasons"] == {"model_refusal": 1}


def test_a_shared_provider_family_warns_but_does_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = replace(make_record(id="fam"), models={"synthesis": "gen (openai:gpt-other)"})
    path = _dump(tmp_path / "fam.jsonl", record)
    _patch_model(monkeypatch, ScriptedChatModel(responder=_by_schema))  # judge: openai:analyzer

    assert judge_cli.main(["--answers", str(path)]) == EXIT_OK
    err = capsys.readouterr().err
    assert "shares its provider family" in err
    assert "openai:gpt-other" in err


def test_a_different_family_gives_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = replace(make_record(id="fam"), models={"synthesis": "gen (anthropic:claude)"})
    path = _dump(tmp_path / "fam.jsonl", record)
    _patch_model(monkeypatch, ScriptedChatModel(responder=_by_schema))

    assert judge_cli.main(["--answers", str(path)]) == EXIT_OK
    assert "provider family" not in capsys.readouterr().err
