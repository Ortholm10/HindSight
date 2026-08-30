import json

from hindsight_cli.main import MODES, build_parser, main
from hindsight_core.pipeline import audit as pipeline_audit


def test_eval_prints_a_table(capsys):
    assert main(["eval", "--suite", "clean", "--detector", "null"]) == 0
    out = capsys.readouterr().out
    assert "false positives     0/8" in out


def test_eval_writes_json(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert (
        main(
            [
                "eval",
                "--suite",
                "injected",
                "--detector",
                "everything",
                "--json",
                "--out",
                str(out),
                "--no-metrics",
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text("utf-8"))
    assert payload["detected"] == 12
    assert payload["localisation_precision"] < 0.10


def test_unknown_case_exits_nonzero_with_a_message(capsys):
    assert main(["eval", "--case", "nope", "--no-metrics"]) == 2
    assert "no case named" in capsys.readouterr().err


def test_audit_on_a_missing_file_exits_nonzero_with_a_message(capsys):
    assert main(["audit", "nowhere/strategy.py"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_audit_keeps_pipeline_addressable_by_name():
    """--mode pipeline must keep working once the agent loop lands: the
    pipeline is the documented baseline the agent is measured against."""
    args = build_parser().parse_args(["audit", "strategy.py", "--mode", "pipeline"])
    assert args.mode == "pipeline"
    assert MODES["pipeline"] is pipeline_audit


def test_audit_offers_the_agent_mode():
    args = build_parser().parse_args(["audit", "s.py", "--mode", "agent"])
    assert args.mode == "agent"
    assert "agent" in MODES


def test_the_agent_detectors_are_registered():
    from eval.detectors import DETECTORS

    assert "agent" in DETECTORS
    assert "agent-reported" in DETECTORS


def test_repeat_records_every_pass_and_the_spread(tmp_path, monkeypatch):
    """Three runs of a non-deterministic agent are three data points, not one
    result. Reporting a single pass, or a mean, hides exactly what the repeat
    was for."""
    from eval.harness import CaseResult, SuiteResult

    seen: list[int] = []

    def fake_suite(detector, suite="all", case_id=None, with_metrics=True, **kwargs):
        seen.append(len(seen) + 1)
        return SuiteResult(
            suite="all",
            results=[
                CaseResult(
                    case_id="l01_future_index",
                    kind="injected",
                    leak_type="L01",
                    candidates=len(seen),
                    detected=True,
                    localised=True,
                    type_correct=True,
                    on_ground_truth=1,
                    sharpe_delta=None,
                    known_limitations=[],
                )
            ],
        )

    monkeypatch.setattr("hindsight_cli.main.run_suite", fake_suite)
    out = tmp_path / "agent.json"

    main(["eval", "--detector", "agent", "--repeat", "3", "--json", "--out", str(out)])

    payload = json.loads(out.read_text("utf-8"))
    assert len(payload["passes"]) == 3
    spread = payload["spread"]["candidates_on_injected"]
    assert spread["values"] == [1, 2, 3]
    assert (spread["min"], spread["max"]) == (1, 3)


def test_repeat_prints_the_spread_table(capsys, monkeypatch):
    from eval.harness import SuiteResult

    monkeypatch.setattr(
        "hindsight_cli.main.run_suite",
        lambda *a, **k: SuiteResult(suite="all", results=[]),
    )
    main(["eval", "--detector", "null", "--repeat", "2"])
    assert "spread over 2 run(s)" in capsys.readouterr().out
