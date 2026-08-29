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
