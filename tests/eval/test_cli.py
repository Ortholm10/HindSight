import json

import pytest

from hindsight_cli.main import main


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


def test_audit_is_not_built_yet_and_says_so():
    with pytest.raises(NotImplementedError, match="measuring instrument"):
        main(["audit", "strategy.py"])
