import json
import os

import pytest

from fedlearn.cli import main
from fedlearn.report import write_report
from fedlearn.simulate import compare, run_federated


def test_cli_simulate(capsys):
    assert main(["simulate", "--rounds", "2", "--clients", "4",
                 "--fraction", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "test accuracy" in out


def test_cli_compare_and_report(tmp_path, capsys):
    assert main(["compare", "--rounds", "2", "--clients", "4",
                 "--fraction", "1.0", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "centralized" in out
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "summary.json").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "final_test_accuracy" in summary
    assert len(summary["history"]) == 2


def test_cli_baselines(capsys):
    assert main(["centralized", "--epochs", "2"]) == 0
    assert "centralized test accuracy" in capsys.readouterr().out
    assert main(["local-only", "--epochs", "2", "--clients", "4"]) == 0
    assert "local-only" in capsys.readouterr().out


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        main([])


def test_write_report_contents(tmp_path):
    result = compare(seed=0, rounds=2, n_clients=4, fraction=1.0)
    md, js = write_report(result, str(tmp_path))
    text = open(md).read()
    assert "Federated Learning Simulation" in text
    assert "Baselines" in text
    assert "Accuracy by round" in text
    summary = json.loads(open(js).read())
    assert summary["baselines"]["centralized"] > 0
