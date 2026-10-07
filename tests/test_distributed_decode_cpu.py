import json
import subprocess
import sys

import pytest

from moe_trace.validation.distributed_decode_cpu import run_validation


def test_independent_fixture_and_cli(tmp_path):
    report = run_validation(tmp_path / "validation")
    assert report["comparison"] == "passed"
    assert report["observed_routes"] == 20
    assert report["logical_routes"] == 10
    assert report["expert_choices"] == 20
    assert report["npu_verified"] is False
    result = subprocess.run([sys.executable, "-m", "moe_trace.distributed.cli", "validate",
                             report["run_dir"]], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["contract_complete"] is True
    result = subprocess.run([sys.executable, "-m", "moe_trace.distributed.cli", "merge",
                             report["run_dir"], str(tmp_path / "merged.json")],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    routes = json.loads((tmp_path / "merged.json").read_text())["routes"]
    assert len(routes) == 10
    assert sum(len(r["expert_ids"]) for r in routes) == 20
    assert sum(r["candidate_accepted"] is False for r in routes) == 2
    assert any(r["row"]["input_role"] == "mask" and r["row"]["token_id"] is None
               for r in routes)


def test_cli_reports_invalid_input_without_success(tmp_path):
    result = subprocess.run([sys.executable, "-m", "moe_trace.distributed.cli", "validate",
                             str(tmp_path / "missing")], text=True, capture_output=True, timeout=30)
    assert result.returncode == 1
    assert json.loads(result.stderr)["contract_complete"] is False


def test_failed_fixture_invalidates_previous_success(tmp_path, monkeypatch):
    from moe_trace.validation import distributed_decode_cpu as validation
    run_validation(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("fixture failed")
    monkeypatch.setattr(validation, "collect_run", fail)
    with pytest.raises(OSError):
        run_validation(tmp_path)
    assert json.loads((tmp_path / "cpu-validation.json").read_text())["state"] == "failed"
