"""Validate documented syntax and the CPU counterpart of the export workflow."""
import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("document", ["README.md", "docs/guides/ascend-quickstart.md"])
def test_quickstart_code_block_syntax(document):
    content = (ROOT / document).read_text()
    for language, block in re.findall(r"```(bash|python)\n(.*?)```", content, re.S):
        if language == "python":
            ast.parse(block)
        else:
            # Parse only: do NOT launch Docker, install dependencies or start a service.
            result = subprocess.run(["bash", "-n"], input=block, text=True,
                                    capture_output=True, timeout=10)
            assert result.returncode == 0, result.stderr


def test_cpu_quickstart_validate_export_and_step_counts(tmp_path):
    pytest.importorskip("torch")
    from moe_trace.validation.ascend_cpu import run_validation
    report = run_validation(tmp_path)
    prefix = [sys.executable, "-m", "moe_trace.cli"]
    run = report["run_dir"]
    result = subprocess.run([*prefix, "validate", run], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"complete": True, "issues": []}
    details = tmp_path / "step.json"
    counts = tmp_path / "step-counts.csv"
    for command in ([*prefix, "details", run, str(details), "--format", "json", "--step-id", "3"],
                    [*prefix, "counts", run, str(counts), "--step-id", "3"]):
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    rows = json.loads(details.read_text())["records"]
    assert len(rows) == 6
    expected = {("A", 0): [3, 1], ("A", 1): [1, 2],
                ("C", 0): [1, 2], ("C", 1): [2, 0],
                ("B", 0): [2, 0], ("B", 1): [3, 1]}
    assert {(r["request_id"], r["layer_id"]): r["expert_ids"] for r in rows} == expected
    with counts.open(newline="") as stream:
        distribution = list(csv.DictReader(stream))
    assert sum(int(r["count"]) for r in distribution) == 12
    assert any(int(r["count"]) == 0 for r in distribution)
