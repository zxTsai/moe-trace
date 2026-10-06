"""Pinned-source checks; optional hardware-free integration uses the staged overlay."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from moe_trace.setup import ascend as staging
from moe_trace.setup.ascend import BASES, EDITS, check_sources, transform

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_covers_all_hooks():
    manifest = json.loads((ROOT / "patches/ascend/0.23.0/base_files.json").read_text())
    assert manifest["base_commits"] == BASES
    assert set(manifest["sha256"]) == set(manifest["patched_sha256"]) == set(EDITS)
    assert all(len(v) == 64 for v in manifest["sha256"].values())


@pytest.mark.parametrize("entry", ["scripts/tools/stage_ascend_023.py", "scripts/dev/build_ascend_023_patch.py"])
def test_stdlib_tools(entry):
    result = subprocess.run([sys.executable, "-S", str(ROOT / entry), "--help"], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_invalid_source_never_silently_patched(tmp_path):
    roots = {"vllm": tmp_path / "vllm", "vllm_ascend": tmp_path / "vllm_ascend"}
    for name in EDITS:
        package, rel = name.split("/", 1)
        file = roots[package] / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# incompatible source\n")
    manifest = json.loads((ROOT / "patches/ascend/0.23.0/base_files.json").read_text())
    with pytest.raises(ValueError, match="hash mismatch"):
        check_sources(roots["vllm"], roots["vllm_ascend"], manifest)


def test_missing_patch_anchor_rejected():
    with pytest.raises(ValueError, match="source anchor"):
        transform("vllm_ascend/ops/fused_moe/fused_moe.py", "pass\n")


def test_overlay_cannot_be_nested_inside_original_package(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "check_sources", lambda *a: {})
    with pytest.raises(ValueError, match="inside a source"):
        staging.stage(tmp_path / "vllm", tmp_path / "ascend", {}, tmp_path / "vllm/overlays")


def test_pinned_source_cpu_integration(tmp_path):
    overlay = os.environ.get("MOE_ASCEND_SOURCE_OVERLAY")
    if not overlay:
        pytest.skip("set MOE_ASCEND_SOURCE_OVERLAY to a hash-checked 0.23.0 overlay")
    pytest.importorskip("torch")
    from moe_trace.validation.ascend_cpu import run_validation
    report = run_validation(tmp_path, overlay=overlay)
    assert report["source_hooks_executed"]
    assert report["independent_comparison"] == "passed"
    assert report["route_records"] == 30
    assert report["storage_complete"]
