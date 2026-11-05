"""Read-only preflight must work without Git, vLLM or NPU imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/inspect_ascend_async_graph.py"


def fixture(tmp_path):
    spec = importlib.util.spec_from_file_location("async_inventory", SCRIPT)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    reference = json.loads(tool.REFERENCE.read_text())
    roots = {name: tmp_path / name for name in ("vllm", "vllm_ascend")}
    for name in reference["sha256"]:
        package, relative = name.split("/", 1)
        path = roots[package] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"# public source fixture: {name}\n".encode()
        path.write_bytes(data)
        reference["sha256"][name] = hashlib.sha256(data).hexdigest()
    tool.REFERENCE = tmp_path / "reference.json"
    tool.REFERENCE.write_text(json.dumps(reference))
    return tool, roots


def test_matching_sources_are_not_advertised_as_adapter_ready(tmp_path):
    tool, roots = fixture(tmp_path)
    report = tool.inspect_sources(roots["vllm"], roots["vllm_ascend"])
    assert report["reference_source_match"] and not report["adapter_ready"]
    assert not report["npu_verified"] and len(report["sha256"]) == 20
    assert not (roots["vllm"] / ".git").exists()


def test_missing_or_changed_sources_report_all_differences(tmp_path):
    tool, roots = fixture(tmp_path)
    (roots["vllm_ascend"] / "models/deepseek_v4.py").write_text("# modified\n")
    (roots["vllm_ascend"] / "models/deepseek_v4_mtp.py").unlink()
    report = tool.inspect_sources(roots["vllm"], roots["vllm_ascend"])
    assert not report["reference_source_match"]
    assert len(report["differences"]) == 2
    assert any(value["actual_sha256"] is None for value in report["differences"])


def test_default_workspace_fallback_without_vendor_import(tmp_path, monkeypatch):
    tool, roots = fixture(tmp_path)
    monkeypatch.setattr(tool.importlib.util, "find_spec", lambda name: None)
    path, reason = tool.locate("vllm_ascend", None, tmp_path)
    assert path == tmp_path / "vllm-ascend/vllm_ascend" and reason == "default_workspace"


def test_stdlib_cli_help_and_report_does_not_overwrite(tmp_path):
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    output = tmp_path / "report.json"
    output.write_text("existing report\n")
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--source-root", str(tmp_path),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 2 and output.read_text() == "existing report\n"
