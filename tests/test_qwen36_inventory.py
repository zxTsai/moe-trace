import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def tool():
    spec = importlib.util.spec_from_file_location("qwen_inventory", ROOT / "scripts/tools/inspect_qwen36_ascend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_absent_sources_never_claims_runtime_support(tmp_path, monkeypatch, capsys):
    module = tool()
    monkeypatch.setattr(module, "package_info", lambda name: {"version": None, "package_path": None})
    output = tmp_path / "inventory.json"
    assert module.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["ascend_sources"] is None
    assert report["runtime_supported"] is False and report["npu_verified"] is False
    original = output.read_bytes()
    assert module.main(["--output", str(output)]) == 1
    assert output.read_bytes() == original


def test_source_missing_and_mutated_files_are_reported(tmp_path):
    module = tool()
    first = module.inspect_sources(tmp_path)
    assert first["all_reference_files_match"] is False
    path = tmp_path / "worker/worker.py"
    path.parent.mkdir()
    path.write_text("# not the pinned source\n")
    second = module.inspect_sources(tmp_path)
    assert second["files"]["worker/worker.py"]["sha256"] is not None
    assert second["files"]["worker/worker.py"]["matches_reference"] is False


def test_model_config_nested_text_read_without_model_load(tmp_path, monkeypatch):
    module = tool()
    monkeypatch.setattr(module, "package_info", lambda name: {"version": None, "package_path": None})
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {"num_experts": 256, "num_experts_per_tok": 8}}))
    output = tmp_path / "model-inventory.json"
    assert module.main(["--model", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["model"]["num_experts"] == 256


def test_package_detection_uses_import_spec_not_site_packages_metadata(tmp_path, monkeypatch):
    module = tool()
    editable = tmp_path / "vllm-workspace/vllm-ascend/vllm_ascend"
    editable.mkdir(parents=True)
    monkeypatch.setattr(module.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(editable / "__init__.py")))
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.23.0.post1")
    result = module.package_info("vllm_ascend")
    assert result == {"version": "0.23.0.post1", "package_path": str(editable)}
