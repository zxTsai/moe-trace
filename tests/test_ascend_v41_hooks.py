import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from moe_trace.adapters.ascend_v41_source import execution, verify_forward, logical_routes, consumer_routes
from moe_trace.distributed.decode import Worker, DecodeRow
from moe_trace.distributed.replay import InputBinding
from moe_trace.errors import IntegrityError, ConfigError
from moe_trace.validation.ascend_v41_source_cpu import run_validation

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def bindings():
    worker = Worker("deployment", "run", "D0", "w0", 0, 0, 0, 0)
    row = DecodeRow("engine", "exec", 0, "A", "A", 0, 7, 41, "target_decode")
    return worker, [InputBinding(0, worker, row)]


def test_disabled_hooks_do_nothing(monkeypatch):
    monkeypatch.delenv("MOE_TRACE_V41_SOURCE_HOOKS", raising=False)
    verify_forward(None, None)
    logical_routes(None, None, None)
    consumer_routes(None, None, None)


def test_enabled_without_context_fails(monkeypatch):
    monkeypatch.setenv("MOE_TRACE_V41_SOURCE_HOOKS", "1")
    with pytest.raises(ConfigError, match="harness"):
        verify_forward(None, None)


@pytest.mark.parametrize("mode,ids", [(NS(name="FULL"), torch.tensor([41])),
                                      (None, torch.tensor([41.5])),
                                      (None, torch.tensor([42]))])
def test_invalid_forward_fails(mode, ids):
    worker, rows = bindings()
    with pytest.raises((IntegrityError, ConfigError)):
        with execution(worker, rows, "target", {"layer": (20, 4)}):
            verify_forward(ids, torch.tensor([7]), mode)


def test_missing_consumer_is_not_success():
    worker, rows = bindings()
    with pytest.raises(IntegrityError, match="consumer"):
        with execution(worker, rows, "target", {"layer": (20, 4)}):
            verify_forward(torch.tensor([41]), torch.tensor([7]))
            logical_routes(NS(layer_name="layer"), torch.tensor([[3, 1]]), torch.tensor([[.75, .25]]))


def test_context_exception_does_not_leak():
    worker, rows = bindings()
    with pytest.raises(RuntimeError):
        with execution(worker, rows, "target", {"layer": (20, 4)}):
            raise RuntimeError("forward failed")
    verify_forward(None, None)


def test_patched_source_methods_export_routes(tmp_path):
    vllm = os.environ.get("MOE_V41_VLLM_SOURCE")
    ascend = os.environ.get("MOE_V41_ASCEND_SOURCE")
    if not vllm or not ascend:
        pytest.skip("fixed source repositories not provided")
    spec = importlib.util.spec_from_file_location("stage_v41", ROOT / "scripts/tools/stage_ascend_v41_cpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = (Path(ascend) / "vllm_ascend/ops/fused_moe/routed_experts.py").read_bytes()
    overlay = module.stage(vllm, ascend, tmp_path / "overlays")
    assert (Path(ascend) / "vllm_ascend/ops/fused_moe/routed_experts.py").read_bytes() == before
    report = run_validation(overlay, tmp_path / "validation")
    assert report["patched_methods_executed"] == 3
    assert report["logical_routes"] == 4 and report["expert_choices"] == 8
    assert report["full_service_run"] is False and report["npu_verified"] is False
    routes = json.loads((tmp_path / "validation/details.json").read_text())["routes"]
    assert routes[0]["expert_ids"] in ([3, 1], [1, 2])
    path = overlay / "vllm_ascend/ops/fused_moe/routed_experts.py"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(IntegrityError, match="modified"):
        run_validation(overlay, tmp_path / "validation")
    assert json.loads((tmp_path / "validation/source-validation.json").read_text())["state"] == "failed"
