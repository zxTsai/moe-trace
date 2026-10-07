"""Source inventory and CPU execution of the actual pinned expert selection method."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/check_ascend_v41_sources.py"


def test_inventory_help_without_dependencies():
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def sources():
    vllm_root = os.environ.get("MOE_V41_VLLM_SOURCE")
    ascend_root = os.environ.get("MOE_V41_ASCEND_SOURCE")
    if not vllm_root or not ascend_root:
        pytest.skip("fixed V4.1 source repositories not provided")
    spec = importlib.util.spec_from_file_location("source_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock = json.loads((ROOT / "docs/design/ascend-v41-source-lock.json").read_text())
    module.check_sources(vllm_root, ascend_root, lock)
    return Path(vllm_root), Path(ascend_root), module, lock


def test_pinned_inventory_and_hash_mismatch():
    vllm_root, ascend_root, module, lock = sources()
    assert module.check_sources(vllm_root, ascend_root, lock)["files"] == 10
    lock["sha256"]["vllm/v1/spec_decode/metadata.py"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        module.check_sources(vllm_root, ascend_root, lock)


def method(path, class_name, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    func = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    func.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])), str(path), "exec")
    exec(code, namespace)
    return namespace[name]


def test_actual_selector_mapping_shared_expert_and_quantized_consumer():
    _, ascend, _, _ = sources()
    torch = pytest.importorskip("torch")
    logical_ids = torch.tensor([[3, 1]], dtype=torch.int32)
    logical_weights = torch.tensor([[0.75, 0.25]])
    router = NS(_select_experts=lambda **kwargs: (logical_weights.clone(), logical_ids.clone()))
    owner = NS(router=router, log2phy=torch.tensor([1, 3, 0, 2]), n_shared_experts=1,
               moe_config=NS(num_experts=4), global_redundant_expert_num=0, mix_placement=True)
    selector = method(ascend / "vllm_ascend/ops/fused_moe/routed_experts.py",
                      "AscendRoutedExperts", "_select_experts", {
                          "torch": torch, "get_moe_num_logical_experts": lambda *a, **kw: 4,
                          "get_ascend_config": lambda: NS(enable_force_eplb=False),
                      })
    weights, ids = selector(owner, torch.zeros(1, 2), torch.zeros(1, 4), False)
    # Independent expected IDs: logical [3,1] -> physical [2,3], then shared [4].
    assert logical_ids.tolist() == [[3, 1]]
    assert ids.tolist() == [[2, 3, 4]]
    assert weights.tolist() == [[0.75, 0.25, 1.0]]
    consumed = []
    def consumer(**kwargs):
        value = kwargs["fused_experts_input"]
        consumed.append((value.topk_ids.clone(), value.topk_weights.clone()))
        return "consumer-called"
    namespace = {"torch": torch, "Any": Any,
                 "_EXTRA_CTX": NS(moe_comm_method=NS(fused_experts=consumer)),
                 "build_fused_experts_input": lambda **kwargs: NS(**kwargs)}
    apply = method(ascend / "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
                   "AscendW8A8DynamicFusedMoEMethod", "apply", namespace)
    layer = NS(ascend_expert_map=None, global_redundant_expert_num=0, ascend_mc2_mask=None,
               apply_router_weight_on_input=False, ascend_pertoken_scale=None)
    result = apply(NS(quant_type="fixture", use_expert_weight_list=False), layer,
                   torch.zeros(1, 2), weights, ids, None, None)
    assert result == "consumer-called"
    assert consumed[0][0].tolist() == [[2, 3, 4]]
    assert consumed[0][1].tolist() == [[0.75, 0.25, 1.0]]
