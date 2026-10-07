"""Execute patched official Python methods on CPU mocks and export captured routes."""
import argparse
import ast
from functools import partial
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
import uuid

from ..adapters.ascend_v41_source import execution
from ..distributed.decode import Worker, DecodeRow
from ..distributed.replay import InputBinding
from ..distributed.storage import create_run, WorkerWriter, export_run, export_counts
from ..errors import IntegrityError
from ..io import atomic_json


def method(overlay, path, cls_name, name, namespace):
    tree = ast.parse((overlay / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, "exec"), namespace)
    return namespace[name]


def run_validation(overlay, output):
    import torch
    overlay, output = Path(overlay), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "source-validation.json"
    atomic_json(report_path, {"state": "running", "npu_verified": False})
    try:
        manifest = json.loads((overlay / "source-hooks.json").read_text())
        trusted_path = Path(__file__).resolve().parents[3] / "patches/ascend/deepseek-v41-cpu/source-hooks.json"
        trusted = json.loads(trusted_path.read_text())
        if manifest != trusted:
            raise IntegrityError("not the pinned CPU source-hook copy")
        for path, expected in manifest["patched_sha256"].items():
            if hashlib.sha256((overlay / path).read_bytes()).hexdigest() != expected:
                raise IntegrityError("modified source-hook copy")
        worker = Worker("source-fixture", str(uuid.uuid4()), "D0", "worker0", 0, 0, 0, 0)
        rows = [DecodeRow("engine0", "forward1", 1, req, req, i, pos, token, "target_decode")
                for i, (req, pos, token) in enumerate([("A", 7, 41), ("B", 9, 51)])]
        bindings = [InputBinding(i, worker, row) for i, row in enumerate(rows)]
        selector = method(overlay, "vllm_ascend/ops/fused_moe/routed_experts.py",
                          "AscendRoutedExperts", "_select_experts", {
                              "torch": torch, "get_moe_num_logical_experts": lambda *a, **kw: 4,
                              "get_ascend_config": lambda: NS(enable_force_eplb=False)})
        consumer_reference = []
        def consumer(**kwargs):
            value = kwargs["fused_experts_input"]
            consumer_reference.append({"layer_name": value.layer.layer_name,
                                       "ids": value.topk_ids.clone().tolist(),
                                       "weights": value.topk_weights.float().clone().tolist()})
            return value.hidden_states
        apply = method(overlay, "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
                       "AscendW8A8DynamicFusedMoEMethod", "apply", {
                           "torch": torch, "Any": Any,
                           "_EXTRA_CTX": NS(moe_comm_method=NS(fused_experts=consumer)),
                           "build_fused_experts_input": lambda **kwargs: NS(**kwargs)})
        mode = NS(NONE=NS(name="NONE"), FULL=NS(name="FULL"))
        forward_context = NS(cudagraph_runtime_mode=mode.NONE)
        forward = method(overlay, "vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", "_model_forward", {
            "torch": torch, "Any": Any, "IntermediateTensors": object, "partial": partial,
            "get_forward_context": lambda: forward_context, "CUDAGraphMode": mode})
        logits = {
            20: torch.tensor([[0.1, 0.8, 0.2, 0.9], [0.9, 0.1, 0.8, 0.2]]),
            21: torch.tensor([[0.1, 0.9, 0.8, 0.2], [0.8, 0.1, 0.2, 0.9]])}
        layer_specs = {f"model.layers.{i}.mlp.experts": (i, 4) for i in logits}
        def model(**kwargs):
            if kwargs["input_ids"].tolist() != [41, 51] or kwargs["positions"].tolist() != [7, 9]:
                raise IntegrityError("actual model inputs differ from independent fixture")
            for layer_id, scores in logits.items():
                router = NS(_select_experts=lambda router_logits, **kw: torch.topk(router_logits, 2, dim=-1))
                layer = NS(router=router, layer_name=f"model.layers.{layer_id}.mlp.experts",
                           log2phy=torch.tensor([1, 3, 0, 2]), n_shared_experts=1,
                           moe_config=NS(num_experts=4, is_sequence_parallel=False),
                           global_redundant_expert_num=0, mix_placement=True,
                           ascend_expert_map=None, ascend_mc2_mask=None,
                           apply_router_weight_on_input=False, ascend_pertoken_scale=None)
                weights, ids = selector(layer, torch.zeros(2, 2), scores, False)
                apply(NS(quant_type="CPU-mock", use_expert_weight_list=False), layer,
                      torch.zeros(2, 2), weights, ids, None, None)
            return "model-executed"
        runner = NS(model=model, enable_enpu=False, _update_full_graph_params_if_needed=lambda *a: None)
        with execution(worker, bindings, "target", layer_specs) as captured:
            result = forward(runner, 2, torch.tensor([41, 51]), torch.tensor([7, 9]))
        expected = {("A", 20): (3, 1), ("B", 20): (0, 2),
                    ("A", 21): (1, 2), ("B", 21): (3, 0)}
        actual = {(r.row.global_request, r.layer): r.expert_ids for r in captured.routes}
        if result != "model-executed" or actual != expected or captured.consumer_inputs != consumer_reference:
            raise IntegrityError("source-hook capture differs from independent consumer")
        expected_consumers = [[[2, 3, 4], [1, 0, 4]], [[3, 0, 4], [2, 1, 4]]]
        if [v["ids"] for v in consumer_reference] != expected_consumers:
            raise IntegrityError("consumer mapping/shared expert differs from oracle")
        run_dir = create_run(output / "runs" / worker.run, [worker], {"target": [20, 21]})
        writer = WorkerWriter(run_dir, worker, shard_records=2)
        writer.declare_execution("target", rows)
        for route in captured.routes:
            writer.append(route)
        writer.close()
        export_run(run_dir, output / "details.json")
        export_counts(run_dir, output / "expert-counts.csv")
        atomic_json(output / "consumer-reference.json", {"inputs": consumer_reference})
        report = {"state": "passed", "validation_level": "cpu_source_hooks", "npu_verified": False,
                  "model_loaded": False, "full_service_run": False, "base_commits": manifest["base_commits"],
                  "run_dir": str(run_dir), "logical_routes": 4, "expert_choices": 8,
                  "comparison": "passed", "patched_methods_executed": 3}
        atomic_json(report_path, report)
        return report
    except Exception as exc:
        atomic_json(report_path, {"state": "failed", "npu_verified": False, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.overlay, args.output), indent=2))


if __name__ == "__main__":
    main()
