"""Independent CPU consumer comparison, optionally executing pinned source hooks."""
from __future__ import annotations

import argparse
import ast
import copy
import json
from contextlib import nullcontext
from enum import Enum
from functools import partial
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from ..adapters import ascend_vllm_0_23_0 as adapter
from ..analysis import expert_counts, write_details_json
from ..config import TraceConfig
from ..io import atomic_json
from ..reader import TraceReader
from ..session import TraceSession
from ..setup.ascend import check_overlay


class RequestFixture(NS):
    __hash__ = object.__hash__


def owner():
    return NS(vllm_config=NS(
        model_config=NS(enforce_eager=True, enable_return_routed_experts=False, quantization=None,
                        hf_config=NS(architectures=["GraniteMoeForCausalLM"])),
        cache_config=NS(enable_prefix_caching=False), speculative_config=None,
        scheduler_config=NS(async_scheduling=False),
        parallel_config=NS(distributed_executor_backend="uni", tensor_parallel_size=1,
                           pipeline_parallel_size=1, data_parallel_size=1)))


def runner_base():
    runner = owner()
    runner.ascend_config = NS(eplb_config=NS(dynamic_eplb=False),
                             profiling_chunk_config=NS(enabled=False))
    runner.compilation_config = NS(cudagraph_mode=NS(name="NONE"))
    runner.cache_config = NS(mamba_cache_mode="none")
    runner.enable_enpu = False
    runner.maybe_get_kv_connector_output = lambda *args, **kwargs: nullcontext(None)
    runner._update_full_graph_params_if_needed = lambda *args: None
    return runner


def _function(source, class_name, name):
    tree = ast.parse(source)
    klass = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    return copy.deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name))


def _compile(function, namespace):
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<pinned-ascend-source>", "exec"), namespace)
    return namespace[function.name]


def load_source_hooks(overlay, torch, consumer):
    """Execute real patched Python methods, mocking only device/framework dependencies."""
    overlay = Path(overlay)
    check_overlay(overlay)
    moe = (overlay / "vllm_ascend/ops/fused_moe/fused_moe.py").read_text()
    runner_source = (overlay / "vllm_ascend/worker/model_runner_v1.py").read_text()

    class Comm(Enum):
        ALLGATHER = 0
        FUSED_MC2 = 1

    def selector(**kwargs):
        probabilities = torch.softmax(kwargs["router_logits"], dim=-1)
        weights, ids = torch.topk(probabilities, kwargs["top_k"], dim=-1)
        return weights / weights.sum(dim=-1, keepdim=True), ids

    apply_ns = dict(torch=torch, select_experts=selector, vllm_version_is=lambda v: v == "0.23.0",
                    get_moe_num_logical_experts=lambda layer, num_experts, **kw: num_experts,
                    get_forward_context=lambda: NS(input_ids=None),
                    _EXTRA_CTX=NS(moe_comm_type=Comm.ALLGATHER,
                                  moe_comm_method=NS(fused_experts=consumer)),
                    MoECommType=Comm, QuantType=NS(NONE="none"),
                    build_fused_experts_input=lambda **kwargs: kwargs)
    apply = _compile(_function(moe, "AscendUnquantizedFusedMoEMethod", "apply"), apply_ns)
    forward_ns = dict(torch=torch, partial=partial, CUDAGraphMode=NS(PIECEWISE="piecewise"),
                      get_forward_context=lambda: NS(flash_comm_v1_enabled=False))
    forward = _compile(_function(runner_source, "NPUModelRunner", "_model_forward"), forward_ns)
    original = _function(runner_source, "NPUModelRunner", "execute_model")
    forward_with = next(n for n in ast.walk(original) if isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Name)
        and i.context_expr.func.id == "trace_execution" for i in n.items))
    # Keep the real context managers and forward body, excluding unrelated preparation/sampling.
    wrapper = ast.parse("def execute(self, scheduler_output, input_ids, positions):\n    pass").body[0]
    wrapper.body = [*ast.parse("num_tokens_padded = len(input_ids)").body,
                    ast.ImportFrom(module="moe_trace.adapters.ascend_vllm_0_23_0",
                                  names=[ast.alias(name="trace_execution")], level=0),
                    forward_with, ast.Return(value=ast.Name(id="hidden_states", ctx=ast.Load()))]
    execute_ns = dict(record_function_or_nullcontext=lambda *a: nullcontext(),
                      set_ascend_forward_context=lambda *a, **kw: nullcontext(),
                      attn_metadata=None, num_tokens_padded=0, num_tokens_across_dp=None,
                      cudagraph_mode=None, batch_desc=None, has_encoder_input=False,
                      defer_kv_connector_finalize=False, intermediate_tensors=None,
                      inputs_embeds=None, model_kwargs={})
    execute = _compile(wrapper, execute_ns)

    scheduler = (overlay / "vllm/v1/core/sched/scheduler.py").read_text()
    scheduler_ns = dict(logger=NS(debug_once=lambda *a: None))
    add = _compile(_function(scheduler, "Scheduler", "add_request"), scheduler_ns)
    free = _compile(_function(scheduler, "Scheduler", "_free_request"), scheduler_ns)
    shutdown = _compile(_function(scheduler, "Scheduler", "shutdown"), scheduler_ns)
    return apply, forward, execute, add, free, shutdown


def _run_validation(output_root, *, overlay=None, config=None):
    import torch
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = config or TraceConfig(enabled=True, output_root=output_root / "traces",
                                   save_token_ids=True, save_expert_weights=True,
                                   shard_records=4, buffer_records=8)
    # Reject accidentally reusing a live hook session; this utility never enables NPU inference.
    if adapter._session is not None or adapter._active.get() is not None:
        raise RuntimeError("CPU validation requires an idle adapter")
    adapter._reject_unsupported(owner(), config)
    adapter.check_versions("0.23.0", "0.23.0")
    session = TraceSession(config, model={"architecture": "GraniteMoeForCausalLM",
                                         "name_or_path": "synthetic-cpu-fixture"},
                           moe_layers=[0, 1], experts_per_layer={0: 4, 1: 4}, top_k=2,
                           expert_weight_semantics="CPU fixture consumer weights after BF16 conversion; not re-normalized",
                           framework={"adapter": adapter.ADAPTER, "backend": "cpu_simulation",
                                      "validation_level": "cpu_only", "npu_verified": False,
                                      "vllm_base_commit": adapter.VLLM_COMMIT,
                                      "ascend_base_commit": adapter.ASCEND_COMMIT})
    reference = []
    current_rows, current_step, current_layer = [], 0, 0

    def consumer(*, fused_experts_input):
        args = fused_experts_input
        ids = args["topk_ids"].detach().cpu().tolist()
        weights = args["topk_weights"].detach().float().cpu().tolist()
        # Use the explicit scenario layout, never adapter.build_rows or saved steps.
        for index, (req_id, pos, token_id, prompt_length) in enumerate(current_rows):
            reference.append(dict(step_id=current_step, request_id=req_id, token_position=pos,
                                  phase="prefill" if pos < prompt_length else "decode",
                                  token_id=token_id, layer_id=current_layer,
                                  expert_ids=ids[index], expert_weights=weights[index]))
        return args["hidden_states"]

    hooks = load_source_hooks(overlay, torch, consumer) if overlay else None
    runner = runner_base()
    scheduler = owner()
    scheduler.requests = {}
    scheduler.connector = scheduler.ec_connector = scheduler.kv_event_publisher = None
    scheduler.log_stats = False
    scheduler._enqueue_waiting_request = lambda r: None
    scheduler._inflight_prefills = set()
    scheduler._connector_finished = lambda r: (False, None)
    scheduler.encoder_cache_manager = NS(free=lambda r: None)
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._free_blocks = lambda r: scheduler.requests.pop(r.request_id)
    scenarios = [[("A", 0, 2)], [("B", 0, 2), ("A", 2, 2)],
                 [("A", 4, 1), ("C", 0, 1), ("B", 2, 1)],
                 [("C", 1, 2), ("B", 3, 1), ("A", 5, 1)],
                 [("B", 4, 1), ("C", 3, 1)]]
    prompts = {"A": 4, "B": 2, "C": 3, "D": 1}
    tokens = {"A": [11, 12, 13, 14, 15, 16], "B": [21, 22, 23, 24, 25],
              "C": [31, 32, 33, 34], "D": [41]}

    def request(req_id):
        return RequestFixture(request_id=req_id, num_prompt_tokens=prompts[req_id],
                  num_output_tokens=len(tokens[req_id]) - prompts[req_id],
                  status=NS(name="FINISHED_STOPPED"), stop_reason=None, resumable=False,
                  is_finished=lambda: True, get_finished_reason=lambda: "stop",
                  get_token_id=lambda pos: tokens[req_id][pos])

    def start(req_id):
        req = request(req_id)
        if hooks:
            hooks[3](scheduler, req)
        else:
            adapter.request_started(scheduler, req)

    def finish(req_id, cancel=False):
        req = scheduler.requests[req_id] if hooks else request(req_id)
        if cancel:
            req.status.name = "FINISHED_ABORTED"
            req.get_finished_reason = lambda: "abort"
        if hooks:
            hooks[4](scheduler, req)
        else:
            adapter.request_finished(scheduler, req)

    def model(**kwargs):
        nonlocal current_layer
        actual_ids = kwargs["input_ids"].detach().cpu().tolist()
        actual_positions = kwargs["positions"].detach().cpu().tolist()
        if actual_ids != [r[2] for r in current_rows] or actual_positions != [r[1] for r in current_rows]:
            raise AssertionError("independent model forward input mismatch")
        for current_layer in (0, 1):
            layer = NS(layer_name=f"model.layers.{current_layer}.block_sparse_moe.experts",
                       vllm_config=runner.vllm_config, w13_weight=None, w2_weight=None, swiglu_limit=0)
            x = torch.zeros((len(actual_ids), 3), dtype=torch.bfloat16)
            logits = torch.tensor([[(token_id + current_layer * 3 + expert * 7) % 13
                                    for expert in range(4)] for token_id in actual_ids],
                                  dtype=torch.float32)
            if hooks:
                hooks[0](NS(tid2eid=None, dynamic_eplb=False, moe=NS(has_bias=False)),
                         layer, x, False, 2, logits, True, num_experts=4)
            else:
                values, ids = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)
                weights = (values / values.sum(dim=-1, keepdim=True)).to(x.dtype)
                adapter.record_routes(layer.layer_name, ids, weights)
                consumer(fused_experts_input=dict(topk_ids=ids, topk_weights=weights, hidden_states=x))
        return x

    runner.model = model
    if hooks:
        runner.pcp_size, runner._has_sinks, runner.dynamic_eplb = 1, False, False
        runner._model_forward = MethodType(hooks[1], runner)
    try:
        with patch.object(adapter, "_config", config), patch.object(adapter, "_session", session), \
                patch.object(adapter, "_counter", 0):
            # Calls outside real execution must remain no-ops, including dummy/profiling routes.
            adapter.record_routes(None, None, None, force_load_balance=True)
            for current_step, layout in enumerate(scenarios, 1):
                for req_id in ({1: ["A"], 2: ["B"], 3: ["C", "D"]}.get(current_step, [])):
                    start(req_id)
                current_rows = [(req_id, pos, tokens[req_id][pos], prompts[req_id])
                                for req_id, start_pos, count in layout
                                for pos in range(start_pos, start_pos + count)]
                runner.input_batch = NS(req_ids=[r for r, _, _ in layout],
                                        num_computed_tokens_cpu=np.array([p for _, p, _ in layout]))
                runner.requests = {r: request(r) for r, _, _ in layout}
                sched = NS(num_scheduled_tokens={r: n for r, _, n in reversed(layout)},
                           total_num_scheduled_tokens=len(current_rows), preempted_req_ids=set(),
                           scheduled_cached_reqs=NS(resumed_req_ids=set()))
                ids = torch.tensor([r[2] for r in current_rows], dtype=torch.int32)
                positions = torch.tensor([r[1] for r in current_rows], dtype=torch.int64)
                if hooks:
                    hooks[2](runner, sched, ids, positions)
                else:
                    with adapter.trace_execution(runner, sched, ids, positions):
                        model(input_ids=ids, positions=positions)
                if current_step == 3:
                    finish("D", cancel=True)
                if current_step == 4:
                    finish("A")
                if current_step == 5:
                    finish("B")
                    finish("C")
            if hooks:
                hooks[5](scheduler)
            else:
                adapter.close_trace()
    except Exception as exc:
        if not session._failed:
            session.fail(exc)
        raise
    reader = TraceReader(session.run_dir)
    issues = reader.validate()
    if issues:
        raise AssertionError(issues)
    expected = [r for r in reference if r["layer_id"] in session.moe_layers
                and r["phase"] in config.phases]
    actual = list(reader.detail_rows())
    key = lambda r: (r["step_id"], r["request_id"], r["token_position"], r["layer_id"])
    actual_map = {key(r): r for r in actual}
    if len(actual_map) != len(actual) or len(actual) != len(expected):
        raise AssertionError("independent reference cardinality mismatch")
    for row in expected:
        saved = actual_map[key(row)]
        for field in ("request_id", "token_position", "phase", "expert_ids"):
            if saved[field] != row[field]:
                raise AssertionError(f"independent {field} mismatch: {key(row)}")
        if config.save_token_ids and saved["token_id"] != row["token_id"]:
            raise AssertionError("token ID mismatch")
        if config.save_expert_weights and saved["expert_weights"] != row["expert_weights"]:
            raise AssertionError("consumer weights mismatch")
    steps = list(reader.jsonl("steps.jsonl"))
    counts = expert_counts(reader)
    if sum(r["count"] for r in counts) != len(actual) * 2:
        raise AssertionError("expert distribution totals mismatch")
    write_details_json(reader, output_root / "details.json")
    report = dict(state="passed", validation_level="cpu_only", npu_verified=False,
                  source_hooks_executed=bool(hooks), run_dir=str(session.run_dir),
                  executions=len(steps), route_records=len(actual), expert_choices=len(actual) * 2,
                  multi_request_executions=sum(len({r["request_id"] for r in s["rows"]}) > 1 for s in steps),
                  mixed_phase_executions=sum(len({r["phase"] for r in s["rows"]}) > 1 for s in steps),
                  independent_comparison="passed", storage_complete=reader.complete,
                  vllm_base_commit=adapter.VLLM_COMMIT, ascend_base_commit=adapter.ASCEND_COMMIT)
    atomic_json(output_root / "consumer_reference.json", {"records": reference})
    atomic_json(output_root / "cpu-validation.json", report)
    return report


def run_validation(output_root, *, overlay=None, config=None):
    """Invalidate old summaries before work; never leave a stale success on failure."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = output_root / "cpu-validation.json"
    atomic_json(summary, {"state": "running", "validation_level": "cpu_only", "npu_verified": False})
    try:
        return _run_validation(output_root, overlay=overlay, config=config)
    except Exception as exc:
        atomic_json(summary, {"state": "failed", "validation_level": "cpu_only",
                              "npu_verified": False, "independent_comparison": "not_passed",
                              "error": {"type": type(exc).__name__, "message": str(exc)}})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path,
                        help="hash-checked staged v0.23.0 sources; enables real Python hook execution")
    args = parser.parse_args()
    print(json.dumps(run_validation(args.output_root, overlay=args.overlay), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
