import ast
from dataclasses import replace
import importlib.util
import json
import os
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from moe_trace.adapters import ascend_qwen36 as adapter
from moe_trace.distributed.decode import Worker, DecodeRow, Route
from moe_trace.distributed.qwen_cli import collect
from moe_trace.distributed.storage import collect_run
from moe_trace.errors import ConfigError, IntegrityError

ROOT = Path(__file__).resolve().parents[1]


def runtime():
    return NS(vllm_config=NS(model_config=NS(enforce_eager=True, quantization="ascend",
        hf_config=NS(architectures=["Qwen3_5MoeForConditionalGeneration"])),
        parallel_config=NS(data_parallel_size=8, tensor_parallel_size=2, pipeline_parallel_size=1,
                           enable_expert_parallel=True),
        scheduler_config=NS(async_scheduling=False), speculative_config=None,
        cache_config=NS(enable_prefix_caching=False), kv_transfer_config=None,
        compilation_config=NS(cudagraph_mode=NS(name="NONE"))),
        ascend_config=NS(eplb_config=NS(dynamic_eplb=False)))


def config(tmp_path, run="cpu"):
    return dict(enabled=True, output_root=str(tmp_path), run_id=run, node_id="A3",
                save_token_ids=True, save_expert_weights=True)


def load_stage(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/tools"))
    spec = importlib.util.spec_from_file_location("qwen_stage", ROOT / "scripts/tools/stage_qwen36_ascend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function(path, class_name, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


def test_cpu_actual_patched_forward_prepare_and_w8a8(tmp_path, monkeypatch):
    source = os.environ.get("MOE_QWEN_ASCEND_SOURCE")
    if not source:
        pytest.skip("set MOE_QWEN_ASCEND_SOURCE for actual source hook execution")
    stage = load_stage(monkeypatch)
    overlay = stage.stage(Path(source) / "vllm_ascend", tmp_path / "overlays")
    assert (overlay / "hooks.patch").read_bytes() == (ROOT / "patches/ascend/qwen36-023post1/0001-qwen-hooks.patch").read_bytes()
    # The source remains untouched, while the copied real methods invoke new hooks.
    from inspect_qwen36_ascend import inspect_sources
    assert inspect_sources(Path(source) / "vllm_ascend")["all_reference_files_match"]
    cfg = config(tmp_path)
    monkeypatch.setattr(adapter, "read_config", lambda: cfg)
    prepare_fn = function(overlay / "vllm_ascend/ops/fused_moe/prepare_finalize.py",
        "PrepareAndFinalizeWithAll2All", "prepare",
        dict(torch=torch, nn=torch.nn, QuantType=NS(NONE="none"), MoEPrepareOutput=lambda **kw: NS(**kw)))
    kernel_inputs = []
    def kernel(fused_experts_input):
        value = fused_experts_input
        kernel_inputs.append((value.topk_ids.clone(), value.topk_weights.clone()))
        return value.hidden_states
    prepare = type("PrepareAndFinalizeWithAll2All", (), {"prepare": prepare_fn})()
    prepare.tp_size = 2
    comm = NS(prepare_finalize=prepare, fused_experts=kernel)
    extra = NS(moe_comm_method=comm, moe_comm_type="ALLTOALL", flash_comm_v1_enabled=False)
    def selector(**kw):
        # Mock router implementation used by the actual apply; collector never recomputes it.
        values, ids = torch.topk(kw["router_logits"], kw["top_k"], dim=-1)
        return values / values.sum(-1, keepdim=True), ids
    namespace = dict(torch=torch, Callable=object, Any=object,
        get_moe_num_logical_experts=lambda layer, n, **kw: n, select_experts=selector,
        _EXTRA_CTX=extra, get_forward_context=lambda: NS(dp_metadata=None),
        get_ascend_config=lambda: NS(enable_fused_mc2=0), MoECommType=NS(FUSED_MC2="FUSED_MC2"),
        build_fused_experts_input=lambda **kw: NS(**kw))
    apply_fn = function(overlay / "vllm_ascend/quantization/methods/w8a8_dynamic.py",
        "AscendW8A8DynamicFusedMoEMethod", "apply", namespace)
    method = NS(multistream_overlap_gate=False, dynamic_eplb=False, in_dtype=torch.float32, quant_type="W8A8")
    routed = [NS(w13_weight=None, w2_weight=None, w13_weight_scale_fp32=None, w2_weight_scale=None,
                 swiglu_limit=None) for _ in range(2)]
    modules = [(str(i), NS(layer_name=f"model.language_model.layers.{i+2}.mlp.experts",
        router=object(), moe_config=NS(), routed_experts=routed[i])) for i in range(2)]
    runner = runtime()
    runner.model = NS(named_modules=lambda: modules)
    runner.requests = {
        "A": NS(num_prompt_tokens=2, get_token_id=lambda pos: [10, 11, 12, 13][pos]),
        "B": NS(num_prompt_tokens=1, get_token_id=lambda pos: [20, 21, 22][pos])}
    tree = ast.parse((overlay / "vllm_ascend/worker/model_runner_v1.py").read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Name)
        and i.context_expr.func.id == "trace_target" for i in n.items))
    wrapper = ast.parse("def forward():\n    pass\n")
    wrapper.body[0].body = [block]
    sessions = []
    expected = {}
    for dp in range(8):
        for tp in range(2):
            worker = Worker(adapter.PROFILE, "cpu", "A3", f"dp-{dp}-tp-{tp}", dp, tp, dp*2+tp, dp*2+tp)
            session = adapter.Session(runner, cfg, worker)
            sessions.append(session)
            monkeypatch.setattr(adapter, "get_session", lambda *args, s=session: s)
            prepare.tp_rank = tp
            if dp:
                session.close()
                continue
            for step in (1, 2):
                if step == 1:
                    runner.input_batch = NS(req_ids=["B", "A"], num_computed_tokens_cpu=[1, 1])
                    schedule = NS(num_scheduled_tokens={"A": 2, "B": 1}, total_num_scheduled_tokens=3)
                    ids, positions = torch.tensor([21, 11, 12, 0]), torch.tensor([1, 1, 2, 0])
                    logits = torch.tensor([[1., 8., 2., 9.], [7., 1., 8., 2.], [8., 1., 9., 2.], [1., 2., 3., 4.]])
                else:
                    runner.input_batch = NS(req_ids=["A", "B"], num_computed_tokens_cpu=[3, 2])
                    schedule = NS(num_scheduled_tokens={"B": 1, "A": 1}, total_num_scheduled_tokens=2)
                    ids, positions = torch.tensor([13, 22]), torch.tensor([3, 2])
                    logits = torch.tensor([[7., 8., 1., 2.], [1., 8., 9., 2.]])
                def model_forward(*args, **kwargs):
                    prepared = prepare.prepare(torch.zeros(len(ids), 2), logits)
                    for layer in routed:
                        apply_fn(method, layer, prepared.hidden_states, prepared.router_logits,
                                 top_k=2, renormalize=True, num_experts=4)
                runner._model_forward = model_forward
                ns = dict(self=runner, scheduler_output=schedule, input_ids=ids,
                    positions=positions.repeat(3, 1), cudagraph_mode=NS(name="NONE"),
                    trace_target=adapter.trace_target, num_tokens_padded=len(ids),
                    intermediate_tensors=None, inputs_embeds=None, model_kwargs={})
                exec(compile(ast.fix_missing_locations(wrapper), "actual-forward-hook", "exec"), ns)
                ns["forward"]()
            session.close()
    routes = collect(tmp_path / "cpu")
    assert len(routes) == 8  # four Decode tokens times two actual MoE layers
    by_token = {(r.row.local_request, r.row.position, r.layer): r.expert_ids for r in routes}
    for layer in (2, 3):
        assert by_token[("B", 1, layer)] == (3, 1)
        assert by_token[("A", 2, layer)] == (2, 0)
        assert by_token[("A", 3, layer)] == (1, 0)
        assert by_token[("B", 2, layer)] == (2, 1)
    assert len(kernel_inputs) == 8
    assert kernel_inputs[0][0].tolist() == [[3, 1], [2, 0]]
    assert kernel_inputs[4][0].tolist() == [[2, 0], [3, 2]]
    # No prompt or padding rows in output; original source checksum still matches.
    assert {r.row.token_id for r in routes} == {12, 13, 21, 22}
    assert inspect_sources(Path(source) / "vllm_ascend")["all_reference_files_match"]


@pytest.mark.parametrize("field,value", [("async_scheduling", True)])
def test_runtime_rejects_async(field, value):
    runner = runtime()
    setattr(runner.vllm_config.scheduler_config, field, value)
    with pytest.raises(ConfigError):
        adapter.check_runtime(runner, NS(name="NONE"))


def test_multimodal_and_disabled_capture(monkeypatch):
    with pytest.raises(ConfigError):
        adapter.text_positions(torch.tensor([[0, 1], [0, 2], [0, 1]]))
    monkeypatch.delenv("MOE_TRACE_QWEN36_CONFIG", raising=False)
    with adapter.trace_target(None, None, None, None, None):
        adapter.capture_routes(None, None, None, None, None, None, None)
        adapter.capture_consumer(None, None)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "consumer", "token", "privacy"])
def test_capture_failures_and_privacy(tmp_path, monkeypatch, failure):
    runner = runtime()
    routed = NS()
    runner.model = NS(named_modules=lambda: [("moe", NS(layer_name="model.layers.2.mlp.experts",
        router=object(), moe_config=NS(), routed_experts=routed))])
    runner.input_batch = NS(req_ids=["A"], num_computed_tokens_cpu=[1])
    runner.requests = {"A": NS(num_prompt_tokens=1, get_token_id=lambda pos: 21)}
    schedule = NS(num_scheduled_tokens={"A": 1}, total_num_scheduled_tokens=1)
    cfg = config(tmp_path)
    if failure == "privacy":
        cfg["save_token_ids"] = False
    worker = Worker(adapter.PROFILE, "cpu", "A3", "dp-0-tp-0", 0, 0, 0, 0)
    session = adapter.Session(runner, cfg, worker)
    monkeypatch.setattr(adapter, "read_config", lambda: cfg)
    monkeypatch.setattr(adapter, "get_session", lambda *args: session)
    prepare = type("PrepareAndFinalizeWithAll2All", (), {})()
    def execute():
        with adapter.trace_target(runner, schedule, torch.tensor([22 if failure == "token" else 21, 0]),
                                  torch.tensor([1, 0]), NS(name="NONE")):
            if failure == "missing":
                return
            ids = torch.tensor([[3, 1]])
            adapter.capture_routes(routed, ids, torch.ones(1, 2), prepare, NS(), NS(), 4)
            if failure == "duplicate":
                adapter.capture_routes(routed, ids, torch.ones(1, 2), prepare, NS(), NS(), 4)
            adapter.capture_consumer(routed, torch.tensor([[2, 1]]) if failure == "consumer" else ids)
    if failure != "privacy":
        with pytest.raises(IntegrityError):
            execute()
        assert session.writer.failed
        with pytest.raises(IntegrityError):
            collect_run(session.root)
    else:
        execute()
        session.close()
        _, rows = collect_run(session.root)
        assert rows[0][0].row.token_id is None and rows[0][0].row.input_role == "redacted"


def test_launcher_and_overlay_integrity(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("qwen_launch", ROOT / "scripts/deployment/qwen36_capture.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    argv = launcher.command("/models/qwen")
    assert argv[argv.index("--data-parallel-size-local") + 1] == "8"
    assert "--enforce-eager" in argv and "--no-async-scheduling" in argv
    assert "--no-enable-prefix-caching" in argv and "--enable-expert-parallel" in argv
    source = os.environ.get("MOE_QWEN_ASCEND_SOURCE")
    if source:
        stage = load_stage(monkeypatch)
        overlay = stage.stage(Path(source) / "vllm_ascend", tmp_path / "overlays")
        launcher.verify_overlay(overlay)
        (overlay / "vllm_ascend/worker/worker.py").write_text("# modified\n")
        with pytest.raises(ValueError, match="changed"):
            launcher.verify_overlay(overlay)


@pytest.mark.parametrize("failure", [None, "missing_route", "coverage_hash", "tp_disagree", "all_empty", "incomplete"])
def test_cross_worker_coverage(tmp_path, failure):
    runner = runtime()
    routed = NS()
    runner.model = NS(named_modules=lambda: [("moe", NS(layer_name="model.layers.2.mlp.experts",
        router=object(), moe_config=NS(), routed_experts=routed))])
    row = DecodeRow("Qwen-dp-0", "execution-1", 1, "Qwen-dp-0:A", "A", 0, 1, 21, "target_decode")
    for dp in range(8):
        for tp in range(2):
            worker = Worker(adapter.PROFILE, "cpu", "A3", f"dp-{dp}-tp-{tp}", dp, tp, dp*2+tp, dp*2+tp)
            session = adapter.Session(runner, config(tmp_path), worker)
            if dp == 0 and failure != "all_empty":
                session.step = 1
                session.record_inputs({0: row})
                if tp == 0 and failure != "missing_route":
                    session.writer.declare_execution("target", [row])
                    session.writer.append(Route(worker, worker, row, "target", 2, 4, (3, 1)))
            if failure == "incomplete" and (dp, tp) == (7, 1):
                continue
            session.close()
    root = tmp_path / "cpu"
    target = root / "dp-0-tp-1"
    if failure == "coverage_hash":
        with (target / "inputs.jsonl").open("ab") as stream:
            stream.write(b"\n")
    elif failure == "tp_disagree":
        data = (json.dumps({"execution": "execution-1", "rows": []}, sort_keys=True) + "\n").encode()
        (target / "inputs.jsonl").write_bytes(data)
        (target / "inputs-index.json").write_text(json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "executions": 1}))
    if failure:
        with pytest.raises((IntegrityError, OSError)):
            collect(root)
    else:
        assert len(collect(root)) == 1


@pytest.mark.parametrize("tp", [0, 1])
def test_unequal_dp_allgather_ownership(tmp_path, monkeypatch, tp):
    runner = runtime()
    routed = NS()
    runner.model = NS(named_modules=lambda: [("moe", NS(layer_name="model.layers.2.mlp.experts",
        router=object(), moe_config=NS(), routed_experts=routed))])
    runner.input_batch = NS(req_ids=["A"], num_computed_tokens_cpu=[1])
    runner.requests = {"A": NS(num_prompt_tokens=1, get_token_id=lambda pos: 21)}
    schedule = NS(num_scheduled_tokens={"A": 1}, total_num_scheduled_tokens=1)
    cfg = config(tmp_path)
    worker = Worker(adapter.PROFILE, "cpu", "A3", f"dp-1-tp-{tp}", 1, tp, 2+tp, 2+tp)
    session = adapter.Session(runner, cfg, worker)
    monkeypatch.setattr(adapter, "read_config", lambda: cfg)
    monkeypatch.setattr(adapter, "get_session", lambda *args: session)
    prepare = type("PrepareAndFinalizeWithAllGather", (), {})()
    # Max DP 3, padded forward 2, local actual 1. Local block begins at selector row 3.
    meta = NS(num_tokens_across_dp_cpu=torch.tensor([3, 1, 2, 0, 0, 0, 0, 0]))
    ids = torch.tensor([[0, 1]] * 24)
    ids[3] = torch.tensor([3, 2])
    with adapter.trace_target(runner, schedule, torch.tensor([21, 0]), torch.tensor([1, 0]), NS(name="NONE")):
        adapter.capture_routes(routed, ids, torch.ones(24, 2), prepare, NS(), NS(dp_metadata=meta), 4)
        adapter.capture_consumer(routed, ids)
    session.close()
    _, records = collect_run(session.root, allow_empty=True)
    assert len(records) == (1 if tp == 0 else 0)
    if tp == 0:
        assert records[0][0].expert_ids == (3, 2)
