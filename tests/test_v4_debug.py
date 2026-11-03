"""V4 source-hook CPU validation; NPU kernels/weights/communication are mocked."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from moe_trace.adapters import ascend_v4_debug as adapter
from moe_trace.distributed.decode import Worker
from moe_trace.distributed.storage import collect_run
from moe_trace.distributed.v4_cli import collect, analyze
from moe_trace.errors import ConfigError, IntegrityError

ROOT = Path(__file__).resolve().parents[1]


def script(path):
    spec = importlib.util.spec_from_file_location("v4_tool", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method(path, cls_name, name, namespace):
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


def runtime():
    return NS(vllm_config=NS(model_config=NS(enforce_eager=True, quantization="ascend",
        hf_config=NS(architectures=["DeepseekV4ForCausalLM"], num_hidden_layers=2)),
        parallel_config=NS(data_parallel_size=16, tensor_parallel_size=1, pipeline_parallel_size=1,
                           enable_expert_parallel=True),
        scheduler_config=NS(async_scheduling=False), speculative_config=None,
        cache_config=NS(enable_prefix_caching=False),
        kv_transfer_config=NS(kv_connector="MooncakeHybridConnector", kv_role="kv_consumer"),
        compilation_config=NS(cudagraph_mode=NS(name="NONE"))),
        ascend_config=NS(eplb_config=NS(dynamic_eplb=False)))


def config(tmp_path):
    return dict(enabled=True, output_root=str(tmp_path), run_id="cpu", node_id="D0",
                save_token_ids=True, save_expert_weights=True)


def batch(runner, second=False):
    runner.requests = {
        "A": NS(num_prompt_tokens=2, get_token_id=lambda p: [10, 11, 12, 13][p]),
        "B": NS(num_prompt_tokens=1, get_token_id=lambda p: [20, 21, 22][p])}
    # Deliberately invert schedule insertion order versus actual input batch.
    runner.input_batch = NS(req_ids=["A", "B"] if second else ["B", "A"],
        num_computed_tokens_cpu=[3, 2] if second else [1, 1])
    scheduler = NS(num_scheduled_tokens={"B": 1, "A": 1} if second else {"A": 2, "B": 1},
                   total_num_scheduled_tokens=2 if second else 3)
    return (scheduler, torch.tensor([13, 22]) if second else torch.tensor([21, 11, 12]),
            torch.tensor([3, 2]) if second else torch.tensor([1, 1, 2]))


def layer(number):
    return NS(layer_name=f"model.layers.{number}.mlp.experts", router=NS(bias_vl=None),
              moe_config=NS(is_sequence_parallel=False, num_experts=4), log2phy=None)


def session(tmp_path, monkeypatch, dp=0):
    runner = runtime()
    layers = [layer(0), layer(1)]
    runner.model = NS(named_modules=lambda: [(str(i), l) for i, l in enumerate(layers)])
    cfg = config(tmp_path)
    worker = Worker(adapter.PROFILE, "cpu", "D0", f"dp-{dp}-tp-0", dp, 0, dp, dp)
    current = adapter.Session(runner, cfg, worker)
    monkeypatch.setattr(adapter, "read_config", lambda: cfg)
    monkeypatch.setattr(adapter, "get_session", lambda *a: current)
    return runner, layers, current


def make_deployment(tmp_path):
    run = tmp_path / "cpu"
    (run / "deployment.json").write_text(json.dumps(dict(profile=adapter.PROFILE, run_id="cpu",
        node_id="D0", expected_workers=[f"dp-{d}-tp-0" for d in range(16)])))
    return run


def test_actual_patched_v4_source_cpu(tmp_path, monkeypatch):
    vllm, ascend = os.environ.get("MOE_V4_VLLM_SOURCE"), os.environ.get("MOE_V4_ASCEND_SOURCE")
    if not vllm or not ascend:
        pytest.skip("set MOE_V4_VLLM_SOURCE and MOE_V4_ASCEND_SOURCE for source integration")
    stage = script("scripts/tools/stage_ascend_v4_debug.py")
    overlay = stage.stage(Path(vllm) / "vllm", Path(ascend) / "vllm_ascend", tmp_path / "overlays")
    launcher = script("scripts/deployment/ascend_v4_1p1d_debug.py")
    launcher.verify_overlay(overlay)
    worker_tree = ast.parse((overlay / "vllm_ascend/worker/worker.py").read_text())
    hook_names = [alias.name for n in ast.walk(worker_tree) if isinstance(n, ast.ImportFrom)
                  and n.module == "moe_trace.adapters.ascend_v4_debug" for alias in n.names]
    assert set(hook_names) == {"initialize", "close"}
    assert all(callable(getattr(adapter, name)) for name in hook_names)
    assert (overlay / "debug-hooks.patch").read_bytes() == (
        ROOT / "patches/ascend/deepseek-v4-debug/0001-debug-hooks.patch").read_bytes()
    stage.check_sources(Path(vllm) / "vllm", Path(ascend) / "vllm_ascend")
    runner, layers, current = session(tmp_path, monkeypatch, dp=3)
    source = overlay / "vllm_ascend"
    prepare_fn = method(source / "ops/fused_moe/prepare_finalize.py", "PrepareAndFinalizeWithAll2All", "prepare",
                        dict(torch=torch, QuantType=NS(NONE="none"), MoEPrepareOutput=lambda **kw: NS(**kw)))
    pad_ids_fn = method(source / "ops/fused_moe/prepare_finalize.py", "PrepareAndFinalizeWithAll2All",
                        "pad_and_split_input_ids", dict(torch=torch, nn=torch.nn))
    prepare = type("PrepareAndFinalizeWithAll2All", (), {"prepare": prepare_fn,
                    "pad_and_split_input_ids": pad_ids_fn})()
    prepare.tp_size, prepare.tp_rank, prepare.lora_context = 1, 0, None
    kernel_inputs, hash_calls = [], []
    def kernel(*, fused_experts_input, quant_method):
        kernel_inputs.append(fused_experts_input.topk_ids.clone())
        return NS(routed_out=fused_experts_input.hidden_states)
    comm = NS(prepare_finalize=prepare, prepare=prepare.prepare, fused_experts=kernel,
        pad_and_split_input_ids=prepare.pad_and_split_input_ids, finalize=lambda **kw: kw["hidden_states"])
    extra = NS(moe_comm_method=comm, moe_comm_type="ALLTOALL", in_profile_run=False,
               eplb_heat_collection_status=False, flash_comm_v1_enabled=False)
    context = NS(dp_metadata=None, all_moe_layers=[])
    # Mock only the NPU routing kernel, not the actual Python router method.
    def gating(**kw):
        hash_calls.append(kw["tid2eid"] is not None)
        if kw["tid2eid"] is not None:
            ids = kw["tid2eid"][kw["input_ids"]].to(torch.int32)
            weights = torch.tensor([[0.75, 0.25]]).repeat(len(ids), 1)
        else:
            weights, ids = torch.topk(kw["x"], kw["k"], dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
        return weights, ids, None
    monkeypatch.setattr(torch.ops._C_ascend, "moe_gating_top_k_hash", gating, raising=False)
    routing = method(source / "ops/fused_moe/router/fused_topk_router.py", "AscendFusedTopKRouter",
        "_compute_routing", dict(torch=torch, _EXTRA_CTX=extra, MoECommType=NS(ALLGATHER="ALLGATHER"),
                                DEEPSEEK_V4_IMAGE_SENTINEL_COUNT=1))
    namespace = dict(torch=torch, _EXTRA_CTX=extra, get_forward_context=lambda: context,
        get_ascend_config=lambda: NS(enable_force_eplb=False),
        get_moe_num_logical_experts=lambda l, n, **kw: n,
        get_moe_comm_method=lambda *a: comm, AllGatherCommImpl=type("AllGatherCommImpl", (), {}),
        build_fused_experts_input=lambda **kw: NS(**kw))
    select = method(source / "ops/fused_moe/routed_experts.py", "AscendRoutedExperts", "_select_experts", namespace)
    forward = method(source / "ops/fused_moe/routed_experts.py", "AscendRoutedExperts", "forward_impl", namespace)
    apply = method(source / "quantization/methods/w8a8/w8a8_dynamic.py",
                   "AscendW8A8DynamicFusedMoEMethod", "apply", namespace)
    hash_table = torch.tensor([[(i + 1) % 4, (i + 3) % 4] for i in range(32)])
    for i, l in enumerate(layers):
        router = NS(bias_vl=None, tid2eid=hash_table if i == 0 else None,
            topk_group=None, num_expert_group=None, renormalize=True, scoring_func="sqrtsoftplus",
            e_score_correction_bias=None, top_k=2, routed_scaling_factor=1.0,
            image_sentinel_lo=129257, is_fused_supported=lambda x: True)
        router._select_experts = lambda _r=router, **kw: routing(_r, indices_type=None, **kw)
        l.router, l.log2phy = router, torch.tensor([1, 3, 0, 2])
        l.n_shared_experts, l.global_redundant_expert_num, l.mix_placement = 0, 0, False
        l.enable_npugraph_ex_static_kernel, l._use_v2_model_runner = False, False
        l.dynamic_eplb, l.return_with_event, l.quant_type = False, False, "w8a8"
        l.ascend_expert_map, l.apply_router_weight_on_input = None, False
        scheme = NS(quant_type="w8a8", use_expert_weight_list=False)
        l.quant_method = NS(apply=lambda _s=scheme, **kw: apply(_s, **kw))
        l._select_experts = lambda _l=l, **kw: select(_l, **kw)
    # The real inserted runner block executes the actual prepare/selector/consumer Python methods.
    tree = ast.parse((source / "worker/model_runner_v1.py").read_text())
    blocks = [n for n in ast.walk(tree) if isinstance(n, ast.With) and any(
        isinstance(x.context_expr, ast.Call) and isinstance(x.context_expr.func, ast.Name)
        and x.context_expr.func.id == "trace_target" for x in n.items)]
    assert len(blocks) == 1
    wrapper = ast.parse("def run():\n    pass\n")
    wrapper.body[0].body = [blocks[0]]
    for second in (False, True):
        scheduler, ids, positions = batch(runner, second)
        def model_forward(*a, **kw):
            for l in layers:
                logits = torch.tensor([[1., 4., 2., 3.]]).repeat(len(ids), 1)
                forward(l, hidden_states=torch.zeros(len(ids), 2), router_logits=logits, input_ids=ids)
        runner._model_forward = model_forward
        values = dict(self=runner, scheduler_output=scheduler, input_ids=ids, positions=positions,
            cudagraph_mode=NS(name="NONE"), trace_target=adapter.trace_target,
            num_tokens_padded=len(ids), intermediate_tensors=None, inputs_embeds=None, model_kwargs={})
        exec(compile(ast.fix_missing_locations(wrapper), "actual-v4-forward-block", "exec"), values)
        values["run"]()
    current.close()
    for d in range(16):
        if d == 3:
            continue
        w = Worker(adapter.PROFILE, "cpu", "D0", f"dp-{d}-tp-0", d, 0, d, d)
        adapter.Session(runner, config(tmp_path), w).close()
    run = make_deployment(tmp_path)
    records = collect(run)
    assert len(records) == 8 and hash_calls == [True, False, True, False]
    actual = {(r.row.execution, r.row.local_request, r.layer):
              (r.row.token_id, r.row.position, r.expert_ids) for r in records}
    expected_rows = [("execution-1", "B", 21, 1, (2, 0)), ("execution-1", "A", 12, 2, (1, 3)),
                     ("execution-2", "A", 13, 3, (2, 0)), ("execution-2", "B", 22, 2, (3, 1))]
    expected = {(ex, req, layer): (tok, pos, experts if layer == 0 else (1, 3))
                for ex, req, tok, pos, experts in expected_rows for layer in range(2)}
    assert actual == expected
    assert [x.tolist() for x in kernel_inputs] == [
        [[0, 1], [1, 0], [3, 2]], [[3, 2]] * 3, [[0, 1], [2, 3]], [[3, 2]] * 2]
    counts, summary, _ = analyze(records)
    assert sum(counts.values()) == 16 and summary[1]["unused_experts"] == 2
    assert summary[1]["max_over_mean"] == 2


@pytest.mark.parametrize("field,value", [("speculative_config", NS(method="mtp")),
    ("use_v2_model_runner", True)])
def test_reject_unsupported_runtime(field, value):
    runner = runtime()
    setattr(runner.vllm_config, field, value)
    with pytest.raises(ConfigError):
        adapter.check_runtime(runner, NS(name="NONE"))


def test_topology_graph_async_multistream_guards():
    runner = runtime()
    adapter.check_runtime(runner, NS(name="NONE"))
    for obj, key, value in [(runner.vllm_config.parallel_config, "data_parallel_size", 8),
        (runner.vllm_config.parallel_config, "tensor_parallel_size", 2),
        (runner.vllm_config.scheduler_config, "async_scheduling", True),
        (runner.ascend_config, "multistream_overlap_shared_expert", True)]:
        old = getattr(obj, key, False)
        setattr(obj, key, value)
        with pytest.raises(ConfigError):
            adapter.check_runtime(runner, NS(name="NONE"))
        setattr(obj, key, old)
    with pytest.raises(ConfigError):
        adapter.check_runtime(runner, NS(name="FULL_DECODE_ONLY"))
    runner.vllm_config.model_config.hf_config.architectures = ["DeepseekV41ForCausalLM"]
    with pytest.raises(ConfigError):
        adapter.check_runtime(runner, NS(name="NONE"))


@pytest.mark.parametrize("failure", ["missing_layer", "consumer_mutation", "row_loss", "duplicate_layer"])
def test_capture_failure_marks_run_failed(tmp_path, monkeypatch, failure):
    runner, layers, current = session(tmp_path, monkeypatch)
    scheduler, ids, positions = batch(runner)
    prepare = type("PrepareAndFinalizeWithAll2All", (), {})()
    with pytest.raises(IntegrityError):
        with adapter.trace_target(runner, scheduler, ids, positions, NS(name="NONE")):
            for l in layers[:1] if failure == "missing_layer" else layers:
                values = torch.tensor([[3, 1], [1, 2], [0, 2]])
                adapter.capture_routes(l, values[:2] if failure == "row_loss" else values,
                    torch.ones(2 if failure == "row_loss" else 3, 2), prepare, NS(), NS(), 4)
                adapter.capture_consumer(l, torch.zeros_like(values) if failure == "consumer_mutation" else values)
                if failure == "duplicate_layer":
                    adapter.capture_routes(l, values, torch.ones(3, 2), prepare, NS(), NS(), 4)
    assert current.writer.failed
    with pytest.raises(IntegrityError):
        collect_run(current.root)


def test_launcher_topology_matches_user_and_no_speculation():
    launcher = script("scripts/deployment/ascend_v4_1p1d_debug.py")
    for role, dp, tp in [("P", 4, 4), ("D", 16, 1)]:
        tasks = launcher.commands(role, "/models/dsv4", "192.0.2.1")
        assert len(tasks) == dp
        assert [int(x) for t in tasks for x in t["devices"].split(",")] == list(range(16))
        for task in tasks:
            command = task["command"]
            assert command[command.index("--tensor-parallel-size") + 1] == str(tp)
            assert "--enforce-eager" in command and "--no-async-scheduling" in command
            assert "--speculative-config" not in command and "--engram-config" not in command
            assert "deepseek_v4" in command and "dsv4" in command
            kv = json.loads(command[command.index("--kv-transfer-config") + 1])
            assert kv["kv_connector_extra_config"]["decode"] == {"dp_size": 16, "tp_size": 1}


def test_disabled_does_not_touch_model_or_disk(monkeypatch):
    monkeypatch.delenv("MOE_TRACE_V4_DEBUG_CONFIG", raising=False)
    adapter.initialize(None)
    with adapter.trace_target(None, None, None, None, None):
        adapter.capture_routes(None, None, None, None, None, None, None)
        adapter.capture_consumer(None, None)


def test_tp1_allgather_filters_foreign_dp_and_prompt(tmp_path, monkeypatch):
    runner, layers, current = session(tmp_path, monkeypatch, dp=1)
    scheduler, ids, positions = batch(runner)
    values = torch.tensor([[3, 1], [1, 2], [0, 2]] * 16)
    prepare = type("PrepareAndFinalizeWithAllGather", (), {})()
    meta = NS(num_tokens_across_dp_cpu=torch.tensor([1, 3] + [0] * 14))
    with adapter.trace_target(runner, scheduler, ids, positions, NS(name="NONE")):
        for l in layers:
            adapter.capture_routes(l, values, torch.ones(48, 2), prepare, NS(), NS(dp_metadata=meta), 4)
            adapter.capture_consumer(l, values)
    current.close()
    _, joined = collect_run(current.root)
    assert len(joined) == 4
    assert {(r.row.local_request, r.expert_ids) for r, _ in joined} == {("B", (3, 1)), ("A", (0, 2))}


@pytest.mark.parametrize("failure", ["missing_worker", "unfinished", "hash", "all_idle", "missing_coverage"])
def test_merge_rejects_incomplete_runs(tmp_path, failure):
    runner = runtime()
    runner.model = NS(named_modules=lambda: [("l0", layer(0)), ("l1", layer(1))])
    for d in range(16):
        w = Worker(adapter.PROFILE, "cpu", "D0", f"dp-{d}-tp-0", d, 0, d, d)
        current = adapter.Session(runner, config(tmp_path), w)
        if failure == "unfinished" and d == 15:
            continue
        current.close()
    run = make_deployment(tmp_path)
    if failure == "missing_worker":
        (run / "dp-15-tp-0").rename(tmp_path / "removed-worker")
    elif failure == "hash":
        (run / "dp-0-tp-0" / "inputs.jsonl").write_text("{}\n")
    elif failure == "missing_coverage":
        p = run / "dp-0-tp-0"
        data = b'{"execution":"execution-1","rows":[{"input_row":0,"request":"B","position":1}]}\n'
        (p / "inputs.jsonl").write_bytes(data)
        (p / "inputs-index.json").write_text(json.dumps({"executions": 1, "sha256": hashlib.sha256(data).hexdigest()}))
    with pytest.raises(IntegrityError):
        collect(run)
