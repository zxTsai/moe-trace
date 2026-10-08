import ast
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from moe_trace.adapters import ascend_v41_debug as adapter
from moe_trace.distributed.debug_cli import collect_deployment
from moe_trace.distributed.decode import Worker
from moe_trace.distributed.storage import create_run, WorkerWriter
from moe_trace.errors import IntegrityError
from test_v41_debug import inputs

ROOT = Path(__file__).resolve().parents[1]


def script(path):
    spec = importlib.util.spec_from_file_location("debug_tool", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_topology_and_profile():
    launcher = script("scripts/deployment/ascend_v41_1p1d_debug.py")
    for role, dp, tp in [("P", 4, 4), ("D", 8, 2)]:
        tasks = launcher.commands(role, "/models/model", "192.0.2.1")
        assert len(tasks) == dp
        assert [int(i) for task in tasks for i in task["devices"].split(",")] == list(range(16))
        for task in tasks:
            command = task["command"]
            assert command[command.index("--tensor-parallel-size") + 1] == str(tp)
            assert "--enforce-eager" in command and "--no-async-scheduling" in command
            assert "--speculative-config" not in command
            assert "--enable-return-routed-experts" not in command


def test_pinned_debug_copy_actual_forward_block(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    vllm, ascend = os.getenv("MOE_V41_VLLM_SOURCE"), os.getenv("MOE_V41_ASCEND_SOURCE")
    if not vllm or not ascend:
        pytest.skip("fixed sources required")
    stage = script("scripts/tools/stage_ascend_v41_debug.py")
    overlay = stage.stage(Path(vllm) / "vllm", Path(ascend) / "vllm_ascend", tmp_path / "copies")
    launcher = script("scripts/deployment/ascend_v41_1p1d_debug.py")
    launcher.verify_overlay(overlay)
    assert (overlay / "debug-hooks.patch").read_bytes() == (
        ROOT / "patches/ascend/deepseek-v41-debug/0001-debug-hooks.patch").read_bytes()
    tree = ast.parse((overlay / "vllm_ascend/worker/model_runner_v1.py").read_text())
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.With)
              and any(isinstance(item.context_expr, ast.Call)
                      and isinstance(item.context_expr.func, ast.Name)
                      and item.context_expr.func.id == "trace_target" for item in node.items)]
    assert len(blocks) == 1
    # Execute the exact inserted real forward block, not the whole GPU scheduler.
    wrapper = ast.parse("def forward():\n    pass\n")
    wrapper.body[0].body = [blocks[0]]
    runner, scheduler, ids, pos = inputs()
    layer = NS(layer_name="model.layers.20.mlp.experts", router=object(),
               moe_config=NS(is_sequence_parallel=False))
    runner.model = NS(named_modules=lambda: [("layer", layer)])
    cfg = dict(enabled=True, output_root=str(tmp_path), run_id="source", node_id="D0",
               save_token_ids=True, save_expert_weights=True)
    worker = Worker("ascend-v41-debug", "source", "D0", "dp-0-tp-0", 0, 0, 0, 0)
    session = adapter.DebugSession(runner, cfg, worker)
    monkeypatch.setattr(adapter, "read_config", lambda: cfg)
    monkeypatch.setattr(adapter, "_get_session", lambda *args: session)
    prepare = type("PrepareAndFinalizeWithAll2All", (), {})()
    def model_forward(*args, **kwargs):
        adapter.capture_routes(layer, torch.tensor([[3, 1], [1, 2]]), torch.ones(2, 2),
                               prepare, NS(), NS(), 4)
        adapter.capture_consumer(layer, torch.tensor([[2, 3, 4], [3, 0, 4]]), torch.ones(2, 3))
    runner._model_forward = model_forward
    namespace = dict(self=runner, scheduler_output=scheduler, input_ids=ids, positions=pos,
                     cudagraph_mode=NS(name="NONE"), trace_target=adapter.trace_target,
                     num_tokens_padded=4, intermediate_tensors=None, inputs_embeds=None, model_kwargs={})
    exec(compile(ast.fix_missing_locations(wrapper), "patched-forward-block", "exec"), namespace)
    namespace["forward"]()
    session.writer.close()
    expected = [f"dp-{d}-tp-{t}" for d in range(8) for t in range(2)]
    run = tmp_path / "source"
    (run / "deployment.json").write_text(json.dumps(dict(profile="1P1D-debug", run_id="source",
                                                        node_id="D0", expected_workers=expected)))
    for dp in range(8):
        for tp in range(2):
            if (dp, tp) == (0, 0):
                continue
            w = Worker("ascend-v41-debug", "source", "D0", f"dp-{dp}-tp-{tp}", dp, tp, dp*2+tp, dp*2+tp)
            directory = create_run(run / w.worker / "trace", [w], {"target": [20]},
                                   validation_level="a3_unverified_debug")
            WorkerWriter(directory, w).close()
    _, records = collect_deployment(run)
    assert len(records) == 1
    route = records[0][0]
    assert (route.row.local_request, route.row.token_id, route.row.position, route.expert_ids) == (
        "B", 21, 1, (3, 1))
    (run / "dp-7-tp-1" / "trace" / "worker-00000" / "status.json").unlink()
    with pytest.raises(IntegrityError):
        collect_deployment(run)


def test_idle_initialization_and_disabled_hook(monkeypatch):
    monkeypatch.setattr(adapter, "read_config", lambda: None)
    adapter.initialize_debug(None)
    calls = []
    cfg = {"enabled": True}
    monkeypatch.setattr(adapter, "read_config", lambda: cfg)
    monkeypatch.setattr(adapter, "check_runtime", lambda *args: calls.append("checked"))
    monkeypatch.setattr(adapter, "_get_session", lambda *args: calls.append("initialized"))
    adapter.initialize_debug(NS(vllm_config=NS(compilation_config=NS(cudagraph_mode="NONE"))))
    assert calls == ["checked", "initialized"]
