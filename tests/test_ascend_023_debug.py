"""CPU tensors and framework stubs; these tests never claim NPU acceptance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

from moe_trace.adapters import ascend_023_debug as adapter
from moe_trace.adapters.ascend_graph_bridge import arm_graph_capture

ROOT = Path(__file__).resolve().parents[1]


def script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime():
    return NS(vllm_config=NS(
        parallel_config=NS(data_parallel_size=16, tensor_parallel_size=1, enable_expert_parallel=True),
        model_config=NS(quantization="ascend", hf_config=NS(architectures=["DeepseekV4ForCausalLM"], num_hidden_layers=1)),
        cache_config=NS(enable_prefix_caching=False), speculative_config=NS(method="mtp", enforce_eager=True, num_speculative_tokens=1),
        kv_transfer_config=NS(kv_connector="MooncakeHybridConnector", kv_role="kv_consumer"),
        compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY")),
        scheduler_config=NS(async_scheduling=True)),
        ascend_config=NS(multistream_overlap_shared_expert=True,
                         ascend_compilation_config=NS(enable_npugraph_ex=True)),
        requests={"A": NS(num_prompt_tokens=2), "B": NS(num_prompt_tokens=1)},
        input_batch=NS(req_ids=["A", "B"]),
        discard_request_indices=NS(np=np.array([], dtype=np.int32)), num_discarded_requests=0)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    adapter.close()
    runner = runtime()
    cfg = dict(output_root=str(tmp_path), run_id="cpu", node_id="D", save_token_ids=True,
               save_expert_weights=True, max_executions=100, max_file_bytes=1024*1024)
    session = adapter.Session(runner, cfg, dict(dp=0, tp=0, ep=0))
    monkeypatch.setattr(adapter, "_session", session)
    monkeypatch.setattr(adapter, "synchronize", lambda: None)
    vendor = ModuleType("vllm")
    context = ModuleType("vllm.forward_context")
    context.get_forward_context = lambda: NS(dp_metadata=None)
    ascend = ModuleType("vllm_ascend")
    extra = ModuleType("vllm_ascend.ascend_forward_context")
    extra._EXTRA_CTX = NS(padded_num_tokens=None)
    for name, value in (("vllm", vendor), ("vllm.forward_context", context),
                        ("vllm_ascend", ascend), ("vllm_ascend.ascend_forward_context", extra)):
        monkeypatch.setitem(sys.modules, name, value)
    yield runner, session
    adapter.close()
    arm_graph_capture(None)


def capture(name, ids):
    prepare = type("PrepareAndFinalizeWithAll2All", (), {})()
    adapter.capture_routes(NS(layer_name=name), ids, torch.ones_like(ids, dtype=torch.float32), prepare, 8)


def events(session):
    return [json.loads(line) for line in (session.root/"events.jsonl").read_text().splitlines()]


def execute_target(runner, session, candidate=False):
    scheduler = NS(num_scheduled_tokens={"A": 2 if candidate else 1, "B": 1},
        total_num_scheduled_tokens=3 if candidate else 2,
        scheduled_spec_decode_tokens={"A": [-1]} if candidate else {})
    ids = torch.tensor([12, 13, 22] if candidate else [12, 22])
    positions = torch.tensor([2, 3, 1] if candidate else [2, 1])
    with adapter.trace_target(runner, scheduler, ids, positions, NS(name="FULL")):
        capture("model.layers.0.mlp.experts", torch.tensor([[1,2], [3,4], [5,6]])[:len(ids)])
    return ids, positions


def test_async_actual_inputs_ignore_placeholder_request_state_and_freeze_order(setup):
    runner, session = setup
    ids, positions = execute_target(runner, session, True)
    ids[:] = 99
    positions[:] = 99
    runner.input_batch.req_ids.reverse()
    output = NS(sampled_token_ids=torch.tensor([[13, 14], [23, -1]]))
    fn = adapter.trace_sample(lambda self, logits, meta: output)
    assert fn(runner, None, NS(num_draft_tokens=[1, 0], draft_token_ids=torch.tensor([13]))) is output
    recorded = events(session)
    rows = recorded[0]["routes"]
    assert [(r["request_id"], r["token_position"], r["token_id"]) for r in rows] == [("A",2,12),("A",3,13),("B",1,22)]
    assert recorded[1]["decisions"][0]["request_id"] == "A"
    assert recorded[1]["decisions"][0]["accepted"]
    assert recorded[1]["decisions"][0]["producer_link"] == "unresolved"


@pytest.mark.parametrize("accepted", [True, False])
def test_sampler_links_real_candidate_position_and_keeps_rejected_route(setup, accepted):
    runner, session = setup
    session.pending["A"] = dict(round="round-previous", positions=[3], tokens=[13])
    execute_target(runner, session, True)
    samples = [[13,14],[23,-1]] if accepted else [[15,-1],[23,-1]]
    fn = adapter.trace_sample(lambda *args: NS(sampled_token_ids=torch.tensor(samples)))
    fn(runner, None, NS(num_draft_tokens=[1,0], draft_token_ids=torch.tensor([13])))
    recorded = events(session)
    assert len(recorded[0]["routes"]) == 3
    decision = recorded[1]["decisions"][0]
    assert decision["accepted"] == accepted
    assert decision["candidate_round"] == "round-previous"


def test_eager_mtp_proposal_outputs_link_to_actual_shifted_input(setup):
    runner, session = setup
    proposer = NS(runner=runner, num_speculative_tokens=1, query_start_loc_group=[torch.tensor([0,1,2])])
    kwargs = dict(input_ids=torch.tensor([14,23]), positions=torch.tensor([2,1]))
    def original(self, scheduler_output=None):
        with adapter.trace_mtp_forward(self, kwargs, 2, 2, torch.tensor([0,1]), 0):
            capture("draft_model.mtp.0.mlp.experts", torch.tensor([[1,2],[3,4]]))
        return torch.tensor([[15],[24]])
    result = adapter.trace_proposal(original)(proposer, scheduler_output=object())
    assert result.tolist() == [[15],[24]]
    recorded = events(session)
    assert recorded[0]["routes"][0]["input_sequence_position"] == 3
    assert recorded[1]["candidates"][0]["token_position"] == 4
    assert session.pending["A"]["positions"] == [4]


def test_privacy_removes_input_and_candidate_token_values(setup):
    runner, session = setup
    session.cfg.update(save_token_ids=False, save_expert_weights=False)
    execute_target(runner, session)
    rows = events(session)[0]["routes"]
    assert all("token_id" not in r and "expert_weights" not in r for r in rows)


def test_missing_layer_fails_and_records_status(setup):
    runner, session = setup
    runner.vllm_config.model_config.hf_config.num_hidden_layers = 2
    with pytest.raises(ValueError, match="incomplete"):
        execute_target(runner, session)
    assert json.loads((session.root/"status.json").read_text())["state"] == "failed"


def test_limit_stops_capture_without_stopping_original_forward(setup):
    runner, session = setup
    session.cfg["max_executions"] = 1
    execute_target(runner, session)
    execute_target(runner, session)
    assert len(events(session)) == 1 and session.capped
    assert json.loads((session.root/"status.json").read_text())["state"] == "limit_reached"


def test_summary_hash_verification_and_zero_expert_counts(setup):
    runner, session = setup
    execute_target(runner, session)
    session.close()
    tool = script("scripts/validation/analyze_ascend_023_debug.py")
    report = tool.analyze(session.root.parent.parent, expected_workers=1)
    assert report["expert_choices"] == dict(target=4, mtp=0, combined=4)
    assert any(r["count"] == 0 for r in report["counts"])
    with (session.root/"events.jsonl").open("r+b") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="hash mismatch"):
        tool.analyze(session.root.parent.parent, expected_workers=1)


def test_runtime_accepts_requested_flags_without_mutation():
    runner = runtime()
    adapter.check_runtime(runner)
    assert runner.vllm_config.scheduler_config.async_scheduling
    assert runner.ascend_config.multistream_overlap_shared_expert
    assert runner.ascend_config.ascend_compilation_config.enable_npugraph_ex
    runner.vllm_config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        adapter.check_runtime(runner)


def test_disabled_hooks_preserve_original_return(monkeypatch):
    monkeypatch.setattr(adapter, "_session", None)
    marker = object()
    fn = adapter.trace_sample(lambda *args: marker)
    assert fn(None, None, None) is marker
    with adapter.trace_target(None, None, None, None, None):
        pass


def test_profile_forced_balancing_is_ignored_outside_real_forward_or_graph_capture(setup):
    _,session=setup
    adapter.capture_routes(None,None,None,None,0,forbidden=True)
    assert events(session)==[]


def test_checked_stage_transforms_reject_unknown_sources(tmp_path):
    tool = script("scripts/tools/stage_ascend_async_graph.py")
    with pytest.raises(ValueError, match="source mismatch"):
        tool.stage(tmp_path/"vllm", tmp_path/"ascend", tmp_path/"overlay")
    assert not (tmp_path/"overlay").exists()


def test_byte_limit_is_explicit_partial_result(setup):
    runner, session = setup
    session.cfg["max_file_bytes"] = 1
    execute_target(runner, session)
    assert events(session) == [] and session.capped
    tool=script("scripts/validation/analyze_ascend_023_debug.py")
    with pytest.raises(ValueError, match="limit_reached"):
        tool.analyze(session.root.parent.parent,expected_workers=1)
    report=tool.analyze(session.root.parent.parent,expected_workers=1,allow_incomplete=True)
    assert report["limits_reached"]==1 and not report["capture_observed"]


def test_sampler_mismatched_actual_candidate_fails_instead_of_guessing(setup):
    runner,session=setup
    execute_target(runner,session,True)
    fn=adapter.trace_sample(lambda *args:NS(sampled_token_ids=torch.tensor([[99,-1],[23,-1]])))
    with pytest.raises(ValueError,match="actual input"):
        fn(runner,None,NS(num_draft_tokens=[1,0],draft_token_ids=torch.tensor([98])))
    assert json.loads((session.root/"status.json").read_text())["state"]=="failed"


def test_summary_refuses_missing_workers_by_default(setup):
    runner,session=setup
    execute_target(runner,session)
    session.close()
    tool=script("scripts/validation/analyze_ascend_023_debug.py")
    with pytest.raises(ValueError,match="expected 16 workers"):
        tool.analyze(session.root.parent.parent)


def test_shutdown_with_missing_verification_is_explicitly_incomplete(setup):
    runner,session=setup
    execute_target(runner,session,True)
    session.close()
    assert json.loads((session.root/"status.json").read_text())["state"]=="incomplete"
    tool=script("scripts/validation/analyze_ascend_023_debug.py")
    with pytest.raises(ValueError,match="incomplete"):
        tool.analyze(session.root.parent.parent,expected_workers=1)
    report=tool.analyze(session.root.parent.parent,expected_workers=1,allow_incomplete=True)
    assert report["pending_verification_executions"]==1
    assert not report["candidate_attribution_complete"]
