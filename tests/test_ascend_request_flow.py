"""Per-request flow identity/mapping oracle on CPU, not real network traffic."""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from moe_trace.adapters import ascend_023_debug as adapter
from moe_trace.adapters.ascend_request_flow import RequestFlowSession,owners_from_maps,placements_from_runner
from moe_trace.io import atomic_json
from test_ascend_023_debug import setup,capture,script


def placements():
    return {name:dict(scope=scope,num_experts=8,owners=[3,1,3,2,0,2,1,0])
        for scope,name in [("target","model.layers.0.mlp.experts"),("mtp","draft_model.mtp.0.mlp.experts")]}


@pytest.fixture
def flow(setup,monkeypatch):
    runner,old=setup
    cfg=dict(old.cfg,mode="request_flow",run_id="flow",save_token_ids=False,save_expert_weights=False,
             interval_seconds=5,max_files=3,max_requests_per_window=128)
    session=RequestFlowSession(runner,cfg,dict(dp=0,tp=0,ep=0,device=4),placements=placements(),start_thread=False)
    monkeypatch.setattr(adapter,"_session",session)
    for rank in range(1,4):
        folder=session.root.parent/f"dp-{rank}-tp-0"
        folder.mkdir()
        atomic_json(folder/"worker.json",dict(ep_rank=rank,dp_rank=rank,tp_rank=0,
            node_id="D",hostname="cpu-test",device_index=rank+4,identity_status="resolved"))
    return runner,session


def target(runner,reqs=None):
    reqs=reqs or ["A","B"]
    runner.input_batch.req_ids=reqs
    scheduler=NS(num_scheduled_tokens={r:1 for r in reqs},total_num_scheduled_tokens=len(reqs),scheduled_spec_decode_tokens={})
    with adapter.trace_target(runner,scheduler,None,torch.tensor([2 if r=="A" else 1 for r in reqs]),NS(name="NONE")):
        capture("model.layers.0.mlp.experts",torch.tensor([[1,2] if r=="A" else [3,4] for r in reqs]))


def records(session):
    return [json.loads(line) for line in (session.root/"load.jsonl").read_text().splitlines()]


def test_two_requests_route_to_actual_permuted_owners_without_mixing(flow):
    runner,session=flow
    target(runner)
    target(runner,["B","A"])
    session.flush()
    event=records(session)[0]
    assert event["type"]=="request_flow_summary" and "layers" not in event
    requests={r["request_id"]:r for r in event["requests"]}
    assert set(requests)=={"A","B"}
    assert requests["A"]["source"]["device_index"]==4
    a=requests["A"]["layers"][0]
    assert a["observed_input_tokens"]==2
    assert {d["destination"]["ep_rank"]:d["expert_selection_count"] for d in a["destinations"]}=={1:2,3:2}
    assert {d["destination"]["device_index"] for d in a["destinations"]}=={5,7}
    assert event["expert_choices"]["combined"]==8
    target(runner,["A"])
    session.flush()
    next_request=records(session)[1]["requests"][0]
    assert next_request["request_id"]=="A" and next_request["layers"][0]["observed_input_tokens"]==1


def test_same_device_two_experts_are_two_choices_not_two_network_messages(flow):
    runner,session=flow
    session.placements["model.layers.0.mlp.experts"]["owners"][2]=1
    target(runner,["A"])
    session.flush()
    destinations=records(session)[0]["requests"][0]["layers"][0]["destinations"]
    assert len(destinations)==1 and destinations[0]["expert_selection_count"]==2
    assert len(destinations[0]["expert_counts"])==2


def test_missing_endpoint_is_explicitly_unresolved(flow):
    runner,session=flow
    (session.root.parent/"dp-3-tp-0/worker.json").unlink()
    target(runner,["A"])
    session.flush()
    destination=records(session)[0]["requests"][0]["layers"][0]["destinations"][-1]["destination"]
    assert destination==dict(ep_rank=3,identity_status="unresolved")


def test_request_cap_early_flushes_without_dropping_counts(flow):
    runner,session=flow
    session.cfg["max_requests_per_window"]=1
    target(runner,["A"])
    target(runner,["B"])
    session.flush()
    rows=records(session)
    assert rows[0]["partial"] and rows[0]["flush_reason"]=="request_limit"
    assert [r["requests"][0]["request_id"] for r in rows]==["A","B"]
    assert sum(r["expert_choices"]["combined"] for r in rows)==4


def test_bigger_batch_than_cap_fails_instead_of_dropping_request(flow):
    runner,session=flow
    session.cfg["max_requests_per_window"]=1
    with pytest.raises(ValueError,match="one batch"):
        target(runner)
    # Explicit failed close is checked here; fixture's second close is idempotent.
    with pytest.raises(RuntimeError):
        session.close()


@pytest.mark.parametrize("maps",[[[0,1],[-1,0]],[[0,-1],[-1,-1]]])
def test_ambiguous_and_missing_owners_are_rejected(maps):
    with pytest.raises(ValueError,match="exactly one"):
        owners_from_maps(maps,2)


def test_mapping_provider_is_checked_against_actual_module():
    local=torch.tensor([-1,0,-1,1])
    target_layer=NS(layer_name="model.layers.0.mlp.experts",_expert_map=local,moe_config=NS(),
                    global_expert_map=None)
    draft_layer=NS(**dict(target_layer.__dict__,layer_name="draft_model.mtp.0.mlp.experts"))
    runner=NS(model=NS(named_modules=lambda:[("target",target_layer)]),
        drafter=NS(model=NS(named_modules=lambda:[("draft",draft_layer)])),
        vllm_config=NS(model_config=NS(hf_config=NS(num_hidden_layers=1))))
    def provider(size,rank,n):
        return 2,torch.tensor([0,-1,1,-1]) if rank==0 else local,None
    result=placements_from_runner(runner,1,2,provider)
    assert result[target_layer.layer_name]["owners"]==[0,1,0,1]
    target_layer._expert_map=torch.tensor([0,-1,1,-1])
    with pytest.raises(ValueError,match="differs"):
        placements_from_runner(runner,1,2,provider)


def test_config_selects_request_flow(tmp_path,monkeypatch):
    cfg=dict(mode="request_flow",output_root=str(tmp_path),run_id="flow",node_id="D",
        interval_seconds=5,max_files=4,max_file_bytes=16384,max_requests_per_window=128)
    path=tmp_path/"config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv(adapter.ENV,str(path))
    assert adapter.config()["mode"]=="request_flow"
    cfg["max_requests_per_window"]=0
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        adapter.config()
