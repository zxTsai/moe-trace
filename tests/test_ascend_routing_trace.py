"""Ordered per-token source/destination routes, CPU-only collector contracts."""
import json
import logging
import time
from types import SimpleNamespace as NS

import pytest
import torch

from moe_trace.adapters import ascend_023_debug as adapter
from moe_trace.adapters.ascend_routing_trace import RoutingTraceSession
from moe_trace.io import atomic_json
from test_ascend_023_debug import setup,script

EXPERTS=[0,3,5,6,12,34]
DESTINATIONS=[7,1,10,3,4,2]


def placements():
    owners=[i%16 for i in range(64)]
    for expert,rank in zip(EXPERTS,DESTINATIONS):
        owners[expert]=rank
    return {name:dict(scope=scope,num_experts=64,owners=list(owners)) for scope,name in (
        ("target","model.layers.0.mlp.experts"),("target","model.layers.1.mlp.experts"),
        ("mtp","draft_model.mtp.0.mlp.experts"))}


@pytest.fixture
def routing(setup,monkeypatch):
    runner,old=setup
    runner.vllm_config.model_config.hf_config.num_hidden_layers=2
    cfg=dict(mode="routing_trace",output_root=old.cfg["output_root"],run_id="routes",node_id="D",
        interval_seconds=5,max_buffer_bytes=65536,max_file_bytes=262144,max_files=3,
        save_token_ids=True,save_expert_weights=False)
    session=RoutingTraceSession(runner,cfg,dict(dp=0,tp=0,ep=0,device=4),placements=placements(),start_thread=False)
    monkeypatch.setattr(adapter,"_session",session)
    for rank in range(1,16):
        folder=session.root.parent/f"source-dp-{rank:03d}"
        folder.mkdir()
        atomic_json(folder/"worker.json",dict(node_id="D",dp_rank=rank,tp_rank=0,ep_rank=rank,
            device_index=rank,hostname="cpu-test",identity_status="resolved"))
    return runner,session


def capture(name,values):
    ids=torch.tensor(values)
    adapter.capture_routes(NS(layer_name=name),ids,torch.ones_like(ids,dtype=torch.float32),
        type("PrepareAndFinalizeWithAll2All",(),{})(),64)


def target(runner,candidate=False,ids=None):
    runner.input_batch.req_ids=["A","B"]
    schedule=NS(num_scheduled_tokens={"A":2,"B":1},total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={"A":[-1]} if candidate else {})
    with adapter.trace_target(runner,schedule,torch.tensor(ids or [12,13,22]),torch.tensor([2,3,1]),NS(name="FULL")):
        capture("model.layers.0.mlp.experts",[EXPERTS]*3)
        capture("model.layers.1.mlp.experts",[list(reversed(EXPERTS))]*3)


def windows(session):
    return [json.loads(line) for line in (session.root/"routes.jsonl").read_text().splitlines()]


def test_round_request_token_layer_topk_order_is_preserved(routing):
    runner,session=routing
    initial_status=(session.root/"status.json").read_bytes()
    target(runner)
    target(runner)
    assert windows(session)==[]
    assert (session.root/"status.json").read_bytes()==initial_status  # no per-forward status fsync
    session.flush()
    record=windows(session)[0]
    assert session.root.name=="source-dp-000"
    assert [r["schedule_seq"] for r in record["rounds"]]==[1,2]
    executions=[r["executions"][0] for r in record["rounds"]]
    assert executions[0]["execution_id"]!=executions[1]["execution_id"]
    a=executions[0]["requests"][0]
    assert a["request_id"]=="A" and [t["token_position"] for t in a["tokens"]]==[2,3]
    assert [t["token_id"] for t in a["tokens"]]==[12,13]
    layers=a["tokens"][0]["layers"]
    assert layers[0]["expert_ids"]==EXPERTS and layers[0]["destination_ep_ranks"]==DESTINATIONS
    assert layers[1]["expert_ids"]==list(reversed(EXPERTS))
    assert layers[1]["destination_ep_ranks"]==list(reversed(DESTINATIONS))
    assert record["source"]["device_index"]==4
    assert record["device_table"]["7"]["device_index"]==7


def test_idle_has_no_window_and_no_periodic_info(routing,caplog,monkeypatch):
    runner,session=routing
    runner.input_batch.req_ids=[]
    monkeypatch.setattr(adapter,"synchronize",lambda:pytest.fail("idle capture synchronize"))
    with caplog.at_level(logging.INFO,logger="vllm.moe_trace.routing"):
        session.flush()
        with adapter.trace_target(runner,None,None,None,None):
            pass
        session.flush()
    assert windows(session)==[] and session.executions==0
    assert not caplog.records


def test_idle_sampler_and_proposer_keep_native_outputs_without_capture(routing,monkeypatch):
    runner,session=routing
    runner.input_batch.req_ids=[]
    monkeypatch.setattr(adapter,"synchronize",lambda:pytest.fail("idle synchronize"))
    monkeypatch.setattr(adapter,"host",lambda *args:pytest.fail("idle host conversion"))
    marker=object()
    assert adapter.trace_sample(lambda *args:marker)(runner,None,None) is marker
    @adapter.trace_proposal
    def proposal(proposer,scheduler_output=None):
        with adapter.trace_mtp_forward(proposer,None,0,0,None,0):
            pass
        return marker
    assert proposal(NS(runner=runner),scheduler_output=object()) is marker
    session.flush()
    assert windows(session)==[] and session.executions==0


def test_recorded_flush_is_also_quiet(routing,caplog):
    runner,session=routing
    target(runner)
    caplog.clear()
    with caplog.at_level(logging.INFO,logger="vllm.moe_trace.routing"):
        session.flush()
    assert not caplog.records


def test_capacity_flush_preserves_both_executions(routing):
    runner,session=routing
    target(runner)
    session.cfg["max_buffer_bytes"]=session.pending_bytes+100
    target(runner)
    assert windows(session)[0]["flush_reason"]=="buffer_limit"
    session.flush()
    records=windows(session)
    assert len(records)==2
    assert [r["rounds"][0]["schedule_seq"] for r in records]==[1,2]


def test_privacy_and_shutdown_partial_window(routing):
    runner,session=routing
    session.cfg["save_token_ids"]=False
    target(runner)
    session.close()
    record=windows(session)[0]
    assert record["partial"]
    tokens=record["rounds"][0]["executions"][0]["requests"][0]["tokens"]
    assert all("token_id" not in token for token in tokens)
    assert json.loads((session.root/"status.json").read_text())["state"]=="closed"


def test_shutdown_missing_verification_stays_incomplete(routing):
    runner,session=routing
    target(runner,candidate=True)
    session.close()
    assert json.loads((session.root/"status.json").read_text())["state"]=="incomplete"


def test_single_execution_buffer_overflow_fails_explicitly(routing):
    runner,session=routing
    session.cfg["max_buffer_bytes"]=1
    with pytest.raises(ValueError,match="one execution"):
        target(runner)
    with pytest.raises(RuntimeError):
        session.close()
    assert json.loads((session.root/"status.json").read_text())["state"]=="failed"


def test_trace_file_rotation_retains_complete_windows(routing):
    runner,session=routing
    target(runner)
    session.flush()
    size=(session.root/"routes.jsonl").stat().st_size
    session.cfg.update(max_files=2,max_file_bytes=size+100)
    for _ in range(5):
        target(runner)
        session.flush()
    files=list(session.root.glob("routes*.jsonl"))
    assert len(files)==2
    assert not json.loads((session.root/"status.json").read_text())["full_history_retained"]
    assert all(json.loads(line)["type"]=="routing_trace_window" for p in files for line in p.read_text().splitlines())


def test_timer_writes_data_but_does_not_create_empty_windows(routing):
    runner,session=routing
    session.cfg["interval_seconds"]=.05
    session.start_timer()
    target(runner)
    deadline=time.monotonic()+2
    while session.window==0 and time.monotonic()<deadline:
        session.stop.wait(.01)
    session.close()
    assert len(windows(session))==1 and not session.thread.is_alive()


def test_routing_config_is_strict(tmp_path,monkeypatch):
    cfg=dict(mode="routing_trace",output_root=str(tmp_path),run_id="r",node_id="D",interval_seconds=5,
        max_buffer_bytes=16384,max_file_bytes=65536,max_files=4,save_token_ids=False,save_expert_weights=False)
    path=tmp_path/"config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv(adapter.ENV,str(path))
    assert adapter.config()==cfg
    cfg["max_buffer_bytes"]=False
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        adapter.config()


def test_inspector_exports_raw_order_and_detects_mapping_corruption(routing):
    runner,session=routing
    target(runner)
    session.close()
    inspect=script("scripts/validation/inspect_ascend_routing_trace.py").analyze
    root=session.root.parent.parent
    report=inspect(root,expected_workers=1)
    assert report["expert_choices"]==dict(target=36,mtp=0,combined=36)
    window=inspect(root,source_dp=0,window_id=0)
    layer=window["rounds"][0]["executions"][0]["requests"][0]["tokens"][0]["layers"][0]
    assert layer["expert_ids"]==EXPERTS
    layer["destination_ep_ranks"][0]=0
    (session.root/"routes.jsonl").write_text(json.dumps(window)+"\n")
    with pytest.raises(ValueError,match="destination mapping"):
        inspect(root,expected_workers=1)


def test_repeated_token_ids_keep_distinct_positions(routing):
    runner,session=routing
    target(runner,ids=[12,12,22])
    session.flush()
    tokens=windows(session)[0]["rounds"][0]["executions"][0]["requests"][0]["tokens"]
    assert [(t["token_id"],t["token_position"],t["input_row"]) for t in tokens]==[(12,2,0),(12,3,1)]


def test_sampler_can_cross_output_window_without_losing_execution_link(routing):
    runner,session=routing
    target(runner,candidate=True)
    session.flush()
    adapter.trace_sample(lambda *args:NS(sampled_token_ids=torch.tensor([[13,14],[23,-1]])))(runner,None,
        NS(num_draft_tokens=[1,0],draft_token_ids=torch.tensor([13])))
    session.close()
    records=windows(session)
    assert len(records)==2 and records[1]["rounds"]==[]
    assert records[1]["sampler_events"][0]["execution_id"]==records[0]["rounds"][0]["executions"][0]["execution_id"]
    report=script("scripts/validation/inspect_ascend_routing_trace.py").analyze(session.root.parent.parent,expected_workers=1)
    assert report["sources"][0]["unresolved_auxiliary_execution_refs"]==[]


def test_main_and_mtp_share_round_but_have_distinct_execution_ids(routing):
    runner,session=routing
    target(runner)
    proposer=NS(runner=runner,num_speculative_tokens=1,query_start_loc_group=[torch.tensor([0,2,3])])
    kwargs=dict(input_ids=torch.tensor([13,14,23]),positions=torch.tensor([2,3,1]))
    @adapter.trace_proposal
    def proposal(self,scheduler_output=None):
        with adapter.trace_mtp_forward(self,kwargs,3,2,torch.tensor([1,2]),0):
            capture("draft_model.mtp.0.mlp.experts",[EXPERTS]*3)
        return torch.tensor([[15],[24]])
    proposal(proposer,scheduler_output=object())
    session.flush()
    rounds=windows(session)[0]["rounds"]
    assert len(rounds)==1 and rounds[0]["schedule_seq"]==1
    executions=rounds[0]["executions"]
    assert [e["scope"] for e in executions]==["target","mtp"]
    assert executions[0]["execution_id"]!=executions[1]["execution_id"]
    assert executions[1]["mtp_step"]==0
