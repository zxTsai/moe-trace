"""CPU oracle and timer/retention checks; no NPU performance claims."""
import json
import threading
import time
from types import SimpleNamespace as NS

import pytest
import torch

from moe_trace.adapters import ascend_023_debug as adapter
from moe_trace.adapters.ascend_load_summary import SummarySession, local_indices, mtp_masks
from test_ascend_023_debug import setup, capture, script


@pytest.fixture
def summary(setup,monkeypatch):
    runner,old=setup
    cfg=dict(old.cfg,mode="summary",run_id="summary",save_token_ids=False,save_expert_weights=False,
             interval_seconds=5,max_files=3,max_file_bytes=16384)
    session=SummarySession(runner,cfg,dict(dp=0,tp=0,ep=0),start_thread=False)
    monkeypatch.setattr(adapter,"_session",session)
    return runner,session


def records(session):
    return [json.loads(line) for line in (session.root/"load.jsonl").read_text().splitlines()]


def target(runner,ids=None):
    positions=torch.tensor([1,2,3,1,0])
    scheduler=NS(num_scheduled_tokens={"A":3,"B":1},total_num_scheduled_tokens=4,
                 scheduled_spec_decode_tokens={"A":[-1]})
    with adapter.trace_target(runner,scheduler,None,positions,NS(name="FULL")):
        capture("model.layers.0.mlp.experts",ids if ids is not None else torch.tensor([[7,6],[1,2],[1,3],[2,3],[7,6]]))


def test_continuous_device_counts_match_oracle_without_token_host_reads(summary,monkeypatch):
    runner,session=summary
    monkeypatch.setattr(adapter,"host",lambda *args:pytest.fail("per-token host conversion"))
    monkeypatch.setattr(adapter,"synchronize",lambda:pytest.fail("per-forward device synchronize"))
    target(runner)
    target(runner)
    assert not records(session)  # Only the window flush writes data.
    session.flush()
    event=records(session)[0]
    assert event["sampling"]=="continuous" and event["executions"]==dict(target=2,mtp=0)
    assert event["expert_choices"]==dict(target=12,mtp=0,combined=12)
    layers={row["kind"]:row for row in event["layers"]}
    assert layers["target_verify"]["expert_selection_counts"]==[0,4,2,2,0,0,0,0]
    assert layers["target_verify"]["observed_input_tokens"]==4
    assert layers["target_decode"]["expert_selection_counts"]==[0,0,2,2,0,0,0,0]
    assert not session.counts and not session.tokens
    session.flush()
    assert records(session)[1]["expert_choices"]["combined"]==0


def test_mtp_masks_exclude_discard_rejected_tail_prefill_and_padding(summary):
    runner,session=summary
    runner.num_discarded_requests=1
    runner.discard_request_indices.np=torch.tensor([1])
    proposer=NS(runner=runner,query_start_loc_group=[torch.tensor([0,3,5])])
    kwargs=dict(positions=torch.tensor([0,1,2,0,1,0]))
    mask=mtp_masks(proposer,kwargs,5,2,torch.tensor([1,4]),0)["mtp_draft"]
    assert mask.tolist()==[False,True,False,False,False,False]


def test_periodic_thread_outputs_while_idle_and_flushes_shutdown(setup):
    runner,old=setup
    cfg=dict(old.cfg,run_id="timer",interval_seconds=.05,max_files=3,max_file_bytes=16384)
    session=SummarySession(runner,cfg,dict(dp=0,tp=0,ep=0))
    # Synchronize on a real flush, not an assumed elapsed sleep.
    deadline=time.monotonic()+2
    try:
        while session.window<2 and time.monotonic()<deadline:
            session.stop.wait(.01)
        assert session.window>=2
    finally:
        session.close()
    assert records(session)[-1]["partial"]
    assert json.loads((session.root/"status.json").read_text())["state"]=="closed"
    assert not session.thread.is_alive()


def test_rotation_is_bounded_and_reports_history_eviction(summary):
    runner,session=summary
    session.cfg.update(max_file_bytes=4000,max_files=2)
    for _ in range(16):
        target(runner)
        session.flush()
    files=list(session.root.glob("load*.jsonl"))
    assert len(files)==2 and all(p.stat().st_size<=4000 for p in files)
    status=json.loads((session.root/"status.json").read_text())
    assert status["rotations"]>=2 and not status["full_history_retained"]
    all_rows=[json.loads(line) for p in files for line in p.read_text().splitlines()]
    assert max(r["window_id"] for r in all_rows)==15


def test_invalid_id_and_oversized_window_fail_explicitly(summary):
    runner,session=summary
    target(runner,torch.tensor([[1,2],[1,99],[1,3],[2,3],[1,2]]))
    with pytest.raises(ValueError,match="invalid logical expert"):
        session.flush()
    session.counts.clear()
    session.tokens.clear()
    session.cfg["max_file_bytes"]=1
    with pytest.raises(ValueError,match="exceeds"):
        session.close()
    assert json.loads((session.root/"status.json").read_text())["state"]=="failed"


def test_worker_surfaces_timer_failure(summary):
    runner,session=summary
    session.fail(OSError("disk unavailable"))
    with pytest.raises(RuntimeError,match="failed"):
        target(runner)
    with pytest.raises(RuntimeError,match="failed"):
        session.close()


def test_allgather_maps_local_rows_with_graph_padding():
    context=NS(dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([2,1])))
    indices=local_indices("PrepareAndFinalizeWithAllGather",3,4,1,context,NS(),"cpu")
    assert indices.tolist()==[2]
    with pytest.raises(ValueError,match="layout"):
        local_indices("PrepareAndFinalizeWithAllGather",3,5,1,context,NS(),"cpu")


@pytest.mark.parametrize("field,value",[("interval_seconds",0),("interval_seconds",float('nan')),
    ("max_files",0),("max_files",True),("max_file_bytes",False),("node_id","../bad")])
def test_summary_config_rejects_invalid_values(tmp_path,monkeypatch,field,value):
    cfg=dict(mode="summary",output_root=str(tmp_path),run_id="summary",node_id="D",
             interval_seconds=5,max_files=4,max_file_bytes=4096)
    cfg[field]=value
    path=tmp_path/"config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setenv(adapter.ENV,str(path))
    with pytest.raises(ValueError):
        adapter.config()


def test_only_one_compact_host_copy_per_window(summary,monkeypatch):
    runner,session=summary
    original=torch.Tensor.cpu
    calls=[]
    def observed(self,*args,**kwargs):
        calls.append(tuple(self.shape))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,"cpu",observed)
    target(runner)
    target(runner)
    assert calls==[]
    session.flush()
    assert len(calls)==1 and len(calls[0])==1


def test_load_analysis_counts_retained_windows_and_rejects_corruption(summary):
    runner,session=summary
    target(runner)
    session.close()
    analyze=script("scripts/validation/analyze_ascend_load.py").analyze
    report=analyze(session.root.parent.parent,expected_workers=1)
    assert report["aggregation"]=="retained_windows_only"
    assert report["expert_choices"]==dict(target=6,mtp=0,combined=6)
    assert any(row["count"]==0 for row in report["counts"])
    path=session.root/"load.jsonl"
    rows=records(session)
    rows[0]["layers"][0]["expert_selection_counts"][0]+=1
    path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    with pytest.raises(ValueError,match="Top-K"):
        analyze(session.root.parent.parent,expected_workers=1)


def test_background_write_error_is_saved_and_propagated(setup):
    runner,old=setup
    cfg=dict(old.cfg,run_id="write-failure",interval_seconds=.05,max_files=2,max_file_bytes=16384)
    session=SummarySession(runner,cfg,dict(dp=0,tp=0,ep=0))
    def broken(record):
        raise OSError("injected disk write failure")
    session._write=broken
    assert session.stop.wait(2)
    with pytest.raises(RuntimeError,match="failed"):
        session.close()
    assert json.loads((session.root/"status.json").read_text())["state"]=="failed"


def test_npu_transfer_waits_for_producer_event_on_correct_device(summary,monkeypatch):
    _,session=summary
    calls=[]
    session.device=NS(type="npu",index=3)
    session.event=NS(synchronize=lambda:calls.append("event_complete"))
    monkeypatch.setattr(torch,"npu",NS(set_device=lambda device:calls.append(("device",device.index))),raising=False)
    session.flush()
    assert calls==[("device",3),"event_complete"]


def test_summary_initialize_arms_guard_without_debug_barrier(setup,monkeypatch,tmp_path):
    import sys
    from types import ModuleType
    from moe_trace.adapters import ascend_graph_bridge as bridge
    runner,_=setup
    adapter.close()
    config=dict(mode="summary",output_root=str(tmp_path),run_id="initialized",node_id="D",
                interval_seconds=5,max_files=4,max_file_bytes=16384)
    path=tmp_path/"initialize.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv(adapter.ENV,str(path))
    group=ModuleType("vllm.distributed")
    group.get_dp_group=lambda:NS(world_size=16,rank_in_group=0)
    group.get_tp_group=lambda:NS(world_size=1,rank_in_group=0)
    group.get_ep_group=lambda:NS(world_size=16,rank_in_group=0)
    monkeypatch.setitem(sys.modules,"vllm.distributed",group)
    monkeypatch.setattr(torch,"npu",NS(current_device=lambda:0),raising=False)
    adapter.initialize(runner)
    session=adapter._session
    assert isinstance(session,SummarySession) and session.thread.is_alive()
    assert bridge._before_snapshot is None and bridge._call_guard() is session.lock
    target(runner)
    adapter.close()
    assert session.closed and not session.thread.is_alive()
    assert records(session)[-1]["partial"]
