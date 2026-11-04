"""Dynamic-K validation executes the real patched autoregressive Python loop."""
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from moe_trace.adapters import ascend_v4_mtp as adapter
from moe_trace.distributed.decode import Worker
from moe_trace.distributed.v4_mtp_cli import collect, analyze, route_decisions
from moe_trace.errors import ConfigError, IntegrityError
from test_v4_debug import script, method
from test_v4_mtp import setup, actual_block


def scenario(tmp_path, monkeypatch, k, accepted, scheduled=None, fault=None):
    vllm, ascend = os.environ.get("MOE_V4_VLLM_SOURCE"), os.environ.get("MOE_V4_ASCEND_SOURCE")
    if not vllm or not ascend:
        pytest.skip("fixed V4 source repositories required")
    stage=script("scripts/tools/stage_ascend_v4_mtp.py")
    overlay=stage.stage(Path(vllm)/"vllm",Path(ascend)/"vllm_ascend",tmp_path/"overlays")
    script("scripts/deployment/ascend_v4_mtp_1p1d.py").verify_overlay(overlay)
    source=overlay/"vllm_ascend/spec_decode/llm_base_proposer.py"
    tree=ast.parse(source.read_text())
    runner,target,draft,current,cfg=setup(tmp_path,monkeypatch)
    runner.vllm_config.speculative_config.num_speculative_tokens=k
    runner.vllm_config.model_config.max_model_len=100 if fault!="max_length" else 2
    seq={"B":[20,21],"A":[10,11,12]}
    runner.requests={r:NS(num_prompt_tokens=1 if r=="B" else 2,
        get_token_id=lambda p,_r=r:seq[_r][p]) for r in seq}
    runner.input_batch=NS(req_ids=["B","A"],num_computed_tokens_cpu=[1,2])
    runner.discard_request_indices,runner.num_discarded_requests=NS(np=torch.tensor([],dtype=torch.int32)),0
    prepare=type("PrepareAndFinalizeWithAll2All",(),{})()
    def model(layers, ids):
        values=torch.tensor([[0,2]],dtype=torch.int32).repeat(len(ids),1)
        for layer in layers:
            adapter.capture_routes(layer,values,torch.ones(len(ids),2),prepare,NS(),NS(),4)
            adapter.capture_consumer(layer,values)
        return torch.zeros(len(ids),2)
    initial=NS(num_scheduled_tokens={"A":1,"B":1},total_num_scheduled_tokens=2,scheduled_spec_decode_tokens={})
    with adapter.trace_target(runner,initial,torch.tensor([21,12]),torch.tensor([1,2]),NS(name="NONE")):
        model(target,torch.tensor([21,12]))
    seq["B"].append(22)
    seq["A"].append(13)
    p=NS(runner=runner,method="mtp",num_speculative_tokens=k,needs_extra_input_slots=False,use_cuda_graph=False,
        query_start_loc_group=[torch.tensor([0,1,2])],input_ids=torch.zeros(8,dtype=torch.long),
        hidden_states=torch.zeros(8,2),positions=torch.zeros(8,dtype=torch.long),uses_mrope=False,
        supports_mm_inputs=False,pass_hidden_states_to_model=True,use_compress=False,vllm_config=runner.vllm_config)
    def set_positions(n, positions):
        p.positions[:n].copy_(positions)
        if fault=="next_input" and getattr(p,"calls",0)>0:
            p.input_ids[0]+=1
    p._set_positions=set_positions
    p._get_positions=lambda n:p.positions[:n]
    class Model:
        def __call__(self,**kw):
            p.calls=getattr(p,"calls",0)+1
            return model(draft,kw["input_ids"])
        def compute_logits(self,hidden):
            return torch.tensor([23+p.calls-1,14+p.calls-1])
    p.model=Model()
    p._sample_draft_from_logits=lambda logits,metadata:(logits.clone(),None)
    set_inputs=method(source,"AscendSpecDecodeBaseProposer","set_inputs_first_pass",dict(torch=torch))
    n,selected,_,_=set_inputs(p,torch.tensor([21,12]),torch.tensor([22,13]),torch.tensor([1,2]),
        torch.zeros(2,2),torch.tensor([0,1]),NS(query_start_loc=torch.tensor([0,1,2])),None)
    @adapter.trace_proposal_call
    def proposal(proposer,**kw):
        values=dict(self=p,model_kwargs=dict(input_ids=p.input_ids[:6],positions=p.positions[:6]),
            num_tokens=n,batch_size=2,token_indices_to_sample=selected,trace_draft=adapter.trace_draft)
        actual_block(source,"trace_draft",values)()
        tokens=torch.zeros(k,2,dtype=torch.long)
        tokens[0],_=p._sample_draft_from_logits(p.model.compute_logits(torch.zeros(2,2)),None)
        if k>1:
            loops=[node for node in ast.walk(tree) if isinstance(node,ast.For) and isinstance(node.target,ast.Name)
                   and node.target.id=="draft_index" and isinstance(node.iter,ast.Call)
                   and ast.unparse(node.iter)=="range(self.num_speculative_tokens - 1)"]
            assert len(loops)==1
            env=dict(torch=torch,nn=torch.nn,self=p,draft_token_ids_tensor=tokens,batch_size=2,
                positions=p.positions[selected].clone(),hidden_states=torch.zeros(2,2),
                input_batch_size=6,token_indices_to_sample=torch.arange(2),
                get_forward_context=lambda:NS(attn_metadata=None),multi_steps_attn_metadata=[],
                _split_draft_outputs=lambda out:(out,out),lmhead_tp_enable=lambda:False,
                sampling_metadata=None,draft_probs_list=None)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[loops[0]],type_ignores=[])),str(source),"exec"),env)
        output=tokens.swapaxes(0,1)
        if fault=="matrix":
            output[0,0]+=1
        returns=[node for node in ast.walk(tree) if isinstance(node,ast.Return) and isinstance(node.value,ast.Call)
                 and isinstance(node.value.func,ast.Name) and node.value.func.id=="record_proposals"
                 and isinstance(node.value.args[2],ast.Constant)]
        if k==1:
            return adapter.record_proposals(p,output,selected)
        assert len(returns)==1
        wrapper=ast.parse("def returned():\n    pass\n")
        wrapper.body[0].body=[returns[0]]
        env=dict(self=p,draft_token_ids=output,record_proposals=adapter.record_proposals)
        exec(compile(ast.fix_missing_locations(wrapper),str(source),"exec"),env)
        return env["returned"]()
    if fault=="missing_step":
        @contextmanager
        def skipped(*args):
            yield
        monkeypatch.setattr(adapter,"trace_draft_next",skipped)
    result=proposal(p,scheduler_output=initial,num_speculative_tokens=k)
    assert p.calls==k and result.tolist()==[list(range(23,23+k)),list(range(14,14+k))]
    m=k if scheduled is None else scheduled
    # Invert request order, preserve scheduler's different insertion order.
    runner.input_batch=NS(req_ids=["A","B"],num_computed_tokens_cpu=[3,2])
    a,b=list(range(14,14+m)),list(range(23,23+m))
    ids=torch.tensor([13,*a,22,*b])
    pos=torch.tensor([*range(3,4+m),*range(2,3+m)])
    schedule=NS(num_scheduled_tokens={"B":m+1,"A":m+1},total_num_scheduled_tokens=len(ids),
                scheduled_spec_decode_tokens={"B":b,"A":a})
    with adapter.trace_target(runner,schedule,ids,pos,NS(name="NONE")):
        model(target,ids)
    count_a,count_b=accepted
    samples=[a[:count_a]+[99],b[:count_b]+[98]]
    samples=[row+[-1]*(m+1-len(row)) for row in samples]
    adapter.record_verification(runner,NS(num_draft_tokens=[m,m],draft_token_ids=torch.tensor(a+b)),
                                NS(sampled_token_ids=torch.tensor(samples)))
    current.close()
    for dp in range(1,16):
        worker=Worker(adapter.PROFILE,"cpu","D0",f"dp-{dp}-tp-0",dp,0,dp,dp)
        adapter.Session(runner,cfg,worker).close()
    run=tmp_path/"cpu"
    (run/"deployment.json").write_text(json.dumps(dict(profile=adapter.PROFILE,run_id="cpu",node_id="D0",
        expected_workers=[f"dp-{dp}-tp-0" for dp in range(16)])))
    return run


@pytest.mark.parametrize("k,accepted",[(k,a) for k in (1,2,3) for a in range(k+1)]+[(7,3)])
def test_dynamic_k_real_autoregressive_loop(tmp_path,monkeypatch,k,accepted):
    run=scenario(tmp_path,monkeypatch,k,(accepted,accepted))
    joined,candidates=collect(run)
    totals,counts,_,_=analyze(joined)
    assert totals==dict(target=8*k+16,mtp=4*k,combined=12*k+16)
    assert len(candidates)==2*k and sum(counts.values())==12*k+16
    assert sum(c["state"]=="accepted" for c in candidates)==2*accepted
    assert sum(c["state"]=="rejected" for c in candidates)==(2 if accepted<k else 0)
    assert sum(c["state"]=="not_adopted" for c in candidates)==2*max(0,k-accepted-1)
    decisions=route_decisions(joined)
    assert len(decisions)==2*k
    assert {int(c["candidate_node"]) for c in candidates}==set(range(k))


def test_dynamic_k_clipped_schedule_and_different_acceptance(tmp_path,monkeypatch):
    run=scenario(tmp_path,monkeypatch,3,(0,2),scheduled=2)
    joined,candidates=collect(run)
    assert analyze(joined)[0]==dict(target=32,mtp=12,combined=44)
    assert sorted(c["state"] for c in candidates)==["accepted","accepted","not_adopted","rejected","unverified","unverified"]


@pytest.mark.parametrize("fault",["next_input","missing_step","matrix","max_length"])
def test_dynamic_k_rejects_incomplete_or_wrong_steps(tmp_path,monkeypatch,fault):
    with pytest.raises((IntegrityError,ConfigError)):
        scenario(tmp_path,monkeypatch,3,(1,2),fault=fault)
    status=tmp_path/"cpu/dp-0-tp-0/trace/worker-00000/status.json"
    assert json.loads(status.read_text())["state"]=="failed"


def test_runtime_k_zero_skips_drafting_and_bounds_k(tmp_path,monkeypatch):
    runner,_,_,current,_=setup(tmp_path,monkeypatch)
    runner.vllm_config.speculative_config.num_speculative_tokens=3
    @adapter.trace_proposal_call
    def proposal(p,**kwargs):
        return torch.empty(2,0,dtype=torch.long)
    p=NS(runner=runner)
    assert proposal(p,scheduler_output=object(),num_speculative_tokens=0).shape==(2,0)
    assert not current.writer.failed and current.ledger_counts["candidates"]==0
    with pytest.raises(ConfigError):
        proposal(p,scheduler_output=object(),num_speculative_tokens=4)


def test_launcher_accepts_configured_k():
    launcher=script("scripts/deployment/ascend_v4_mtp_1p1d.py")
    for k in (1,2,3,7):
        for task in launcher.commands("D","/model","192.0.2.1",num_speculative_tokens=k):
            argv=task["command"]
            assert json.loads(argv[argv.index("--speculative-config")+1])["num_speculative_tokens"]==k
