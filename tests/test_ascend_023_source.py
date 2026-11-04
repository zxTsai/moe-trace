"""Execute checked vendor Python methods on CPU; hardware operations are mocks."""
import ast
from contextlib import ExitStack, contextmanager
from enum import Enum
import os
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from moe_trace.adapters import ascend_023_debug as adapter
from moe_trace.adapters.ascend_graph_bridge import arm_graph_capture, execution_routes
from test_ascend_023_debug import setup, script, events, capture


def method(path, class_name, name, namespace, decorators=False):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    if not decorators:
        node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def overlay(tmp_path):
    vllm, ascend = os.environ.get("MOE_VLLM_023_SOURCE"), os.environ.get("MOE_ASCEND_023_SOURCE")
    if not vllm or not ascend:
        pytest.skip("use scripts/validation/ascend_023_source_cpu.py with reference source paths")
    stage = script("scripts/tools/stage_ascend_async_graph.py")
    output=Path(stage.stage(Path(vllm)/"vllm", Path(ascend)/"vllm_ascend", tmp_path/"overlays"))
    reference=Path(__file__).resolve().parents[1]/"patches/ascend/deepseek-v4-023post1-async-graph/0001-debug-hooks.patch"
    assert (output/"debug-hooks.patch").read_bytes()==reference.read_bytes()
    # A successful stage still leaves the original sources matching the lock.
    assert stage.stage(Path(vllm)/"vllm",Path(ascend)/"vllm_ascend",None,True)["source_check"]=="passed"
    return output


def test_actual_runner_forward_w8a8_and_sampler(setup, overlay):
    runner, session = setup
    prepare = type("PrepareAndFinalizeWithAll2All", (), {})()
    observed = []
    def kernel(fused_experts_input):
        observed.append(fused_experts_input.topk_ids.clone())
        return fused_experts_input.hidden_states
    comm = NS(prepare_finalize=prepare, fused_experts=kernel)
    extra = NS(moe_comm_method=comm, moe_comm_type="ALLTOALL")
    def selector(**kw):
        values, ids = torch.topk(kw["router_logits"], kw["top_k"], dim=-1)
        return values, ids
    apply = method(overlay/"vllm_ascend/quantization/methods/w8a8_dynamic.py",
        "AscendW8A8DynamicFusedMoEMethod", "apply", dict(torch=torch,
        get_moe_num_logical_experts=lambda layer,n,**kw:n, select_experts=selector,
        _EXTRA_CTX=extra, get_ascend_config=lambda:NS(enable_fused_mc2=0),
        MoECommType=NS(FUSED_MC2="FUSED_MC2"), build_fused_experts_input=lambda **kw:NS(**kw)))
    layer = NS(layer_name="model.layers.0.mlp.experts", w13_weight=None, w2_weight=None,
               w13_weight_scale_fp32=None, w2_weight_scale=None, swiglu_limit=None)
    scheme = NS(multistream_overlap_gate=False, in_dtype=torch.float32, dynamic_eplb=False, quant_type="W8A8")
    original_output = torch.zeros(3, 4)
    def model(*args, **kwargs):
        return apply(scheme, layer, original_output,
                     torch.tensor([[0.,1.,2.,3.],[4.,3.,2.,1.],[0.,4.,1.,3.]]), 2, True, num_experts=4)
    runner._model_forward = model
    scheduler = NS(num_scheduled_tokens={"B":1,"A":2}, total_num_scheduled_tokens=3,
                   scheduled_spec_decode_tokens={"A":[-1]})
    env = dict(self=runner, scheduler_output=scheduler, input_ids=torch.tensor([12,13,22]),
        positions=torch.tensor([2,3,1]), cudagraph_mode=NS(name="FULL"),
        num_tokens_padded=3, intermediate_tensors=None, inputs_embeds=None,
        model_kwargs={}, trace_target=adapter.trace_target)
    tree = ast.parse((overlay/"vllm_ascend/worker/model_runner_v1.py").read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n,ast.With) and any(
        isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name)
        and i.context_expr.func.id=="trace_target" for i in n.items))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])), "actual-runner-block", "exec"),env)
    assert env["hidden_states"] is original_output
    rows=events(session)[0]["routes"]
    assert [r["expert_ids"] for r in rows] == observed[0].tolist()
    assert [(r["request_id"],r["input_row"]) for r in rows] == [("A",0),("A",1),("B",2)]
    sampler = method(overlay/"vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", "_sample",
        dict(lmhead_tp_enable=lambda:False))
    runner.input_batch.update_async_output_token_ids=lambda:None
    runner.input_batch.sampling_metadata=NS(top_k=None)
    output = NS(sampled_token_ids=torch.tensor([[13,14],[23,-1]]))
    runner.rejection_sampler=lambda *args:output
    assert adapter.trace_sample(sampler)(runner,None,NS(num_draft_tokens=[1,0],draft_token_ids=torch.tensor([13]))) is output
    assert events(session)[1]["decisions"][0]["accepted"]


@pytest.mark.parametrize("k", [1,2,3])
@pytest.mark.parametrize("summary_mode", [False,True])
def test_actual_mtp_first_pass_and_autoregressive_loop(setup, overlay, k, summary_mode,monkeypatch):
    runner, session = setup
    if summary_mode:
        from moe_trace.adapters.ascend_load_summary import SummarySession
        session=SummarySession(runner,dict(session.cfg,mode="summary",run_id="summary-source",
            save_token_ids=False,save_expert_weights=False,interval_seconds=5,max_files=4),
            dict(dp=0,tp=0,ep=0),start_thread=False)
        monkeypatch.setattr(adapter,"_session",session)
        monkeypatch.setattr(adapter,"synchronize",lambda:pytest.fail("summary must not synchronize per forward"))
        monkeypatch.setattr(adapter,"host",lambda *args:pytest.fail("summary must not read per-token host data"))
    runner.vllm_config.model_config.max_model_len=100
    runner.vllm_config.speculative_config.num_speculative_tokens=k
    path=overlay/"vllm_ascend/spec_decode/llm_base_proposer.py"
    proposer=NS(runner=runner,method="mtp",num_speculative_tokens=k,needs_extra_input_slots=False,
        input_ids=torch.zeros(8,dtype=torch.long),positions=torch.zeros(8,dtype=torch.long),
        hidden_states=torch.zeros(8,4),query_start_loc_group=[torch.tensor([0,1,2])],
        uses_mrope=False,uses_xdrope_dim=0,draft_uses_xdrope_dim=0,pass_hidden_states_to_model=True,
        _share_mtp_indices=False,pcp_size=1,parallel_drafting=False,arange=torch.arange(8),
        device="cpu",use_cuda_graph=False,supports_mm_inputs=False,vllm_config=runner.vllm_config)
    proposer._set_positions=lambda n,p:proposer.positions[:n].copy_(p)
    proposer._get_positions=lambda n:proposer.positions[:n]
    proposer.model_returns_tuple=lambda:False
    proposer.maybe_pad_and_reduce=lambda h,p:(h,p)
    proposer.maybe_all_gather_and_unpad=lambda last,pos,hidden:(last,pos,hidden)
    class Model:
        calls=0
        def __call__(self,**kw):
            self.calls+=1
            capture("draft_model.mtp.0.mlp.experts",torch.tensor([[1,2],[3,4]]))
            return torch.zeros(len(kw["input_ids"]),4)
        def compute_logits(self, hidden):
            logits=torch.zeros(2,64)
            logits[0,20+self.calls]=1
            logits[1,30+self.calls]=1
            return logits
    proposer.model=Model()
    set_inputs=method(path,"AscendSpecDecodeBaseProposer","set_inputs_first_pass",dict(torch=torch))
    n,indices,_,_=set_inputs(proposer,torch.tensor([12,22]),torch.tensor([13,23]),torch.tensor([2,1]),
        torch.zeros(2,4),torch.tensor([0,1]),NS(query_start_loc=torch.tensor([0,1,2])),None)
    merged=method(path,"AscendSpecDecodeBaseProposer","_run_merged_draft",dict(torch=torch,
        lmhead_tp_enable=lambda:False,get_ascend_config=lambda:NS(enable_reduce_sample=False),
        get_forward_context=lambda:NS(attn_metadata=None),_EXTRA_CTX=NS()))
    @adapter.trace_proposal
    def proposal(self,scheduler_output=None):
        return merged(self,n,2,indices,torch.tensor([2,1]),None,[],n)
    values=proposal(proposer,scheduler_output=object())
    assert values.tolist()==[list(range(21,21+k)),list(range(31,31+k))]
    assert proposer.model.calls==k
    if not summary_mode:
        recorded=events(session)
        assert len([e for e in recorded if e["type"]=="execution"])==k
        assert recorded[-1]["type"]=="proposal" and len(recorded[-1]["candidates"])==2*k
        assert [c["token_position"] for c in recorded[-1]["candidates"] if c["request_id"]=="A"]==list(range(4,4+k))
    # A later scheduler batch reverses request order and contains host placeholders.
    runner.input_batch.req_ids=["B","A"]
    b,a=values[1].tolist(),values[0].tolist()
    scheduler=NS(num_scheduled_tokens={"A":k+1,"B":k+1},total_num_scheduled_tokens=2*(k+1),
        scheduled_spec_decode_tokens={"A":[-1]*k,"B":[-1]*k})
    ids=torch.tensor([23,*b,13,*a])
    positions=torch.tensor([*range(2,3+k),*range(3,4+k)])
    with adapter.trace_target(runner,scheduler,ids,positions,NS(name="NONE")):
        capture("model.layers.0.mlp.experts",torch.tensor([[1,2]]).repeat(len(ids),1))
    accepted=k//2
    samples=[b[:accepted]+[61],a[:accepted]+[62]]
    samples=[r+[-1]*(k+1-len(r)) for r in samples]
    adapter.trace_sample(lambda *args:NS(sampled_token_ids=torch.tensor(samples)))(runner,None,
        NS(num_draft_tokens=[k,k],draft_token_ids=torch.tensor(b+a)))
    session.close()
    if summary_mode:
        record=json.loads((session.root/"load.jsonl").read_text().splitlines()[0])
        assert record["expert_choices"]==dict(target=4*(k+1),mtp=4*k,combined=8*k+4)
        assert record["partial"] and record["sampling"]=="continuous"
        return
    for dp in range(1,16):
        adapter.Session(runner,session.cfg,dict(dp=dp,tp=0,ep=dp)).close()
    report=script("scripts/validation/analyze_ascend_023_debug.py").analyze(session.root.parent.parent)
    assert report["expert_choices"]==dict(target=4*(k+1),mtp=4*k,combined=8*k+4)
    assert report["worker_coverage_complete"] and report["unresolved_candidate_links"]==0
    assert report["decisions"].get("accepted",0)==2*accepted
    assert report["decisions"]["rejected"]==2
    assert report["decisions"]["unverified"]==0


@pytest.mark.parametrize("summary_mode",[False,True])
def test_actual_acl_wrapper_capture_and_replay_cpu_simulation(setup, overlay, monkeypatch,summary_mode):
    runner, session = setup
    if summary_mode:
        from moe_trace.adapters.ascend_load_summary import SummarySession
        session=SummarySession(runner,dict(session.cfg,mode="summary",run_id="graph-summary",
            save_token_ids=False,save_expert_weights=False,interval_seconds=5,max_files=4),
            dict(dp=0,tp=0,ep=0),start_thread=False)
        monkeypatch.setattr(adapter,"_session",session)
        monkeypatch.setattr(adapter,"synchronize",lambda:pytest.fail("summary full-device barrier"))
    class Mode(Enum):
        NONE=0
        FULL=1
    context=NS(batch_descriptor=2,cudagraph_runtime_mode=Mode.FULL)
    sys.modules["vllm.forward_context"].get_forward_context=lambda:context
    offloader=ModuleType("vllm.model_executor.offloader.base")
    offloader.get_offloader=lambda:NS(sync_prev_onload=lambda:None,join_after_forward=lambda:None)
    monkeypatch.setitem(sys.modules,"vllm.model_executor.offloader.base",offloader)
    class Graph:
        def __init__(self):
            self.copies=[]
        def replay(self):
            for source,destination in self.copies:
                destination.copy_(source)
    class CaptureClones(TorchDispatchMode):
        def __init__(self,graph):
            self.graph=graph
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            if func == torch.ops.aten.clone.default:
                self.graph.copies.append((args[0],result))
            return result
    @contextmanager
    def graph_context(graph,pool=None):
        with CaptureClones(graph):
            yield
    monkeypatch.setattr(torch,"npu",NS(NPUGraph=Graph,graph=graph_context,
        current_stream=lambda:NS(synchronize=lambda:None)),raising=False)
    env=dict(torch=torch,CUDAGraphMode=Mode,get_forward_context=lambda:context,
        ACLGraphEntry=lambda **kw:NS(**kw,aclgraph=None),validate_cudagraph_capturing_enabled=lambda:None,
        ExitStack=ExitStack,weak_ref_workspaces=lambda p:None,weak_ref_tensors=lambda x:x,
        _graph_params=None,_draft_graph_params=None,_draft_graph_prefill_params=None,
        compilation_counter=NS(num_cudagraph_captured=0),logger=NS(info_once=lambda *a:None),
        _EXTRA_CTX=NS(is_draft_model=False))
    from moe_trace.adapters.ascend_graph_bridge import trace_acl_graph
    env["trace_acl_graph"]=trace_acl_graph
    call=method(overlay/"vllm_ascend/compilation/acl_graph.py","ACLGraphWrapper","__call__",env,decorators=True)
    route_ids=torch.tensor([[1,2],[3,4]])
    calls=[]
    def model():
        calls.append("capture")
        capture("model.layers.0.mlp.experts",route_ids)
        return torch.ones(2,4)
    wrapper=NS(runtime_mode=Mode.FULL,concrete_aclgraph_entries={},runnable=model,
        aclgraph_options=NS(debug_log_enable=False,gc_disable=False,weak_ref_output=False),
        graph_pool=None,is_debugging_mode=False,use_eagle=True,enable_enpu=False)
    arm_graph_capture(1,None if summary_mode else lambda:None)
    original=call(wrapper) # Real reference __call__ startup capture; graph runtime is simulated.
    assert not (session.root/"load.jsonl").read_text() if summary_mode else not events(session)
    route_ids[:]=torch.tensor([[5,6],[7,0]])
    if summary_mode:
        scheduler=NS(num_scheduled_tokens={"A":1,"B":1},total_num_scheduled_tokens=2,scheduled_spec_decode_tokens={})
        with adapter.trace_target(runner,scheduler,None,torch.tensor([2,1]),Mode.FULL):
            assert call(wrapper) is original
        session.flush()
        record=json.loads((session.root/"load.jsonl").read_text().splitlines()[0])
        counts=next(row for row in record["layers"] if row["kind"]=="target_decode")
        assert counts["expert_selection_counts"]==[1,0,0,0,0,1,1,1]
        assert calls==["capture"]
        return
    with execution_routes(1) as snapshots:
        assert call(wrapper) is original
    assert calls==["capture"]
    assert snapshots[0].expert_ids.tolist()==[[5,6],[7,0]]


def test_launcher_preserves_original_command_and_checks_overlay(setup,overlay,tmp_path):
    _,session=setup
    config=tmp_path/"launch.json"
    config.write_text(json.dumps(dict(session.cfg,run_id="launch-check")))
    launcher=Path(__file__).resolve().parents[1]/"scripts/deployment/ascend_023_debug.py"
    command=[sys.executable,str(launcher),"--overlay",str(overlay),"--config",str(config)]
    result=subprocess.run([*command,"--check"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)["launch_check"]=="passed"
    code="import json,sys,os;print(json.dumps({'args':sys.argv[1:],'config':os.environ['MOE_TRACE_ASCEND_023_CONFIG']}))"
    result=subprocess.run([*command,"--",sys.executable,"-c",code,"literal ; $value"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    output=json.loads(result.stdout.splitlines()[-1])
    assert output["args"]==["literal ; $value"] and output["config"]==str(config)
    with (overlay/"vllm_ascend/worker/worker.py").open("a") as stream:
        stream.write("\n# modified after staging\n")
    result=subprocess.run([*command,"--check"],capture_output=True,text=True)
    assert result.returncode==2 and "integrity mismatch" in result.stderr
