"""CPU execution of real patched MTP hook blocks; NPU kernels and models mocked."""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from moe_trace.adapters import ascend_v4_mtp as adapter
from moe_trace.distributed.decode import Worker
from moe_trace.distributed.storage import collect_run
from moe_trace.distributed.v4_mtp_cli import collect, analyze
from moe_trace.errors import ConfigError, IntegrityError
from test_v4_debug import script, method, runtime, layer

ROOT = Path(__file__).resolve().parents[1]


def setup(tmp_path, monkeypatch, privacy=False):
    runner = runtime()
    runner.vllm_config.speculative_config = NS(method="mtp", num_speculative_tokens=1,
                                               disable_padded_drafter_batch=False)
    target = [layer(0), layer(1)]
    draft = [layer(0)]
    draft[0].layer_name = "draft_model.mtp.0.mlp.experts"
    runner.model = NS(named_modules=lambda: [(str(i), l) for i,l in enumerate(target)])
    runner.drafter = NS(model=NS(named_modules=lambda: [("mtp", draft[0])]))
    cfg = dict(enabled=True, output_root=str(tmp_path), run_id="cpu", node_id="D0",
               save_token_ids=not privacy, save_expert_weights=not privacy)
    worker = Worker(adapter.PROFILE, "cpu", "D0", "dp-0-tp-0", 0, 0, 0, 0)
    session = adapter.Session(runner, cfg, worker)
    monkeypatch.setattr(adapter, "read_config", lambda: cfg)
    monkeypatch.setattr(adapter, "get_session", lambda *a: session)
    return runner, target, draft, session, cfg


def actual_block(path, name, namespace):
    nodes = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.With)
             and any(isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Name)
                     and i.context_expr.func.id == name for i in n.items)]
    assert len(nodes) == 1
    tree = ast.parse("def run():\n    pass\n")
    tree.body[0].body = [nodes[0]]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace["run"]


def source_scenario(tmp_path, monkeypatch, privacy=False, omit_result=False, fault=None):
    vllm, ascend = os.environ.get("MOE_V4_VLLM_SOURCE"), os.environ.get("MOE_V4_ASCEND_SOURCE")
    if not vllm or not ascend:
        pytest.skip("fixed V4 source repositories required")
    stage = script("scripts/tools/stage_ascend_v4_mtp.py")
    overlay = stage.stage(Path(vllm)/"vllm", Path(ascend)/"vllm_ascend", tmp_path/"overlays")
    launcher = script("scripts/deployment/ascend_v4_mtp_1p1d.py")
    launcher.verify_overlay(overlay)
    assert (overlay/"debug-hooks.patch").read_bytes() == (ROOT/"patches/ascend/deepseek-v4-mtp/0001-mtp-hooks.patch").read_bytes()
    proposer_tree=ast.parse((overlay/"vllm_ascend/spec_decode/llm_base_proposer.py").read_text())
    proposal_method=next(n for n in ast.walk(proposer_tree) if isinstance(n,ast.FunctionDef) and n.name=="_propose")
    assert any(isinstance(n,ast.Name) and n.id=="trace_proposal_call" for n in proposal_method.decorator_list)
    worker_tree=ast.parse((overlay/"vllm_ascend/worker/worker.py").read_text())
    hook_names=[a.name for n in ast.walk(worker_tree) if isinstance(n,ast.ImportFrom)
                and n.module=="moe_trace.adapters.ascend_v4_mtp" for a in n.names]
    assert set(hook_names)=={"initialize","close"}
    runner, target, draft, session, cfg = setup(tmp_path, monkeypatch, privacy)
    source = overlay/"vllm_ascend"
    prepare_fn = method(source/"ops/fused_moe/prepare_finalize.py", "PrepareAndFinalizeWithAll2All", "prepare",
                        dict(torch=torch, QuantType=NS(NONE=None), MoEPrepareOutput=lambda **kw:NS(**kw)))
    prepare = type("PrepareAndFinalizeWithAll2All", (), {"prepare":prepare_fn})()
    prepare.tp_size, prepare.tp_rank, prepare.lora_context = 1, 0, None
    context = NS(dp_metadata=None, all_moe_layers=[])
    consumed = []
    def kernel(*, fused_experts_input, quant_method):
        consumed.append((fused_experts_input.layer.layer_name, fused_experts_input.topk_ids.clone()))
        return NS(routed_out=fused_experts_input.hidden_states)
    comm = NS(prepare_finalize=prepare, prepare=prepare.prepare, fused_experts=kernel,
              finalize=lambda **kw:kw["hidden_states"])
    extra = NS(moe_comm_method=comm, in_profile_run=False, moe_comm_type="alltoall", flash_comm_v1_enabled=False)
    namespace = dict(torch=torch, _EXTRA_CTX=extra, get_forward_context=lambda:context,
        get_ascend_config=lambda:NS(enable_force_eplb=False), get_moe_num_logical_experts=lambda l,n,**kw:n,
        get_moe_comm_method=lambda *a:comm, AllGatherCommImpl=type("AllGatherCommImpl", (), {}),
        build_fused_experts_input=lambda **kw:NS(**kw))
    select = method(source/"ops/fused_moe/routed_experts.py", "AscendRoutedExperts", "_select_experts", namespace)
    forward = method(source/"ops/fused_moe/routed_experts.py", "AscendRoutedExperts", "forward_impl", namespace)
    apply = method(source/"quantization/methods/w8a8/w8a8_dynamic.py", "AscendW8A8DynamicFusedMoEMethod", "apply", namespace)
    for l in target+draft:
        l.router._select_experts = lambda hidden_states, router_logits, input_ids: (
            torch.tensor([[0.75,0.25]]).repeat(len(hidden_states),1),
            torch.tensor([[0,2]]).repeat(len(hidden_states),1).to(torch.int32))
        l.log2phy = torch.tensor([1,3,0,2])
        l.n_shared_experts, l.global_redundant_expert_num, l.mix_placement = 0, 0, False
        l.enable_npugraph_ex_static_kernel, l._use_v2_model_runner = False, False
        l.dynamic_eplb, l.return_with_event, l.quant_type = False, False, "w8a8"
        l.ascend_expert_map, l.apply_router_weight_on_input = None, False
        scheme = NS(quant_type="w8a8", use_expert_weight_list=False)
        l.quant_method = NS(apply=lambda _s=scheme,**kw:apply(_s,**kw))
        l._select_experts = lambda _l=l,**kw:select(_l,**kw)
    def model(layers, ids):
        for l in layers:
            forward(l, hidden_states=torch.zeros(len(ids),2), router_logits=torch.ones(len(ids),4), input_ids=ids)
        return torch.zeros(len(ids),2)
    sequences = {"B":[20,21], "A":[10,11,12]}
    runner.requests = {r:NS(num_prompt_tokens=1 if r=="B" else 2,
                            get_token_id=lambda p,_r=r:sequences[_r][p]) for r in sequences}
    runner.discard_request_indices, runner.num_discarded_requests = NS(np=np.array([],dtype=np.int32)), 0
    def target_forward(reqs, starts, ids, positions, specs):
        runner.input_batch = NS(req_ids=reqs, num_computed_tokens_cpu=starts,
            sampling_metadata=None, update_async_output_token_ids=lambda:None)
        schedule = NS(num_scheduled_tokens={r:2 if specs else 1 for r in reversed(reqs)},
            scheduled_spec_decode_tokens=specs, total_num_scheduled_tokens=len(ids))
        runner._model_forward = lambda *a,**kw:model(target, ids)
        values = dict(self=runner, scheduler_output=schedule, input_ids=ids, positions=positions,
            cudagraph_mode=NS(name="NONE"), trace_target=adapter.trace_target,
            num_tokens_padded=len(ids), intermediate_tensors=None, inputs_embeds=None, model_kwargs={})
        actual_block(source/"worker/model_runner_v1.py", "trace_target", values)()
        return schedule
    def draft_forward(ids, positions, starts, selected, next_ids, outputs, schedule):
        proposer = NS(runner=runner, method="mtp", num_speculative_tokens=1, needs_extra_input_slots=False,
            use_cuda_graph=False, input_ids=torch.zeros(8,dtype=torch.long), hidden_states=torch.zeros(8,2),
            query_start_loc_group=[torch.tensor(starts)])
        position_buffer = torch.zeros(8,dtype=torch.long)
        proposer._set_positions = lambda n,p:position_buffer[:n].copy_(p)
        # Execute the actual MTP shift and accepted/rejected padding preparation.
        set_inputs = method(source/"spec_decode/llm_base_proposer.py", "AscendSpecDecodeBaseProposer",
            "set_inputs_first_pass", dict(torch=torch))
        n, indices, _, _ = set_inputs(proposer, ids, torch.tensor(next_ids), positions,
            torch.zeros(len(ids),2), torch.tensor(selected), NS(query_start_loc=torch.tensor(starts)), None)
        assert n == len(ids)
        padded = max(6,n)
        proposer.model = lambda **kw:model(draft, kw["input_ids"])
        kwargs = dict(input_ids=proposer.input_ids[:padded], positions=position_buffer[:padded])
        @adapter.trace_proposal_call
        def propose(p, **kw):
            values = dict(self=p, model_kwargs=kwargs, num_tokens=n, batch_size=2,
                          token_indices_to_sample=indices, trace_draft=adapter.trace_draft)
            actual_block(source/"spec_decode/llm_base_proposer.py", "trace_draft", values)()
            tree = ast.parse((source/"spec_decode/llm_base_proposer.py").read_text())
            returns = [node for node in ast.walk(tree) if isinstance(node,ast.Return)
                and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                and node.value.func.id=="record_proposals"]
            assert len(returns)==1
            wrapper=ast.parse("def returned():\n    pass\n")
            wrapper.body[0].body=[returns[0]]
            ns=dict(self=p, draft_token_ids=torch.tensor(outputs), record_proposals=adapter.record_proposals,
                    token_indices_to_sample=indices)
            exec(compile(ast.fix_missing_locations(wrapper),"actual-MTP-return","exec"),ns)
            return ns["returned"]()
        return propose(proposer, scheduler_output=schedule, num_speculative_tokens=1)
    first = target_forward(["B","A"],[1,2],torch.tensor([21,12]),torch.tensor([1,2]),{})
    sequences["B"].append(22)
    sequences["A"].append(13)
    draft_forward(torch.tensor([21,12]),torch.tensor([1,2]),[0,1,2],[0,1],[22,13],[23,14],first)
    second = target_forward(["A","B"],[3,2],torch.tensor([13,14,22,23]),torch.tensor([3,4,2,3]),
                            {"B":[24 if fault=="candidate_token" else 23],"A":[14]})
    metadata = NS(num_draft_tokens=[1,1],draft_token_ids=torch.tensor([14,23]))
    if fault=="sampler_token":
        metadata.draft_token_ids=torch.tensor([23,14])
    sampled = NS(sampled_token_ids=torch.tensor([[15,-1],[23,24]]))
    if fault=="sampler_prefix":
        sampled.sampled_token_ids[1,0]=25
    runner._get_spec_decode_draft_probs=lambda m:None
    runner.rejection_sampler=lambda *a:sampled
    sample = method(source/"worker/model_runner_v1.py", "NPUModelRunner", "_sample",
                    dict(lmhead_tp_enable=lambda:False))
    if not omit_result:
        assert sample(runner,torch.zeros(4,32),metadata) is sampled
    sequences["A"].append(15)
    sequences["B"].extend([23,24])
    draft_forward(torch.tensor([13,14,22,23]),torch.tensor([3,4,2,3]),[0,2,4],[2,3] if fault=="sample_index" else [0,3],
                  [15,24],[14,23],second)  # Same candidate IDs, different round/position.
    session.close()
    for d in range(1,16):
        w=Worker(adapter.PROFILE,"cpu","D0",f"dp-{d}-tp-0",d,0,d,d)
        adapter.Session(runner,cfg,w).close()
    run=tmp_path/"cpu"
    (run/"deployment.json").write_text(json.dumps(dict(profile=adapter.PROFILE,run_id="cpu",node_id="D0",
        expected_workers=[f"dp-{d}-tp-0" for d in range(16)])))
    assert all(torch.equal(ids, torch.tensor([[1,0]]).repeat(len(ids),1)) for _,ids in consumed)
    return run, overlay


def test_actual_patched_mtp_source_cpu(tmp_path,monkeypatch):
    run,_=source_scenario(tmp_path,monkeypatch)
    joined,candidates=collect(run)
    totals,counts,_,_=analyze(joined)
    assert len(joined)==17 and totals=={"target":24,"mtp":10,"combined":34}
    assert sum(counts.values())==34
    assert sorted(c["state"] for c in candidates)==["accepted","rejected","unverified","unverified"]
    rejected=[r for r,a in joined if a is False]
    assert len(rejected)==2 and all(r.row.local_request=="A" and r.row.token_id==14 for r in rejected)
    assert {r.row.kind for r,_ in joined}=={"target_decode","target_verify","mtp_draft"}
    draft_rows={(r.row.execution,r.row.input_row,r.row.local_request,r.row.position,r.row.token_id)
                for r,_ in joined if r.model_scope=="mtp"}
    assert draft_rows=={("execution-2",0,"B",1,22),("execution-2",1,"A",2,13),
        ("execution-4",0,"A",3,15),("execution-4",2,"B",2,23),("execution-4",3,"B",3,24)}


def test_mtp_privacy_includes_seed_and_candidate_ledgers(tmp_path,monkeypatch):
    run,_=source_scenario(tmp_path,monkeypatch,privacy=True)
    joined,candidates=collect(run)
    assert all(r.row.token_id is None and r.weights is None for r,_ in joined)
    assert all(c["token_id"] is None for c in candidates)
    assert analyze(joined)[0]["combined"]==34


def test_missing_acceptance_result_is_not_assumed_rejected(tmp_path,monkeypatch):
    with pytest.raises(IntegrityError,match="missing verification result"):
        source_scenario(tmp_path,monkeypatch,omit_result=True)


def test_mtp_runtime_profile_guard():
    runner=runtime()
    runner.vllm_config.speculative_config=NS(method="mtp",num_speculative_tokens=1)
    adapter.check_runtime(runner,NS(name="NONE"))
    for method_name,k in [("dspark",1),("mtp",2),("eagle",1)]:
        runner.vllm_config.speculative_config=NS(method=method_name,num_speculative_tokens=k)
        with pytest.raises(ConfigError):
            adapter.check_runtime(runner,NS(name="NONE"))


def test_mtp_launcher_separate_profile():
    launcher=script("scripts/deployment/ascend_v4_mtp_1p1d.py")
    for role,dp,tp in [("P",4,4),("D",16,1)]:
        tasks=launcher.commands(role,"/model","192.0.2.1")
        assert len(tasks)==dp
        for t in tasks:
            argv=t["command"]
            assert argv[argv.index("--tensor-parallel-size")+1]==str(tp)
            spec=json.loads(argv[argv.index("--speculative-config")+1])
            assert spec==dict(method="mtp",num_speculative_tokens=1,enforce_eager=True)
            assert "--no-async-scheduling" in argv and "--enforce-eager" in argv


@pytest.mark.parametrize("fault",["candidate_token","sampler_token","sampler_prefix","sample_index"])
def test_mtp_cross_associations_rejected(tmp_path,monkeypatch,fault):
    with pytest.raises(IntegrityError):
        source_scenario(tmp_path,monkeypatch,fault=fault)
    status=tmp_path/"cpu/dp-0-tp-0/trace/worker-00000/status.json"
    assert json.loads(status.read_text())["state"]=="failed"


@pytest.mark.parametrize("fault",["hash","wrong_request","wrong_position","missing_worker"])
def test_mtp_offline_integrity(tmp_path,monkeypatch,fault):
    run,_=source_scenario(tmp_path,monkeypatch)
    if fault=="missing_worker":
        (run/"dp-15-tp-0/candidates-index.json").unlink()
        with pytest.raises(OSError):
            collect(run)
        return
    path=run/"dp-0-tp-0/candidates.jsonl"
    records=[json.loads(line) for line in path.read_text().splitlines()]
    if fault=="wrong_request":
        records[0]["request"]="A"
    elif fault=="wrong_position":
        records[0]["position"]+=1
    else:
        records[0]["token_id"]=999
    data=("\n".join(json.dumps(v) for v in records)+"\n").encode()
    path.write_bytes(data)
    if fault!="hash":
        (path.parent/"candidates-index.json").write_text(json.dumps({"records":len(records),
            "sha256":hashlib.sha256(data).hexdigest()}))
    with pytest.raises(IntegrityError):
        collect(run)


def test_mtp_export_counts_keep_rejected_and_zero_experts(tmp_path,monkeypatch):
    import csv
    run,_=source_scenario(tmp_path,monkeypatch)
    result=subprocess.run([sys.executable,"-m","moe_trace.distributed.v4_mtp_cli",str(run),
        "--output",str(tmp_path/"details.json"),"--counts",str(tmp_path/"counts.csv"),
        "--by-execution",str(tmp_path/"executions.csv")],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    output=json.loads((tmp_path/"details.json").read_text())
    assert output["expert_choices"]==dict(target=24,mtp=10,combined=34)
    rows=list(csv.DictReader((tmp_path/"counts.csv").open()))
    assert sum(int(r["choices"]) for r in rows)==34
    assert sum(int(r["choices"]) for r in rows if r["candidate_decision"]=="rejected")==4
    assert any(r["expert_id"]=="1" and r["choices"]=="0" for r in rows)
    assert len(output["routes"])==17


def test_mtp_disabled_and_dummy_proposals_do_nothing(monkeypatch):
    monkeypatch.delenv("MOE_TRACE_V4_MTP_CONFIG",raising=False)
    called=[]
    @adapter.trace_proposal_call
    def original(proposer,**kwargs):
        called.append("original")
        return "output"
    assert original(None,scheduler_output=object(),num_speculative_tokens=1)=="output"
    with adapter.trace_target(None,None,None,None,None):
        adapter.capture_routes(None,None,None,None,None,None,None)
    with adapter.trace_draft(None,None,None,None,None):
        adapter.capture_consumer(None,None)
    adapter.record_verification(None,None,None)
    assert called==["original"]


def test_missing_proposal_hook_fails_worker(tmp_path,monkeypatch):
    runner,_,_,current,_=setup(tmp_path,monkeypatch)
    @adapter.trace_proposal_call
    def original(proposer,**kwargs):
        return torch.tensor([[1]])
    with pytest.raises(IntegrityError,match="no captured producer"):
        original(NS(runner=runner),scheduler_output=object(),num_speculative_tokens=1)
    assert current.writer.failed


def test_mixed_zero_and_one_candidate_sampler_batch(tmp_path,monkeypatch):
    runner,_,_,current,_=setup(tmp_path,monkeypatch)
    runner.input_batch=NS(req_ids=["A","B"])
    current.verifying={"B":{"token_id":23,"candidate_round":"draft-2"}}
    adapter.record_verification(runner,NS(num_draft_tokens=[0,1],draft_token_ids=torch.tensor([23])),
        NS(sampled_token_ids=torch.tensor([[15,-1],[23,24]])))
    assert not current.verifying and not current.writer.failed
    with pytest.raises(IntegrityError,match="candidates differ"):
        adapter.record_verification(runner,NS(num_draft_tokens=[0,1],draft_token_ids=torch.tensor([23])),
            NS(sampled_token_ids=torch.tensor([[15,-1],[23,24]])))
    assert current.writer.failed
