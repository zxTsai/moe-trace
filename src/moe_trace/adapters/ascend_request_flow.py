"""Per-request DP source -> actual expert EP owner, emitted in timed windows."""
import hashlib
import json
import os
import socket

from .ascend_load_summary import SummarySession
from ..io import atomic_json

PROFILE="deepseek-v4-023post1-request-flow"


def owners_from_maps(maps,num_experts):
    if not maps or any(len(row)!=num_experts for row in maps):
        raise ValueError("invalid global expert map dimensions")
    owners=[]
    for expert in range(num_experts):
        ranks=[rank for rank,row in enumerate(maps) if row[expert]>=0]
        if len(ranks)!=1:
            raise ValueError("request flow requires exactly one owner per logical expert")
        owners.append(ranks[0])
    return owners


def placements_from_runner(runner,ep_rank,ep_size,determine=None):
    """Use the same provider as Ascend and verify its actual local assignment."""
    if determine is None:
        from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map
        determine=determine_expert_map
    result={}
    for scope,model in (("target",runner.model),("mtp",runner.drafter.model)):
        discovered=0
        for _,module in model.named_modules():
            name=getattr(module,"layer_name",None)
            local=getattr(module,"_expert_map",None)
            if not name or local is None or not hasattr(module,"moe_config"):
                continue
            if name in result:
                raise ValueError("duplicate expert placement layer")
            if getattr(module,"dynamic_eplb",False) or getattr(module,"global_redundant_expert_num",0):
                raise ValueError("replicated/dynamic expert placement unsupported")
            local=local.detach().cpu().tolist()
            count=len(local)
            global_map=getattr(module,"global_expert_map",None)
            if global_map is None:
                maps=[determine(ep_size,rank,count)[1].detach().cpu().tolist() for rank in range(ep_size)]
            else:
                maps=global_map.detach().cpu().tolist()
            if len(maps)!=ep_size or maps[ep_rank]!=local:
                raise ValueError("runtime local expert map differs from placement provider")
            owners=owners_from_maps(maps,count)
            physical=getattr(module,"log2phy",None)
            if physical is not None:
                slots=physical.detach().cpu().tolist()
                local_count=count//ep_size
                if local_count<1 or len(slots)!=count or any(
                    slots[e]//local_count!=owners[e] or slots[e]%local_count!=maps[owners[e]][e]
                    for e in range(count)):
                    raise ValueError("log2phy differs from unique runtime ownership")
            result[name]=dict(scope=scope,owners=owners,num_experts=count)
            discovered+=1
        expected=runner.vllm_config.model_config.hf_config.num_hidden_layers if scope=="target" else 1
        if discovered!=expected:
            raise ValueError(f"incomplete {scope} expert placement discovery")
    return result


class RequestFlowSession(SummarySession):
    mode="request_flow"
    profile=PROFILE

    def __init__(self,runner,cfg,ranks,*,placements=None,**kwargs):
        self.placements=placements if placements is not None else placements_from_runner(runner,ranks["ep"],16)
        self.window_requests=set()
        self.flush_reason=None
        start_thread=kwargs.pop("start_thread",True)
        super().__init__(runner,cfg,ranks,start_thread=False,**kwargs)
        self.identity=dict(node_id=cfg["node_id"],hostname=socket.gethostname(),
            dp_rank=ranks["dp"],tp_rank=ranks["tp"],ep_rank=ranks["ep"],
            device_index=ranks.get("device"),global_rank=ranks.get("global_rank"),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            identity_status="resolved" if ranks.get("device") is not None else "rank_only")
        atomic_json(self.root/"worker.json",self.identity)
        atomic_json(self.root/"placements.json",self.placements)
        self.placement_sha256=hashlib.sha256(json.dumps(self.placements,sort_keys=True).encode()).hexdigest()
        if start_thread:
            self.start_timer()

    def add(self,scope,snapshots,masks):
        with self.lock:
            self.check()
            requests={req for req,kind in masks if req}
            limit=self.cfg["max_requests_per_window"]
            if len(requests)>limit:
                raise ValueError("one batch exceeds max_requests_per_window")
            if len(self.window_requests|requests)>limit:
                self.flush_reason="request_limit"
                self.flush(partial=True)
            self.window_requests.update(requests)
            for snapshot in snapshots:
                placement=self.placements.get(snapshot.layer)
                if placement is None or placement["scope"]!=scope or placement["num_experts"]!=snapshot.metadata["num_experts"]:
                    raise ValueError("route layer differs from verified placement")
            super().add(scope,snapshots,masks)

    def endpoints(self):
        endpoints={}
        for path in self.root.parent.glob("dp-*-tp-*/worker.json"):
            identity=json.loads(path.read_text())
            rank=identity["ep_rank"]
            if rank in endpoints:
                raise ValueError("duplicate EP endpoint identity")
            endpoints[rank]=identity
        return endpoints

    def format_record(self,record):
        endpoints=self.endpoints()
        requests={}
        for row in record.pop("layers"):
            req,kind=row["kind"]
            if not row["observed_input_tokens"]:
                continue
            owner=self.placements[row["layer_name"]]["owners"]
            destinations={}
            for expert,count in enumerate(row.pop("expert_selection_counts")):
                if count:
                    destination=owner[expert]
                    flow=destinations.setdefault(destination,dict(
                        destination=endpoints.get(destination,dict(ep_rank=destination,identity_status="unresolved")),
                        expert_selection_count=0,expert_counts=[]))
                    flow["expert_selection_count"]+=count
                    flow["expert_counts"].append(dict(expert_id=expert,count=count))
            item=requests.setdefault(req,dict(request_id=req,source=self.identity,layers=[]))
            item["layers"].append(dict(row,kind=kind,destinations=[destinations[r] for r in sorted(destinations)]))
        record.update(schema_version=2,type="request_flow_summary",placement_sha256=self.placement_sha256,
            flush_reason=self.flush_reason or ("shutdown" if record["partial"] else "interval"),
            requests=[requests[r] for r in sorted(requests)])
        return record

    def after_flush(self):
        self.window_requests.clear()
        self.flush_reason=None
