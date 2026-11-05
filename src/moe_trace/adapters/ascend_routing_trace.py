"""Lossless ordered token/layer routes; bounded CPU batches, no count merging."""
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import threading
import time

from .ascend_load_summary import SummarySession
from .ascend_request_flow import placements_from_runner
from ..io import atomic_json

PROFILE="deepseek-v4-023post1-routing-trace"


class RoutingTraceSession(SummarySession):
    mode="routing_trace"
    profile=PROFILE
    file_stem="routes"

    def __init__(self,runner,cfg,ranks,*,placements=None,start_thread=True,clock=time.monotonic):
        self.runner,self.cfg,self.ranks,self.clock=runner,cfg,ranks,clock
        self.placements=placements if placements is not None else placements_from_runner(runner,ranks["ep"],16)
        self.placement_sha256=hashlib.sha256(json.dumps(self.placements,sort_keys=True).encode()).hexdigest()
        self.root=Path(cfg["output_root"])/cfg["run_id"]/cfg["node_id"]/f"source-dp-{ranks['dp']:03d}"
        self.root.mkdir(parents=True,exist_ok=False)
        self.identity=dict(node_id=cfg["node_id"],hostname=socket.gethostname(),dp_rank=ranks["dp"],
            tp_rank=ranks["tp"],ep_rank=ranks["ep"],device_index=ranks.get("device"),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            identity_status="resolved" if ranks.get("device") is not None else "rank_only")
        atomic_json(self.root/"worker.json",self.identity)
        atomic_json(self.root/"placements.json",self.placements)
        self.lock=threading.RLock()
        self.stop=threading.Event()
        self.pending_bytes=0
        self.buffer=[]
        self.sequence=self.executions=self.scheduler_step=0
        self.window=self.rotations=self.bytes=0
        self.file_ranges=[[]]
        self.skipped_idle_windows=0
        self.error=None
        self.closed=False
        self.thread=None
        self.current_target=None
        self.last_forward=None
        self.pending={}
        self.started=self.clock()
        self.wall_started=time.time_ns()
        self.stream=(self.root/"routes.jsonl").open("xb")
        self.logger=logging.getLogger("vllm.moe_trace.routing")
        self.status("recording",force=True)
        if start_thread:
            self.start_timer()
        if ranks["dp"]==0:
            self.logger.info("moe-trace routing_trace enabled; source folders under %s; interval=%ss",
                             self.root.parent,cfg["interval_seconds"])

    def status(self,state,force=False):
        if state=="recording" and not force:
            return
        with self.lock:
            target=self.current_target
            atomic_json(self.root/"status.json",dict(profile=PROFILE,mode=self.mode,state=state,
                run_id=self.cfg["run_id"],node_id=self.cfg["node_id"],ranks=self.ranks,
                source=self.identity,windows_written=self.window,rotations=self.rotations,
                full_history_retained=self.rotations<self.cfg["max_files"],
                first_retained_window=min((r[0] for r in self.file_ranges if r),default=None),
                last_retained_window=max((r[-1] for r in self.file_ranges if r),default=None),
                observed_target_forwards=self.scheduler_step,executions=self.executions,
                buffered_bytes=self.pending_bytes,skipped_idle_windows=self.skipped_idle_windows,
                pending_verification=bool(target and any(target["drafts"].values())),
                error=str(self.error) if self.error else None,npu_verified=False,
                schedule_semantics="source_engine_observed_target_forward_sequence"))

    def next_execution(self):
        self.check()
        self.sequence+=1
        self.executions+=1
        return f"dp-{self.ranks['dp']}:execution-{self.sequence}"

    def emit(self,event):
        with self.lock:
            self.check()
            kind=event["type"]
            if kind=="execution":
                if not event["routes"]:
                    return
                for row in event["routes"]:
                    placement=self.placements.get(row["layer_name"])
                    if placement is None or placement["scope"]!=event["scope"] or placement["num_experts"]!=row["num_experts"]:
                        raise ValueError("trace layer differs from verified expert placement")
            elif kind=="sampler":
                if not event["decisions"]:
                    return
            elif kind=="proposal":
                if not event["candidates"]:
                    return
            else:
                raise ValueError("unknown routing trace event")
            data=json.dumps(event,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
            if len(data)>self.cfg["max_buffer_bytes"]:
                raise ValueError("one execution/event exceeds max_buffer_bytes; increase the buffer")
            if self.pending_bytes+len(data)>self.cfg["max_buffer_bytes"]:
                self.flush(partial=True,reason="buffer_limit")
            self.buffer.append(data)
            self.pending_bytes+=len(data)

    def _device_table(self,needed):
        found={}
        for path in self.root.parent.glob("source-dp-*/worker.json"):
            identity=json.loads(path.read_text())
            rank=identity["ep_rank"]
            if rank in found:
                raise ValueError("duplicate trace EP endpoint")
            found[rank]=identity
        return {str(rank):found.get(rank,dict(ep_rank=rank,identity_status="unresolved")) for rank in sorted(needed)}

    def _execution(self,event,needed):
        requests={}
        scope=event["scope"]
        for route in event["routes"]:
            req=route["request_id"]
            tokens=requests.setdefault(req,{})
            index=route["input_row"]
            metadata={key:value for key,value in route.items() if key not in {
                "request_id","layer_id","layer_name","num_experts","expert_ids","expert_weights"}}
            token=tokens.setdefault(index,dict(metadata,layers=[]))
            if {key:value for key,value in token.items() if key!="layers"}!=metadata:
                raise ValueError("token identity changed between layers")
            owners=self.placements[route["layer_name"]]["owners"]
            destinations=[owners[expert] for expert in route["expert_ids"]]
            needed.update(destinations)
            layer=dict(layer_id=route["layer_id"],layer_name=route["layer_name"],num_experts=route["num_experts"],
                       expert_ids=route["expert_ids"],destination_ep_ranks=destinations)
            if "expert_weights" in route:
                layer["expert_weights"]=route["expert_weights"]
            token["layers"].append(layer)
        return dict(execution_id=event["execution_id"],scope=scope,
            mtp_step=event.get("proposal_step") if scope=="mtp" else None,
            graph_mode=event["graph_mode"],host_time_ns=event["host_time_ns"],
            requests=[dict(request_id=req,tokens=[dict(tokens[i],layers=sorted(tokens[i]["layers"],key=lambda l:l["layer_id"]))
                                                for i in sorted(tokens)]) for req,tokens in requests.items()])

    def flush(self,*,partial=False,reason=None):
        with self.lock:
            self.check()
            now=self.clock()
            if not self.buffer:
                self.skipped_idle_windows+=1
                self.started,self.wall_started=now,time.time_ns()
                return
            rounds={}
            sampler=[]
            proposals=[]
            needed={self.ranks["ep"]}
            for data in self.buffer:
                event=json.loads(data)
                if event["type"]=="execution":
                    seq=event["scheduler_step"]
                    bucket=rounds.setdefault(seq,dict(schedule_seq=seq,executions=[]))
                    bucket["executions"].append(self._execution(event,needed))
                elif event["type"]=="sampler":
                    sampler.append(event)
                else:
                    proposals.append(event)
            record=dict(schema_version=3,type="routing_trace_window",window_id=self.window,
                run_id=self.cfg["run_id"],node_id=self.cfg["node_id"],source=self.identity,
                start_time_ns=self.wall_started,end_time_ns=time.time_ns(),duration_seconds=now-self.started,
                partial=partial,flush_reason=reason or ("shutdown" if partial else "interval"),
                placement_sha256=self.placement_sha256,device_table=self._device_table(needed),
                schedule_semantics="source_engine_observed_target_forward_sequence",
                rounds=list(rounds.values()),sampler_events=sampler,proposal_events=proposals)
            self._write(record)
            self.window+=1
            self.buffer.clear()
            self.pending_bytes=0
            self.started,self.wall_started=now,record["end_time_ns"]
            self.status("recording",force=True)

    def close(self):
        self.stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join()
        with self.lock:
            if self.closed:
                return
            try:
                self.check()
                self.flush(partial=True)
                target=self.current_target
                self.status("incomplete" if target and any(target["drafts"].values()) else "closed")
                if self.ranks["dp"]==0:
                    self.logger.info("moe-trace routing_trace stopped; output=%s",self.root.parent)
            except BaseException as exc:
                self.fail(exc)
                raise
            finally:
                self.stream.close()
                self.closed=True
