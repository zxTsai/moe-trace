"""Continuous device-side logical expert counts with bounded periodic output."""
from contextlib import contextmanager, nullcontext
import json
import logging
import os
from pathlib import Path
import re
import threading
import time

from ..io import atomic_json
from .ascend_graph_bridge import execution_routes

PROFILE = "deepseek-v4-023post1-load-summary"


class SummarySession:
    mode = "summary"
    profile = PROFILE
    capped = False

    def __init__(self, runner, cfg, ranks, *, start_thread=True, clock=time.monotonic):
        self.runner, self.cfg, self.ranks, self.clock = runner, cfg, ranks, clock
        self.root = Path(cfg["output_root"])/cfg["run_id"]/cfg["node_id"]/f"dp-{ranks['dp']}-tp-{ranks['tp']}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.counts, self.tokens = {}, {}
        self.top_k = {}
        self.executions = {"target": 0, "mtp": 0}
        self.total_executions = {"target": 0, "mtp": 0}
        self.event, self.device, self.error = None, None, None
        self.transfer_stream = None
        self.window, self.rotations, self.bytes = 0, 0, 0
        self.file_ranges = [[]]
        self.started = self.clock()
        self.wall_started = time.time_ns()
        self.closed = False
        self.stream = (self.root/"load.jsonl").open("xb")
        self.logger = logging.getLogger("vllm.moe_trace.load")
        self.thread = None
        self.status("recording")
        if start_thread:
            self.start_timer()

    def start_timer(self):
        self.thread = threading.Thread(target=self._loop, name="moe-trace-writer", daemon=True)
        self.thread.start()

    def status(self, state):
        atomic_json(self.root/"status.json", dict(profile=self.profile, mode=self.mode, state=state,
            run_id=self.cfg["run_id"], node_id=self.cfg["node_id"], ranks=self.ranks,
            windows_written=self.window, rotations=self.rotations, retained_files=min(self.rotations+1,self.cfg["max_files"]),
            retention="rolling_windows", full_history_retained=self.rotations < self.cfg["max_files"],
            first_retained_window=min((r[0] for r in self.file_ranges if r),default=None),
            last_retained_window=max((r[-1] for r in self.file_ranges if r),default=None),
            interval_seconds=self.cfg["interval_seconds"], total_executions=self.total_executions,
            error=str(self.error) if self.error else None, npu_verified=False))

    def check(self):
        if self.error is not None:
            raise RuntimeError("load summary failed") from self.error
        if self.closed:
            raise RuntimeError("load summary is closed")

    def fail(self, exc):
        with self.lock:
            self.error = exc
            self.stop.set()
            self.status("failed")

    def _loop(self):
        while not self.stop.wait(self.cfg["interval_seconds"]):
            try:
                self.flush()
            except BaseException as exc:
                self.fail(exc)
                self.logger.exception("moe-trace periodic writer failed")
                return

    def _wait(self):
        if self.device is not None and self.device.type != "cpu":
            import torch
            torch.npu.set_device(self.device)
            if self.event is not None:
                self.event.synchronize()

    def add(self, scope, snapshots, masks):
        """Enqueue counts on the calling stream; no .cpu/.item here."""
        import torch
        from vllm.forward_context import get_forward_context
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        expected = self.runner.vllm_config.model_config.hf_config.num_hidden_layers if scope == "target" else 1
        if len(snapshots) != expected or len({s.layer for s in snapshots}) != expected:
            raise ValueError(f"incomplete {scope} summary routing coverage")
        with self.lock:
            self.check()
            for snapshot in snapshots:
                ids, info = snapshot.expert_ids, snapshot.metadata
                if self.device is None:
                    self.device = ids.device
                    if self.device.type != "cpu":
                        self.transfer_stream = torch.npu.Stream(device=self.device)
                elif self.device != ids.device:
                    raise ValueError("summary tensors changed device")
                if self.device.type != "cpu" and self.event is not None:
                    torch.npu.current_stream().wait_event(self.event)
                if info is None:
                    raise ValueError("missing captured routing layout")
                indices = local_indices(info["prepare"], len(next(iter(masks.values()))), len(ids),
                    self.ranks["dp"], get_forward_context(), _EXTRA_CTX, ids.device)
                ids = ids.index_select(0, indices).long()
                layer = re.search(r"(?:^|\.)(?:layers|mtp)\.(\d+)(?:\.|$)", snapshot.layer)
                if layer is None:
                    raise ValueError("cannot identify summary layer")
                for kind, mask in masks.items():
                    excluded=mask[len(indices):].sum()
                    mask = mask[:len(indices)]
                    key = (scope, kind, snapshot.layer, int(layer[1]), info["num_experts"])
                    if key not in self.counts:
                        self.counts[key] = torch.zeros(info["num_experts"], dtype=torch.int64, device=ids.device)
                        self.tokens[key] = torch.zeros((), dtype=torch.int64, device=ids.device)
                        self.top_k[key] = ids.shape[1]
                    elif self.top_k[key] != ids.shape[1]:
                        raise ValueError("Top-K changed within a summary window")
                    # Invalid IDs remain observable: clamped scatter uses zero weight,
                    # and an extra device error counter is checked only at flush.
                    valid_ids = (ids >= 0) & (ids < info["num_experts"])
                    weights = mask[:,None].expand_as(ids).to(torch.int64)
                    error_key = ("invalid_ids", scope)
                    if error_key not in self.tokens:
                        self.tokens[error_key] = torch.zeros((), dtype=torch.int64, device=ids.device)
                    self.tokens[error_key].add_(((~valid_ids)*weights).sum()+excluded)
                    self.counts[key].scatter_add_(0, ids.clamp(0,info["num_experts"]-1).reshape(-1),
                                                 (weights*valid_ids).reshape(-1))
                    self.tokens[key].add_(mask.sum())
            if self.device is not None and self.device.type != "cpu":
                self.event = torch.npu.Event()
                self.event.record(torch.npu.current_stream())
            self.executions[scope] += 1
            self.total_executions[scope] += 1

    def _write(self, record):
        data = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",",":"))+"\n").encode()
        limit = self.cfg["max_file_bytes"]
        if len(data) > limit:
            raise ValueError("one load window exceeds max_file_bytes; increase the limit")
        if self.bytes + len(data) > limit:
            self.stream.close()
            root, keep = self.root, self.cfg["max_files"]
            stem=getattr(self,"file_stem","load")
            if keep > 1:
                (root/f"{stem}.{keep-1}.jsonl").unlink(missing_ok=True)
                for index in range(keep-2,0,-1):
                    source = root/f"{stem}.{index}.jsonl"
                    if source.exists():
                        source.replace(root/f"{stem}.{index+1}.jsonl")
                (root/f"{stem}.jsonl").replace(root/f"{stem}.1.jsonl")
            self.stream = (root/f"{stem}.jsonl").open("wb")
            self.bytes = 0
            self.rotations += 1
            self.file_ranges = ([[]]+self.file_ranges)[:self.cfg["max_files"]]
        self.stream.write(data)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.bytes += len(data)
        if not self.file_ranges[0]:
            self.file_ranges[0] = [record["window_id"],record["window_id"]]
        else:
            self.file_ranges[0][-1] = record["window_id"]

    def flush(self, *, partial=False):
        with self.lock:
            self.check()
            self._wait()
            now = self.clock()
            if now <= self.started and not any(self.executions.values()):
                return
            layers, totals = [], {"target":0,"mtp":0}
            keys=sorted(self.counts)
            errors=[v for key,v in self.tokens.items() if key[0]=="invalid_ids"]
            # One compact device-to-host transfer per active window.
            import torch
            packed=[self.counts[key] for key in keys]+[self.tokens[key].reshape(1) for key in keys]+[v.reshape(1) for v in errors]
            transfer=torch.npu.stream(self.transfer_stream) if self.transfer_stream is not None else nullcontext()
            with transfer:
                values=torch.cat(packed).detach().cpu().tolist() if packed else []
            count_size=sum(key[4] for key in keys)
            token_values=values[count_size:count_size+len(keys)]
            if any(values[count_size+len(keys):]):
                raise ValueError("invalid logical expert ID or row layout in summary window")
            offset=0
            for index,key in enumerate(keys):
                scope,kind,name,layer,experts = key
                vector=values[offset:offset+experts]
                offset+=experts
                n=token_values[index]
                layers.append(dict(scope=scope, kind=kind, layer_name=name, layer_id=layer,
                                   num_experts=experts, observed_input_tokens=n,
                                   top_k=self.top_k[key],
                                   expert_selection_counts=vector))
                totals[scope] += sum(vector)
            totals["combined"] = totals["target"]+totals["mtp"]
            record = dict(schema_version=1, type="load_summary", window_id=self.window,
                run_id=self.cfg["run_id"], node_id=self.cfg["node_id"], ranks=self.ranks,
                start_time_ns=self.wall_started, end_time_ns=time.time_ns(),
                duration_seconds=now-self.started, nominal_interval_seconds=self.cfg["interval_seconds"],
                partial=partial, sampling="continuous", executions=dict(self.executions),
                expert_choices=totals, layers=layers)
            record=self.format_record(record)
            self._write(record)
            self.window += 1
            # Drop completed window tensors; keep no historical device storage.
            self.counts.clear()
            self.tokens.clear()
            self.top_k.clear()
            self.event = None
            self.executions = {"target":0,"mtp":0}
            self.started, self.wall_started = now, record["end_time_ns"]
            self.after_flush()
            self.status("recording")
            self.logger.info("moe-trace dp=%s window=%s target=%s mtp=%s choices=%s partial=%s",
                self.ranks["dp"],record["window_id"],record["executions"]["target"],
                record["executions"]["mtp"],totals["combined"],partial)

    def format_record(self,record):
        return record

    def after_flush(self):
        pass

    def close(self):
        self.stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join()
        with self.lock:
            if self.closed:
                return
            try:
                if self.error is None:
                    self.flush(partial=True)
                    self.status("closed")
                else:
                    raise RuntimeError("load summary failed") from self.error
            except BaseException as exc:
                self.fail(exc)
                raise
            finally:
                self.stream.close()
                self.closed = True


def local_indices(kind, size, output_size, rank, context, extra, device):
    """Map only this DP worker's rows, using explicit prepare path metadata."""
    import torch
    offset = 0
    if kind == "PrepareAndFinalizeWithAll2All":
        expected = max(size,1)
    elif kind == "PrepareAndFinalizeWithMC2":
        expected = extra.padded_num_tokens
        if expected < size:
            raise ValueError("MC2 padding excludes summary input rows")
    elif kind == "PrepareAndFinalizeWithAllGather":
        sizes = context.dp_metadata.num_tokens_across_dp_cpu.tolist()
        if rank >= len(sizes) or not 0 <= sizes[rank] <= size:
            raise ValueError("DP metadata differs from summary local input")
        expected = max(sizes)*len(sizes)
        offset = rank*max(sizes)
        size = sizes[rank]
    else:
        raise ValueError(f"unsupported summary prepare path: {kind}")
    if output_size != expected:
        raise ValueError("summary selector rows differ from declared prepare layout")
    return torch.arange(size,device=device,dtype=torch.long)+offset


@contextmanager
def capture_forward(session, scope, masks):
    session.check()
    try:
        with execution_routes(session.runner.vllm_config.model_config.hf_config.num_hidden_layers+8) as snapshots:
            yield
        session.add(scope,snapshots,masks)
    except BaseException as exc:
        session.fail(exc)
        raise


def target_masks(runner,scheduler,positions,per_request=False):
    import torch
    reqs = list(runner.input_batch.req_ids)
    counts = dict(scheduler.num_scheduled_tokens)
    if (len(set(reqs))!=len(reqs) or set(reqs) != set(counts)
            or set(scheduler.scheduled_spec_decode_tokens)-set(reqs)
            or sum(counts.values()) != scheduler.total_num_scheduled_tokens):
        raise ValueError("summary target batch differs from scheduler")
    prompts, verify = [], []
    for req in reqs:
        n = counts[req]
        if type(n) is not int or n < 1 or len(scheduler.scheduled_spec_decode_tokens.get(req,[]))>=n:
            raise ValueError("invalid summary request count")
        prompts.extend([runner.requests[req].num_prompt_tokens]*n)
        verify.extend([bool(scheduler.scheduled_spec_decode_tokens.get(req))]*n)
    size = len(positions)
    if positions.ndim != 1 or len(prompts) > size:
        raise ValueError("invalid summary target positions")
    pad = size-len(prompts)
    threshold = torch.tensor(prompts+[0]*pad,device=positions.device)
    verification = torch.tensor(verify+[False]*pad,device=positions.device,dtype=torch.bool)
    valid = (torch.arange(size,device=positions.device)<len(prompts)) & (positions>=threshold)
    if per_request:
        result={}
        rows=torch.arange(size,device=positions.device)
        offset=0
        for req in reqs:
            kind="target_verify" if scheduler.scheduled_spec_decode_tokens.get(req) else "target_decode"
            result[(req,kind)]=valid & (rows>=offset)&(rows<offset+counts[req])
            offset+=counts[req]
        return result or {("","target_decode"):valid}
    masks={}
    if not verify or not all(verify):
        masks["target_decode"]=valid & ~verification
    if any(verify):
        masks["target_verify"]=valid & verification
    return masks


def mtp_masks(proposer,model_kwargs,num_tokens,batch_size,sample_indices,step,per_request=False):
    import torch
    runner = proposer.runner
    reqs = list(runner.input_batch.req_ids)
    positions = model_kwargs["positions"]
    if positions.ndim != 1 or batch_size < len(reqs) or num_tokens > len(positions):
        raise ValueError("invalid MTP summary layout")
    rows = torch.arange(len(positions),device=positions.device)
    mask = torch.zeros_like(positions,dtype=torch.bool)
    starts = proposer.query_start_loc_group[0] if step == 0 else None
    discarded = set(runner.discard_request_indices.np[:runner.num_discarded_requests].tolist())
    individual={}
    for index,req in enumerate(reqs):
        if index in discarded:
            continue
        start = starts[index] if step == 0 else index
        end = starts[index+1] if step == 0 else index+1
        last = sample_indices[index] if step == 0 else index
        current=(rows>=start)&(rows<end)&(rows<=last)&(rows<num_tokens)&(positions+1>=runner.requests[req].num_prompt_tokens)
        mask |= current
        if per_request:
            individual[(req,"mtp_draft")]=current
    if per_request:
        return individual or {("","mtp_draft"):mask}
    return {"mtp_draft":mask}
