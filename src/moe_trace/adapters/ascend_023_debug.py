"""Opt-in 0.23.post1 A3 async/graph debug recorder.

Device-wide barriers are intentional: this is a bounded debugging instrument,
not an asynchronous performance profiler. No NPU import is made until enabled.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import atexit
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

from .ascend_graph_bridge import arm_graph_capture, execution_routes, record_graph_routes, route_recording_active
from ..io import atomic_json

PROFILE = "deepseek-v4-023post1-async-graph-debug"
ENV = "MOE_TRACE_ASCEND_023_CONFIG"
_session = None
_proposal = ContextVar("ascend023_proposal", default=None)


def config():
    name = os.environ.get(ENV)
    if not name:
        return None
    value = json.loads(Path(name).read_text())
    required = {"output_root", "run_id", "node_id", "save_token_ids", "save_expert_weights",
                "max_executions", "max_file_bytes"}
    if set(value) != required:
        raise ValueError("unexpected Ascend 0.23 debug config keys")
    for key in ("run_id", "node_id"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value[key]):
            raise ValueError(f"unsafe {key}")
    if not isinstance(value["output_root"], str) or not Path(value["output_root"]).is_absolute():
        raise ValueError("output_root must be absolute")
    for key in ("save_token_ids", "save_expert_weights"):
        if type(value[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("max_executions", "max_file_bytes"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be positive")
    if any(os.environ.get(key) for key in ("MOE_TRACE_CONFIG", "MOE_TRACE_V4_MTP_CONFIG",
               "MOE_TRACE_V4_DEBUG_CONFIG", "MOE_TRACE_QWEN36_CONFIG", "MOE_TRACE_V41_DEBUG_CONFIG")):
        raise ValueError("cannot combine capture adapters")
    return value


def synchronize():
    import torch
    torch.npu.synchronize()


def host(tensor):
    return tensor.detach().cpu().tolist()


def check_runtime(runner):
    cfg = runner.vllm_config
    parallel = cfg.parallel_config
    for name, expected in {"data_parallel_size": 16, "tensor_parallel_size": 1,
                           "pipeline_parallel_size": 1, "prefill_context_parallel_size": 1,
                           "decode_context_parallel_size": 1}.items():
        if getattr(parallel, name, 1) != expected:
            raise ValueError(f"debug recorder requires {name}={expected}")
    if not parallel.enable_expert_parallel or getattr(parallel, "use_ubatching", False):
        raise ValueError("requires EP without microbatching")
    model = cfg.model_config
    if model.hf_config.architectures != ["DeepseekV4ForCausalLM"] or model.quantization != "ascend":
        raise ValueError("requires Ascend-quantized DeepseekV4ForCausalLM")
    if getattr(model, "enable_return_routed_experts", False) or getattr(cfg, "lora_config", None):
        raise ValueError("disable built-in routing capture and LoRA")
    if getattr(cfg, "use_v2_model_runner", False) or cfg.cache_config.enable_prefix_caching:
        raise ValueError("requires V1 without prefix caching")
    spec = cfg.speculative_config
    if (spec is None or spec.method != "mtp" or type(spec.num_speculative_tokens) is not int
            or spec.num_speculative_tokens < 1 or not spec.enforce_eager
            or getattr(spec, "disable_padded_drafter_batch", False)):
        raise ValueError("requires positive K padded eager MTP")
    kv = cfg.kv_transfer_config
    if kv is None or kv.kv_connector != "MooncakeHybridConnector" or kv.kv_role != "kv_consumer":
        raise ValueError("install this recorder on the MooncakeHybridConnector D consumer only")
    ascend = runner.ascend_config
    for name in ("enable_force_eplb", "enable_flashcomm1", "enable_sp_by_pass", "enable_dsa_cp",
                 "enable_shared_expert_dp", "multistream_overlap_gate", "mix_placement"):
        if getattr(ascend, name, False):
            raise ValueError(f"unsupported debug layout: {name}")
    if getattr(runner, "dynamic_eplb", False) or getattr(runner, "enable_enpu", False):
        raise ValueError("dynamic EPLB and ENPU unsupported")
    if getattr(parallel, "enable_eplb", False) or getattr(parallel, "use_sequence_parallel_moe", False):
        raise ValueError("EPLB and sequence-parallel MoE unsupported")
    if getattr(model.hf_config, "num_nextn_predict_layers", 1) != 1:
        raise ValueError("debug profile requires one physical MTP layer")
    compilation = ascend.ascend_compilation_config
    if getattr(compilation, "enable_static_kernel", False):
        raise ValueError("static kernels bypass the supported hook")
    if os.environ.get("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0") != "0":
        raise ValueError("FlashComm1 unsupported")
    mode = cfg.compilation_config.cudagraph_mode
    if getattr(mode, "name", str(mode)) not in {"NONE", "FULL", "FULL_DECODE_ONLY"}:
        raise ValueError("only NONE/FULL/FULL_DECODE_ONLY graph modes are supported for debugging")


class Session:
    def __init__(self, runner, cfg, ranks):
        self.runner, self.cfg, self.ranks = runner, cfg, ranks
        self.sequence = self.executions = self.bytes = 0
        self.capped = self.failed = self.closed = False
        self.current_target = None
        self.scheduler_step = 0
        self.pending = {}
        self.digest = hashlib.sha256()
        self.root = Path(cfg["output_root"]) / cfg["run_id"] / cfg["node_id"] / f"dp-{ranks['dp']}-tp-{ranks['tp']}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.stream = (self.root / "events.jsonl").open("xb")
        self.status("recording")
        self.last_forward = None

    def status(self, state, error=None):
        atomic_json(self.root / "status.json", {
            "profile": PROFILE, "state": state, "run_id": self.cfg["run_id"],
            "node_id": self.cfg["node_id"], "ranks": self.ranks,
            "executions": self.executions, "bytes": self.bytes,
            "sha256": self.digest.hexdigest(), "npu_verified": False,
            "validation_level": "debug_unverified_a3", "error": error,
            "scope": "logical_routed_experts_only", "performance_representative": False,
            "hostname": socket.gethostname(), "pid": os.getpid(), "python": sys.executable,
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "pending_verification": bool(self.current_target and any(self.current_target["drafts"].values())),
        })

    def emit(self, event):
        if self.failed or self.closed:
            raise RuntimeError("recorder is not writable")
        if self.capped:
            return
        data = (json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if self.bytes + len(data) > self.cfg["max_file_bytes"]:
            self.capped = True
            self.status("limit_reached")
            return
        self.stream.write(data)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.digest.update(data)
        self.bytes += len(data)
        self.status("recording")

    def fail(self, exc):
        self.failed = True
        self.status("failed", f"{type(exc).__name__}: {exc}")

    def close(self):
        if not self.closed:
            self.stream.close()
            self.closed = True
            if not self.failed:
                pending = self.current_target and any(self.current_target["drafts"].values())
                self.status("limit_reached" if self.capped else "incomplete" if pending else "closed")

    def next_execution(self):
        if self.executions >= self.cfg["max_executions"]:
            self.capped = True
            self.status("limit_reached")
            return None
        self.sequence += 1
        self.executions += 1
        return f"dp-{self.ranks['dp']}:execution-{self.sequence}"


def initialize(runner):
    global _session
    cfg = config()
    if cfg is None:
        return
    if _session is not None:
        raise RuntimeError("recorder already initialized")
    check_runtime(runner)
    from vllm.distributed import get_dp_group, get_tp_group, get_ep_group
    dp, tp, ep = get_dp_group(), get_tp_group(), get_ep_group()
    if (dp.world_size, tp.world_size, ep.world_size) != (16, 1, 16):
        raise ValueError("actual process groups differ from DP16/TP1/EP16")
    ranks = dict(dp=dp.rank_in_group, tp=tp.rank_in_group, ep=ep.rank_in_group)
    import torch
    ranks["device"] = int(torch.npu.current_device())
    _session = Session(runner, cfg, ranks)
    arm_graph_capture(runner.vllm_config.model_config.hf_config.num_hidden_layers + 8, synchronize)
    atexit.register(close)


def close():
    global _session
    if _session is not None:
        _session.close()
        _session = None
    arm_graph_capture(None)


def capture_routes(layer, ids, weights, prepare, num_experts, forbidden=False):
    if _session is None or not route_recording_active():
        return
    if forbidden or getattr(layer, "zero_expert_num", 0):
        raise ValueError("forced expert rebalance and zero experts are outside this debug profile")
    name = layer.layer_name
    info = dict(prepare=type(prepare).__name__, num_experts=int(num_experts))
    record_graph_routes(name, ids, weights if _session.cfg["save_expert_weights"] else None, info)


def request_snapshot(runner):
    requests = list(runner.input_batch.req_ids)
    if len(set(requests)) != len(requests):
        raise ValueError("duplicate batch requests")
    return requests, {req: runner.requests[req].num_prompt_tokens for req in requests}


def target_rows(requests, prompts, counts, drafts, ids, positions):
    if set(requests) != set(counts) or sum(counts.values()) > len(ids) or len(ids) != len(positions):
        raise ValueError("target batch metadata differs from actual device inputs")
    rows, offset = [], 0
    for req in requests:
        count = counts[req]
        k = len(drafts.get(req, []))
        if type(count) is not int or count < 1 or k >= count:
            raise ValueError("invalid scheduled request token count")
        for index in range(count):
            i = offset + index
            if ids[i] < 0 or positions[i] < 0:
                raise ValueError("invalid actual target input")
            if index and positions[i] != positions[i-1] + 1:
                raise ValueError("target positions are not contiguous within a request")
            candidate = index - (count-k)
            if positions[i] >= prompts[req]:
                rows.append(dict(request_id=req, input_row=i, token_position=positions[i],
                                 token_id=ids[i], phase="decode",
                                 kind="target_verify" if k else "target_decode",
                                 candidate_index=candidate if candidate >= 0 and k else None))
        offset += count
    return rows


def route_records(session, snapshots, rows, input_size, dp_sizes, padded):
    from ..distributed.debug_layout import selector_rows
    records, seen = [], set()
    for snapshot in snapshots:
        if snapshot.layer in seen:
            raise ValueError("duplicate routing layer in one execution")
        seen.add(snapshot.layer)
        info = snapshot.metadata
        if info is None:
            raise ValueError("routing snapshot lacks its captured prepare layout")
        local_size = dp_sizes[session.ranks["dp"]] if info["prepare"].endswith("AllGather") and dp_sizes else input_size
        mapping = selector_rows(info["prepare"], input_size=local_size,
            valid_indices=[r["input_row"] for r in rows], tp_size=1, tp_rank=0,
            dp_rank=session.ranks["dp"], output_size=len(snapshot.expert_ids),
            dp_sizes=dp_sizes, padded_size=padded) if rows else []
        lookup = {r["input_row"]: r for r in rows}
        indices = [j for _, j in mapping]
        selected = host(snapshot.expert_ids[indices])
        weights = host(snapshot.weights[indices].float()) if snapshot.weights is not None else None
        match = re.search(r"(?:^|\.)(layers|mtp)\.(\d+)(?:\.|$)", snapshot.layer)
        if match is None:
            raise ValueError("cannot identify original model layer from routing name")
        for n, (i, _) in enumerate(mapping):
            experts = selected[n]
            if (not experts or len(set(experts)) != len(experts)
                    or any(type(e) is not int or not 0 <= e < info["num_experts"] for e in experts)):
                raise ValueError("invalid logical expert IDs")
            row = dict(lookup[i])
            if not session.cfg["save_token_ids"]:
                row.pop("token_id", None)
            row.update(layer_name=snapshot.layer, layer_id=int(match[2]),
                       num_experts=info["num_experts"], expert_ids=experts)
            if weights is not None:
                row["expert_weights"] = weights[n]
            records.append(row)
    if rows and not snapshots:
        raise ValueError("no W8A8 routes observed; graph/compiler path bypassed hooks")
    return records


@contextmanager
def forward(session, scope, ids, positions, row_builder, metadata):
    execution = session.next_execution()
    if execution is None or session.capped:
        yield None
        return
    # Snapshot inputs before a later asynchronous scheduling iteration mutates buffers.
    actual_ids, actual_positions = ids.clone(), positions.clone()
    try:
        with execution_routes(session.runner.vllm_config.model_config.hf_config.num_hidden_layers + 8) as snapshots:
            yield execution
        # Covers both route producer streams and graph replay before host access.
        synchronize()
        actual, pos = host(actual_ids), host(actual_positions)
        rows = row_builder(actual, pos)
        from vllm.forward_context import get_forward_context
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        context = get_forward_context()
        dp = getattr(context, "dp_metadata", None)
        sizes = dp.num_tokens_across_dp_cpu.tolist() if dp else None
        expected = session.runner.vllm_config.model_config.hf_config.num_hidden_layers if scope == "target" else 1
        if len(snapshots) != expected:
            raise ValueError(f"incomplete {scope} routing coverage: expected {expected}, got {len(snapshots)}")
        routes = route_records(session, snapshots, rows, len(actual), sizes,
                               getattr(_EXTRA_CTX, "padded_num_tokens", None))
        event = dict(type="execution", schema_version=1, execution_id=execution, scope=scope,
                     scheduler_step=session.scheduler_step, host_time_ns=time.time_ns(),
                     input_rows=len(actual), valid_decode_rows=len(rows),
                     route_records=len(routes), routes=routes, **metadata)
        if any(sum(r["input_row"] == row["input_row"] for r in routes) != expected for row in rows):
            raise ValueError("layer coverage differs between actual input rows")
        session.emit(event)
        session.last_forward = dict(execution_id=execution, rows=rows, ids=actual, positions=pos)
    except BaseException as exc:
        session.fail(exc)
        raise


@contextmanager
def trace_target(runner, scheduler, ids, positions, mode):
    session = _session
    if session is None or session.capped:
        yield
        return
    session.scheduler_step += 1
    requests, prompts = request_snapshot(runner)
    counts = dict(scheduler.num_scheduled_tokens)
    drafts = {req: list(values) for req, values in scheduler.scheduled_spec_decode_tokens.items()}
    if sum(counts.values()) != scheduler.total_num_scheduled_tokens:
        raise ValueError("scheduler count sum mismatch")
    builder = lambda values, pos: target_rows(requests, prompts, counts, drafts, values, pos)
    with forward(session, "target", ids, positions, builder,
                 dict(graph_mode=getattr(mode, "name", str(mode)), requests=requests,
                      scheduled_counts=counts, scheduled_draft_counts={r: len(v) for r,v in drafts.items()})):
        yield
    if not session.capped:
        session.current_target = dict(session.last_forward, requests=requests, counts=counts, drafts=drafts)
        session.status("recording")


def trace_sample(function):
    @wraps(function)
    def wrapped(runner, logits, spec_decode_metadata):
        result = function(runner, logits, spec_decode_metadata)
        session = _session
        if session is None or session.capped:
            return result
        try:
            target = session.current_target
            if target is None:
                raise ValueError("sampler has no frozen target execution")
            synchronize()
            samples = host(result.sampled_token_ids)
            lengths = list(spec_decode_metadata.num_draft_tokens) if spec_decode_metadata is not None else [0]*len(target["requests"])
            draft_ids = host(spec_decode_metadata.draft_token_ids) if spec_decode_metadata is not None else []
            if len(samples) < len(target["requests"]) or len(lengths) != len(target["requests"]):
                raise ValueError("sampler rows differ from frozen execution")
            decisions, offset = [], 0
            for index, req in enumerate(target["requests"]):
                k = lengths[index]
                valid = [v for v in samples[index] if v >= 0]
                if samples[index][:len(valid)] != valid or any(v != -1 for v in samples[index][len(valid):]):
                    raise ValueError("sampler padding must follow the valid token prefix")
                candidates = draft_ids[offset:offset+k]
                offset += k
                if k and (not 1 <= len(valid) <= k+1 or valid[:-1] != candidates[:len(valid)-1]):
                    raise ValueError("invalid rejection sampler accepted prefix")
                executed = [r for r in target["rows"] if r["request_id"] == req and r["candidate_index"] is not None]
                if len(executed) != k:
                    raise ValueError("sampler candidate count differs from actual target inputs")
                previous = session.pending.get(req)
                for node, row in enumerate(executed):
                    if row["token_id"] != candidates[node]:
                        raise ValueError("sampler candidate token differs from actual input")
                    linked = previous is not None and node < len(previous["tokens"]) and previous["tokens"][node] == row["token_id"] and previous["positions"][node] == row["token_position"]
                    decisions.append(dict(request_id=req, input_row=row["input_row"],
                        candidate_index=node, accepted=node < len(valid)-1,
                        status="accepted" if node < len(valid)-1 else "rejected" if node == len(valid)-1 else "not_adopted",
                        candidate_round=previous["round"] if linked else None,
                        producer_link="matched" if linked else "unresolved"))
            if offset != len(draft_ids):
                raise ValueError("extra sampler draft IDs")
            session.emit(dict(type="sampler", execution_id=target["execution_id"],
                              decisions=decisions, valid_sample_counts=[sum(v >= 0 for v in row) for row in samples]))
            session.current_target = None
            session.status("limit_reached" if session.capped else "recording")
        except BaseException as exc:
            session.fail(exc)
            raise
        return result
    return wrapped


def trace_proposal(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(proposer, *args, **kwargs):
        session = _session
        if session is None or session.capped:
            return function(proposer, *args, **kwargs)
        bound = signature.bind(proposer, *args, **kwargs)
        if bound.arguments.get("scheduler_output") is None:
            return function(proposer, *args, **kwargs)  # dummy/profiling proposal
        requests, prompts = request_snapshot(proposer.runner)
        state = dict(requests=requests, prompts=prompts, forwards=[], step=0)
        token = _proposal.set(state)
        try:
            result = function(proposer, *args, **kwargs)
            synchronize()
            values = host(result)
            if session.capped:
                return result
            if len(values) < len(requests) or not state["forwards"]:
                raise ValueError("MTP output has no actual producer forward")
            k = proposer.num_speculative_tokens
            if len(state["forwards"]) != k or any(len(row) != k for row in values):
                raise ValueError("MTP output K differs from captured producer forwards")
            round_id = state["forwards"][0]["execution_id"]
            candidates = []
            for index, req in enumerate(requests):
                positions = []
                for step, captured in enumerate(state["forwards"]):
                    rows = [r for r in captured["rows"] if r["request_id"] == req and r.get("producer")]
                    if not rows:
                        continue  # discarded request has no candidate attribution
                    if len(rows) != 1:
                        raise ValueError("ambiguous MTP producer")
                    row = rows[0]
                    if step:
                        first = next(r for r in state["forwards"][0]["rows"]
                                     if r["request_id"] == req and r.get("producer"))
                        if (row["token_position"] != first["token_position"]+step
                                or row["token_id"] != values[index][step-1]):
                            raise ValueError("MTP later-step input differs from its actual predecessor or position was clamped")
                    position = row["token_position"] + 2
                    positions.append(position)
                    item = dict(request_id=req, candidate_index=step, token_position=position,
                                producer_execution=captured["execution_id"], producer_input_row=row["input_row"])
                    if session.cfg["save_token_ids"]:
                        item["token_id"] = values[index][step]
                    candidates.append(item)
                if positions:
                    if len(positions) != k:
                        raise ValueError("MTP candidate lacks one of its producer steps")
                    session.pending[req] = dict(round=round_id, positions=positions, tokens=list(values[index]))
            # The framework retires finished requests; do not retain their candidates indefinitely.
            session.pending = {r: v for r,v in session.pending.items() if r in proposer.runner.requests}
            session.emit(dict(type="proposal", candidate_round=round_id, k=k, candidates=candidates))
            return result
        except BaseException as exc:
            session.fail(exc)
            raise
        finally:
            _proposal.reset(token)
    return wrapped


@contextmanager
def trace_mtp_forward(proposer, model_kwargs, num_tokens, batch_size, sample_indices, step):
    session, proposal = _session, _proposal.get()
    if session is None or session.capped or proposal is None:
        yield
        return
    requests = proposal["requests"]
    # These small metadata tensors are frozen before the original forward.
    synchronize()
    starts = host(proposer.query_start_loc_group[0][:len(requests)+1]) if step == 0 else list(range(len(requests)+1))
    selected = host(sample_indices[:len(requests)]) if step == 0 else list(range(len(requests)))
    discarded = set(proposer.runner.discard_request_indices.np[:proposer.runner.num_discarded_requests].tolist())
    if (batch_size < len(requests) or len(starts) != len(requests)+1 or starts[-1] > num_tokens
            or starts[0] != 0 or len(selected) != len(requests)
            or any(a >= b for a,b in zip(starts, starts[1:]))):
        raise ValueError("MTP request layout exceeds actual input")
    def build(values, positions):
        rows = []
        for index, req in enumerate(requests):
            if index in discarded:
                continue
            start, end, last = starts[index], starts[index+1], selected[index]
            if not 0 <= start <= last < end <= len(values):
                raise ValueError("MTP sample index outside request input span")
            for i in range(start, last+1):
                if values[i] < 0 or positions[i] < 0:
                    raise ValueError("negative actual MTP input")
                if positions[i]+1 >= proposal["prompts"][req]:
                    rows.append(dict(request_id=req, input_row=i, token_position=positions[i],
                        input_sequence_position=positions[i]+1, token_id=values[i], phase="decode",
                        kind="mtp_draft", candidate_index=step, producer=i == last))
        return rows
    with forward(session, "mtp", model_kwargs["input_ids"], model_kwargs["positions"], build,
                 dict(graph_mode="NONE", proposal_step=step, requests=requests)):
        yield
    if not session.capped:
        proposal["forwards"].append(session.last_forward)
