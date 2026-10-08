"""Eager synchronous D-side debug collector. A3 hardware validation pending.

No new distributed collective is added; selector rows are mapped using the
actual prepare implementation/state. DSpark, graphs and async are rejected.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import json
import os
from pathlib import Path
import re

from ..distributed.decode import Worker, DecodeRow, Route
from ..distributed.debug_layout import selector_rows
from ..distributed.storage import create_run, WorkerWriter
from ..errors import ConfigError, IntegrityError
from ..io import atomic_json, append_jsonl

_active = ContextVar("ascend_v41_debug", default=None)
_session = None


def read_config():
    path = os.environ.get("MOE_TRACE_V41_DEBUG_CONFIG")
    if not path:
        return None
    cfg = json.loads(Path(path).read_text())
    required = {"enabled", "output_root", "run_id", "node_id", "save_token_ids", "save_expert_weights"}
    if set(cfg) != required or any(type(cfg[k]) is not bool for k in ("enabled", "save_token_ids", "save_expert_weights")):
        raise ConfigError("invalid V4.1 debug configuration")
    if not cfg["enabled"]:
        return None
    for key in ("run_id", "node_id"):
        if (not isinstance(cfg[key], str) or cfg[key] in {".", ".."}
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg[key])):
            raise ConfigError(f"unsafe {key}")
    if not isinstance(cfg["output_root"], str) or not cfg["output_root"]:
        raise ConfigError("output_root required")
    return cfg


def check_runtime(runner, mode):
    cfg = runner.vllm_config
    if (getattr(mode, "name", None) != "NONE" or not cfg.model_config.enforce_eager
            or cfg.scheduler_config.async_scheduling or cfg.speculative_config is not None
            or getattr(cfg, "use_v2_model_runner", False)):
        raise ConfigError("debug profile requires V1, eager, synchronous, no DSpark")
    parallel = cfg.parallel_config
    for field, expected in {"tensor_parallel_size": 2, "data_parallel_size": 8,
                            "pipeline_parallel_size": 1, "prefill_context_parallel_size": 1,
                            "decode_context_parallel_size": 1}.items():
        if getattr(parallel, field, 1) != expected:
            raise ConfigError(f"debug requires {field}={expected}")
    if not parallel.enable_expert_parallel or getattr(parallel, "use_ubatching", False):
        raise ConfigError("debug requires EP without microbatching")
    kv = cfg.kv_transfer_config
    if kv is None or kv.kv_role != "kv_consumer" or kv.kv_connector != "MooncakeHybridConnector":
        raise ConfigError("debug requires MooncakeHybridConnector D consumer")
    if (cfg.cache_config.enable_prefix_caching or getattr(cfg, "lora_config", None)
            or getattr(cfg.model_config, "enable_return_routed_experts", False)
            or getattr(runner, "enable_enpu", False)):
        raise ConfigError("prefix caching, LoRA, built-in routed return and ENPU unsupported")
    if cfg.model_config.quantization != "ascend":
        raise ConfigError("debug requires Ascend quantization")
    architectures = cfg.model_config.hf_config.architectures
    if not any("deepseekv41" in re.sub(r"[^a-z0-9]", "", a.lower()) for a in architectures):
        raise ConfigError("debug supports DeepSeek V4.1 only")
    ascend = runner.ascend_config
    if (getattr(runner, "dynamic_eplb", False) or ascend.eplb_config.dynamic_eplb
            or getattr(ascend, "enable_force_eplb", False)
            or any(getattr(ascend, name, False) for name in (
                "enable_fused_mc2", "enable_flashcomm1", "enable_dsa_cp",
                "multistream_overlap_gate", "multistream_overlap_shared_expert"))):
        raise ConfigError("debug requires no EPLB/fused MC2/FlashComm/DSA CP/multistream overlap")


def build_inputs(runner, scheduler, ids, positions, step, engine):
    import torch
    if (ids is None or positions is None or ids.ndim != 1 or positions.shape != ids.shape
            or ids.dtype not in (torch.int32, torch.int64) or positions.dtype not in (torch.int32, torch.int64)):
        raise IntegrityError("integer one-dimensional forward inputs required")
    req_ids = tuple(runner.input_batch.req_ids)
    counts = scheduler.num_scheduled_tokens
    if (len(set(req_ids)) != len(req_ids) or set(req_ids) != set(counts)
            or any(type(v) is not int or v <= 0 for v in counts.values())
            or sum(counts.values()) != scheduler.total_num_scheduled_tokens
            or len(ids) < scheduler.total_num_scheduled_tokens):
        raise IntegrityError("actual batch differs from scheduler")
    actual_ids, actual_pos = ids.detach().cpu().tolist(), positions.detach().cpu().tolist()
    rows, offset = {}, 0
    for index, req_id in enumerate(req_ids):
        request = runner.requests[req_id]
        start = int(runner.input_batch.num_computed_tokens_cpu[index])
        if start < 0 or request.num_prompt_tokens < 1:
            raise IntegrityError("invalid computed position/prompt length")
        for local in range(counts[req_id]):
            i, pos = offset + local, start + local
            if actual_pos[i] != pos or actual_ids[i] < 0 or request.get_token_id(pos) != actual_ids[i]:
                raise IntegrityError(f"actual token/position mismatch: {req_id}:{pos}")
            if pos >= request.num_prompt_tokens:
                rows[i] = DecodeRow(engine, f"execution-{step}", step, f"{engine}:{req_id}",
                                    req_id, i, pos, actual_ids[i], "target_decode")
        offset += counts[req_id]
    return rows


class DebugSession:
    def __init__(self, runner, cfg, worker):
        self.worker, self.cfg, self.step = worker, cfg, 0
        self.layers = {}
        for _, module in runner.model.named_modules():
            name = getattr(module, "layer_name", None)
            if name and getattr(module, "router", None) is not None and hasattr(module, "moe_config"):
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if not match or name in self.layers:
                    raise ConfigError("unidentifiable/duplicate MoE layer")
                self.layers[name] = int(match.group(1))
        if not self.layers or len(set(self.layers.values())) != len(self.layers):
            raise ConfigError("cannot discover unique original MoE layers")
        self.directory = Path(cfg["output_root"]) / cfg["run_id"] / f"dp-{worker.dp_rank}-tp-{worker.tp_rank}"
        root = create_run(self.directory / "trace", [worker], {"target": sorted(self.layers.values())},
                          validation_level="a3_unverified_debug")
        self.writer = WorkerWriter(root, worker, shard_records=256)
        self.consumer_path = self.directory / "consumer-debug.jsonl"
        self.consumer_path.touch(exist_ok=False)
        atomic_json(self.directory / "debug-metadata.json", {
            "worker": asdict(worker), "profile": "1P1D-debug", "npu_verified": False,
            "request_identity": "D-local namespaced; proxy/P ID mapping not verified",
            "phase": "decode only; prompt/KV bootstrap positions excluded",
            "consumer_file": "diagnostic only; IDs may be physical and include shared experts",
            "save_token_ids": cfg["save_token_ids"], "save_expert_weights": cfg["save_expert_weights"],
            "requests_terminal_verified": False})


def _get_session(runner, cfg):
    global _session
    if _session is None:
        from vllm.distributed import get_tp_group, get_dp_group, get_ep_group
        dp, tp, ep = get_dp_group(), get_tp_group(), get_ep_group()
        if (dp.world_size, tp.world_size, ep.world_size) != (8, 2, 16):
            raise ConfigError("actual DP/TP/EP group sizes differ from debug topology")
        if ep.rank_in_group != dp.rank_in_group * 2 + tp.rank_in_group:
            raise ConfigError("unsupported EP rank ordering")
        worker = Worker("ascend-v41-debug", cfg["run_id"], cfg["node_id"],
                        f"dp-{dp.rank_in_group}-tp-{tp.rank_in_group}", dp.rank_in_group,
                        tp.rank_in_group, ep.rank_in_group, ep.rank_in_group)
        _session = DebugSession(runner, cfg, worker)
    return _session


def initialize_debug(runner):
    cfg = read_config()
    if cfg is not None:
        check_runtime(runner, runner.vllm_config.compilation_config.cudagraph_mode)
        _get_session(runner, cfg)  # Include idle workers in the declared deployment.


@contextmanager
def trace_target(runner, scheduler, input_ids, positions, mode):
    cfg = read_config()
    if cfg is None:
        yield
        return
    check_runtime(runner, mode)
    session = _get_session(runner, cfg)
    if _active.get() is not None:
        raise IntegrityError("nested debug forward")
    session.step += 1
    token = None
    try:
        rows = build_inputs(runner, scheduler, input_ids, positions, session.step,
                            f"D-dp-{session.worker.dp_rank}")
        state = {"session": session, "rows": rows, "input_size": len(input_ids),
                 "routes": {}, "indices": {}, "consumer": {}}
        token = _active.set(state)
        yield
        if set(state["routes"]) != set(session.layers) or set(state["consumer"]) != set(session.layers):
            raise IntegrityError("declared target layers/consumer hooks did not execute")
        owned = None
        for name, records in state["routes"].items():
            selected = {r.row.input_row for r in records}
            if owned is not None and selected != owned:
                raise IntegrityError("input ownership changed across layers")
            owned = selected
        if owned:
            # Token-ID privacy: validate actual IDs, then omit them from saved rows.
            from dataclasses import replace
            records = [r for group in state["routes"].values() for r in group]
            if not cfg["save_token_ids"]:
                records = [replace(r, row=replace(r.row, token_id=None, input_role="redacted")) for r in records]
            saved_rows = {r.row.input_row: r.row for r in records}
            session.writer.declare_execution("target", list(saved_rows.values()))
            for route in records:
                session.writer.append(route)
            for name, value in state["consumer"].items():
                append_jsonl(session.consumer_path, {"execution": f"execution-{session.step}",
                             "layer_name": name, **value})
    except BaseException:
        session.writer.fail()
        raise
    finally:
        if token is not None:
            _active.reset(token)


def capture_routes(layer, ids, weights, prepare, extra, forward_context, num_experts, force=False):
    state = _active.get()
    if state is None:
        return  # warmup/profile outside actual execute_model scope
    session = state["session"]
    name = layer.layer_name
    if (force or name not in session.layers or name in state["routes"] or ids.ndim != 2
            or weights.shape != ids.shape):
        raise IntegrityError("invalid/duplicate logical routing")
    meta = getattr(forward_context, "dp_metadata", None)
    dp_sizes = meta.num_tokens_across_dp_cpu.tolist() if meta is not None else None
    ep_sizes = None
    sequence_parallel = layer.moe_config.is_sequence_parallel
    if sequence_parallel and type(prepare).__name__ == "PrepareAndFinalizeWithAllGather":
        from vllm_ascend.ops.register_custom_ops import _get_ep_local_sizes
        from vllm.distributed import get_ep_group
        ep_sizes = _get_ep_local_sizes(meta, get_ep_group())
    mapping_size = state["input_size"]
    if sequence_parallel and type(prepare).__name__ == "PrepareAndFinalizeWithAllGather" and dp_sizes:
        actual = dp_sizes[session.worker.dp_rank]
        if ((actual + 1) // 2 != (mapping_size + 1) // 2
                or any(i >= actual for i in state["rows"])):
            raise IntegrityError("SP metadata does not match actual forward shard size")
        mapping_size = actual
    indices = selector_rows(type(prepare).__name__, input_size=mapping_size,
                            valid_indices=state["rows"], tp_size=2, tp_rank=session.worker.tp_rank,
                            dp_rank=session.worker.dp_rank, output_size=len(ids),
                            sequence_parallel=sequence_parallel,
                            padded_size=getattr(extra, "padded_num_tokens", None),
                            dp_sizes=dp_sizes, ep_sizes=ep_sizes)
    import torch
    if ids.dtype not in (torch.int32, torch.int64):
        raise IntegrityError("logical expert IDs must be integers")
    cpu_ids = ids[[j for _, j in indices]].detach().cpu().tolist()
    cpu_weights = weights[[j for _, j in indices]].detach().float().cpu().tolist() if session.cfg["save_expert_weights"] else None
    state["routes"][name] = [Route(session.worker, session.worker, state["rows"][i], "target",
                                    session.layers[name], int(num_experts), tuple(cpu_ids[k]),
                                    tuple(cpu_weights[k]) if cpu_weights is not None else None)
                             for k, (i, _) in enumerate(indices)]
    state["indices"][name] = indices


def capture_consumer(layer, ids, weights):
    state = _active.get()
    if state is None:
        return
    name = layer.layer_name
    if name not in state["indices"] or name in state["consumer"] or weights.shape != ids.shape:
        raise IntegrityError("consumer does not match logical route hook")
    indices = state["indices"][name]
    if any(j >= len(ids) for _, j in indices):
        raise IntegrityError("consumer input row layout changed")
    state["consumer"][name] = {
        "input_rows": [i for i, _ in indices],
        "expert_ids": ids[[j for _, j in indices]].detach().cpu().tolist(),
        "expert_weights": weights[[j for _, j in indices]].detach().float().cpu().tolist()
        if state["session"].cfg["save_expert_weights"] else None,
        "semantics": "actual W8A8 consumer IDs after mapping; shared columns may be present"}


def close_debug():
    global _session
    if _session is not None:
        if not _session.writer.failed:
            _session.writer.close()
        _session = None
