"""Qwen3.6 text-only eager W8A8 Decode capture; NPU acceptance pending."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import json
import os
from pathlib import Path
import re

from .ascend_v41_debug import build_inputs
from ..distributed.debug_layout import selector_rows
from ..distributed.decode import Worker, Route
from ..distributed.storage import create_run, WorkerWriter
from ..errors import ConfigError, IntegrityError

PROFILE = "qwen36-single-node-debug"
_active = ContextVar("qwen36_capture", default=None)
_session = None


def read_config():
    path = os.environ.get("MOE_TRACE_QWEN36_CONFIG")
    if not path:
        return None
    cfg = json.loads(Path(path).read_text())
    if set(cfg) != {"enabled", "output_root", "run_id", "node_id", "save_token_ids", "save_expert_weights"}:
        raise ConfigError("invalid Qwen capture keys")
    if any(type(cfg[k]) is not bool for k in ("enabled", "save_token_ids", "save_expert_weights")):
        raise ConfigError("capture flags must be bool")
    if not cfg["enabled"]:
        return None
    for key in ("run_id", "node_id"):
        if not isinstance(cfg[key], str) or cfg[key] in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg[key]):
            raise ConfigError("unsafe capture identity")
    if not isinstance(cfg["output_root"], str) or not Path(cfg["output_root"]).is_absolute():
        raise ConfigError("output_root must be absolute")
    if any(os.environ.get(k) for k in ("MOE_TRACE_CONFIG", "MOE_TRACE_V41_DEBUG_CONFIG", "MOE_TRACE_V41_SOURCE_HOOKS")):
        raise ConfigError("do not combine capture adapters")
    return cfg


def check_runtime(runner, mode):
    cfg = runner.vllm_config
    if (getattr(mode, "name", None) != "NONE" or not cfg.model_config.enforce_eager
            or cfg.scheduler_config.async_scheduling or cfg.speculative_config is not None
            or getattr(cfg, "use_v2_model_runner", False)):
        raise ConfigError("Qwen capture requires V1 eager synchronous without speculation")
    p = cfg.parallel_config
    for field, expected in {"data_parallel_size": 8, "tensor_parallel_size": 2,
                            "pipeline_parallel_size": 1, "prefill_context_parallel_size": 1,
                            "decode_context_parallel_size": 1}.items():
        if getattr(p, field, 1) != expected:
            raise ConfigError(f"requires {field}={expected}")
    if not p.enable_expert_parallel or getattr(p, "use_ubatching", False):
        raise ConfigError("requires EP without microbatching")
    if (getattr(cfg, "kv_transfer_config", None) or getattr(cfg, "lora_config", None)
            or cfg.cache_config.enable_prefix_caching
            or getattr(cfg.model_config, "enable_return_routed_experts", False)):
        raise ConfigError("requires colocated without KV connector, LoRA, prefix cache or built-in capture")
    if cfg.model_config.quantization != "ascend":
        raise ConfigError("requires Ascend W8A8")
    if cfg.model_config.hf_config.architectures != ["Qwen3_5MoeForConditionalGeneration"]:
        raise ConfigError("unverified Qwen architecture")
    a = runner.ascend_config
    if (getattr(runner, "dynamic_eplb", False) or getattr(runner, "enable_enpu", False)
            or getattr(a.eplb_config, "dynamic_eplb", False)
            or any(getattr(a, k, False) for k in ("enable_force_eplb", "enable_fused_mc2",
               "enable_flashcomm1", "enable_dsa_cp", "enable_shared_expert_dp",
               "multistream_overlap_gate", "multistream_overlap_shared_expert"))):
        raise ConfigError("EPLB, fused MC2, FlashComm, shared DP and multistream unsupported")


class Session:
    def __init__(self, runner, cfg, worker):
        self.cfg, self.worker, self.step = cfg, worker, 0
        self.layers, self.aliases = {}, {}
        for _, module in runner.model.named_modules():
            name = getattr(module, "layer_name", None)
            if name and getattr(module, "router", None) is not None and hasattr(module, "moe_config"):
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if not match or name in self.layers:
                    raise ConfigError("unknown or duplicate MoE layer")
                self.layers[name] = int(match[1])
                routed = getattr(module, "routed_experts", module)
                self.aliases[id(routed)] = name
        if not self.layers or len(set(self.layers.values())) != len(self.layers):
            raise ConfigError("cannot identify unique MoE layers")
        self.root = Path(cfg["output_root"]) / cfg["run_id"] / worker.worker / "trace"
        create_run(self.root, [worker], {"target": sorted(self.layers.values())},
                   validation_level="a3_unverified_debug")
        self.writer = WorkerWriter(self.root, worker, shard_records=256)


def get_session(runner, cfg):
    global _session
    if _session is None:
        from vllm.distributed import get_dp_group, get_tp_group, get_ep_group
        dp, tp, ep = get_dp_group(), get_tp_group(), get_ep_group()
        d, t, e = dp.rank_in_group, tp.rank_in_group, ep.rank_in_group
        if (dp.world_size, tp.world_size, ep.world_size) != (8, 2, 16) or e != d * 2 + t:
            raise ConfigError("unsupported actual DP/TP/EP topology")
        worker = Worker(PROFILE, cfg["run_id"], cfg["node_id"], f"dp-{d}-tp-{t}", d, t, e, e)
        _session = Session(runner, cfg, worker)
    return _session


def initialize(runner):
    cfg = read_config()
    if cfg:
        check_runtime(runner, runner.vllm_config.compilation_config.cudagraph_mode)
        get_session(runner, cfg)


def text_positions(positions):
    import torch
    if positions.ndim == 2 and positions.shape[0] == 3:
        if not torch.equal(positions[0], positions[1]) or not torch.equal(positions[0], positions[2]):
            raise ConfigError("non-text multimodal positions unsupported")
        return positions[0]
    return positions


@contextmanager
def trace_target(runner, scheduler, ids, positions, mode):
    cfg = read_config()
    if cfg is None:
        yield
        return
    check_runtime(runner, mode)
    session = get_session(runner, cfg)
    if _active.get() is not None:
        raise IntegrityError("nested capture")
    session.step += 1
    token = None
    try:
        rows = build_inputs(runner, scheduler, ids, text_positions(positions), session.step,
                            f"Qwen-dp-{session.worker.dp_rank}")
        state = dict(session=session, rows=rows, input_size=len(ids), routes={}, consumed=set(), tensors={})
        token = _active.set(state)
        yield
        if set(state["routes"]) != set(session.layers) or state["consumed"] != set(session.layers):
            raise IntegrityError("missing MoE layer or consumer")
        ownership = [{r.row.input_row for r in group} for group in state["routes"].values()]
        if any(v != ownership[0] for v in ownership):
            raise IntegrityError("ownership changed between layers")
        records = [r for group in state["routes"].values() for r in group]
        if not cfg["save_token_ids"]:
            records = [replace(r, row=replace(r.row, token_id=None, input_role="redacted")) for r in records]
        if records:
            session.writer.declare_execution("target", list({r.row.input_row: r.row for r in records}.values()))
            for record in records:
                session.writer.append(record)
    except BaseException:
        session.writer.fail()
        raise
    finally:
        if token is not None:
            _active.reset(token)


def capture_routes(layer, ids, weights, prepare, extra, context, num_experts, forbidden=False):
    state = _active.get()
    if state is None:
        return
    import torch
    session = state["session"]
    name = session.aliases.get(id(layer))
    if (forbidden or not name or name in state["routes"] or ids.ndim != 2
            or ids.dtype not in (torch.int32, torch.int64) or ids.shape != weights.shape):
        raise IntegrityError("invalid logical routing path")
    if (getattr(extra, "flash_comm_v1_enabled", False) or getattr(prepare, "replace_allreduce", False)
            or getattr(prepare, "enable_shared_expert_dp", False)):
        raise ConfigError("SP and shared-DP prepare unsupported in first Qwen profile")
    meta = getattr(context, "dp_metadata", None)
    dp_sizes = meta.num_tokens_across_dp_cpu.tolist() if meta else None
    size = state["input_size"]
    if type(prepare).__name__ == "PrepareAndFinalizeWithAllGather" and dp_sizes:
        # Forward padding may exceed actual local DP tokens. Only real rows are mapped.
        size = dp_sizes[session.worker.dp_rank]
        if any(i >= size for i in state["rows"]):
            raise IntegrityError("DP metadata excludes a valid input")
    indices = selector_rows(type(prepare).__name__, input_size=size, valid_indices=state["rows"],
        tp_size=2, tp_rank=session.worker.tp_rank, dp_rank=session.worker.dp_rank,
        output_size=len(ids), padded_size=getattr(extra, "padded_num_tokens", None), dp_sizes=dp_sizes)
    index = [j for _, j in indices]
    selected = ids[index].detach().cpu()
    scales = weights[index].detach().float().cpu() if cfg_weights(session) else None
    state["routes"][name] = [Route(session.worker, session.worker, state["rows"][i], "target",
        session.layers[name], int(num_experts), tuple(selected[k].tolist()),
        tuple(scales[k].tolist()) if scales is not None else None) for k, (i, _) in enumerate(indices)]
    state["tensors"][name] = (index, selected)


def cfg_weights(session):
    return session.cfg["save_expert_weights"]


def capture_consumer(layer, ids):
    state = _active.get()
    if state is None:
        return
    import torch
    name = state["session"].aliases.get(id(layer))
    if name not in state["tensors"] or name in state["consumed"]:
        raise IntegrityError("consumer without unique selector")
    index, selected = state["tensors"][name]
    if ids.ndim != 2 or any(j >= len(ids) for j in index) or not torch.equal(ids[index].detach().cpu(), selected):
        raise IntegrityError("consumer experts differ from captured logical IDs")
    state["consumed"].add(name)


def close():
    global _session
    if _session:
        if not _session.writer.failed:
            _session.writer.close()
        _session = None
