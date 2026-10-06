"""Experimental Ascend V1 hooks; CPU verified, real NPU acceptance pending."""
from __future__ import annotations

import contextvars
import importlib.metadata
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from ..config import load_config
from ..errors import ConfigError, IntegrityError
from ..session import InputRow, TraceSession

ADAPTER = "ascend_vllm_0_23_0"
VLLM_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
ASCEND_COMMIT = "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd"
_config = _session = None
_counter = 0
_request_labels = {}
_active = contextvars.ContextVar(ADAPTER, default=None)
_layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass
class Execution:
    session: TraceSession
    step: int
    identity: str
    rows: list[InputRow]
    routes: dict = field(default_factory=dict)


def _get_config():
    global _config
    if _config is None:
        path = os.environ.get("MOE_TRACE_CONFIG")
        if not path:
            return None
        _config = load_config(path)
    return _config if _config.enabled else None


def check_versions(vllm_version, ascend_version):
    if vllm_version != "0.23.0" or ascend_version != "0.23.0":
        raise ConfigError("Ascend tracing requires vllm=0.23.0 and vllm-ascend=0.23.0")


def _reject_unsupported(owner, config):
    cfg = owner.vllm_config
    expected = dict(enforce_eager=True, enable_prefix_caching=False,
                    speculative_model=None, tensor_parallel_size=1,
                    pipeline_parallel_size=1, data_parallel_size=1, async_scheduling=False)
    for key, value in expected.items():
        if key in config.vllm and config.vllm[key] != value:
            raise ConfigError(f"Ascend tracing requires vllm.{key}={value!r}")
    if not cfg.model_config.enforce_eager:
        raise ConfigError("Ascend tracing requires enforce_eager; graph replay is unsupported")
    if cfg.cache_config.enable_prefix_caching or cfg.speculative_config is not None:
        raise ConfigError("prefix caching and speculative decoding are unsupported")
    if cfg.scheduler_config.async_scheduling:
        raise ConfigError("Ascend tracing requires synchronous scheduling")
    if getattr(cfg, "use_v2_model_runner", False):
        raise ConfigError("Ascend tracing requires VLLM_USE_V2_MODEL_RUNNER=0")
    if getattr(cfg.model_config, "enable_return_routed_experts", False):
        raise ConfigError("disable enable_return_routed_experts when using moe-trace")
    if getattr(cfg.model_config, "quantization", None) or getattr(cfg, "lora_config", None):
        raise ConfigError("quantization and LoRA are unsupported")
    if getattr(cfg.model_config, "enable_prompt_embeds", False):
        raise ConfigError("prompt embeddings are unsupported")
    parallel = cfg.parallel_config
    if parallel.distributed_executor_backend != "uni":
        raise ConfigError("Ascend tracing requires the uni executor")
    for attr in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, attr, 1) != 1:
            raise ConfigError(f"Ascend tracing requires {attr}=1")
    if (getattr(parallel, "enable_expert_parallel", False) or getattr(parallel, "use_ubatching", False)
            or getattr(parallel, "use_sequence_parallel_moe", False)):
        raise ConfigError("expert/sequence parallelism and microbatching are unsupported")
    if getattr(cfg, "kv_transfer_config", None) or getattr(cfg, "ec_transfer_config", None):
        raise ConfigError("disaggregated execution is unsupported")
    if cfg.model_config.hf_config.architectures != ["GraniteMoeForCausalLM"]:
        raise ConfigError("Ascend CPU baseline supports GraniteMoeForCausalLM only")
    if os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", ""):
        raise ConfigError("routing simulation is unsupported in real execution")


def _check_runner(runner):
    ascend = runner.ascend_config
    for name in ("enable_flashcomm1", "enable_flashcomm2_parallel_size", "enable_fused_mc2",
                 "multistream_overlap_gate", "multistream_overlap_shared_expert",
                 "enable_balance_scheduling"):
        if getattr(ascend, name, False):
            raise ConfigError(f"Ascend tracing does not support {name}")
    if ascend.eplb_config.dynamic_eplb or ascend.profiling_chunk_config.enabled:
        raise ConfigError("dynamic EPLB and profiling chunk scheduling are unsupported")
    if getattr(runner, "use_async_scheduling", False) or getattr(runner, "enable_enpu", False):
        raise ConfigError("asynchronous scheduling and ENPU are unsupported")
    mode = getattr(runner.compilation_config.cudagraph_mode, "name", None)
    if mode != "NONE":
        raise ConfigError("Ascend tracing requires graph mode NONE")


def _get_session(owner, config):
    global _session
    if _session is None:
        _reject_unsupported(owner, config)
        check_versions(importlib.metadata.version("vllm"),
                       importlib.metadata.version("vllm-ascend"))
        hf = owner.vllm_config.model_config.hf_config
        layers = tuple(range(int(hf.num_hidden_layers)))
        _session = TraceSession(
            config, model={"name_or_path": str(owner.vllm_config.model_config.model),
                           "architecture": "GraniteMoeForCausalLM"},
            moe_layers=layers, experts_per_layer={i: int(hf.num_local_experts) for i in layers},
            top_k=int(hf.num_experts_per_tok),
            expert_weight_semantics="Actual Ascend fused_experts weights after dtype conversion; not re-normalized",
            framework={"adapter": ADAPTER, "backend": "ascend", "vllm_version": "0.23.0",
                       "vllm_ascend_version": "0.23.0", "validation_level": "npu_unverified",
                       "vllm_base_commit": VLLM_COMMIT, "ascend_base_commit": ASCEND_COMMIT})
    return _session


def request_started(owner, request):
    config = _get_config()
    if config is None:
        return
    session = _get_session(owner, config)
    try:
        if getattr(request, "resumable", False):
            raise ConfigError("resumable requests are unsupported")
        label = getattr(request, "moe_trace_label", None)
        _request_labels[request.request_id] = label
        if config.request_labels is None or label in config.request_labels:
            session.request_event(request.request_id, "started", client_label=label)
    except Exception as exc:
        session.fail(exc)
        raise


def request_finished(owner, request):
    config = _get_config()
    if config is None:
        return
    label = _request_labels.get(request.request_id, getattr(request, "moe_trace_label", None))
    if config.request_labels is not None and label not in config.request_labels:
        return
    session = _get_session(owner, config)
    try:
        reason = request.get_finished_reason()
        if not reason:
            raise IntegrityError("terminal request lacks finished reason")
        session.request_event(request.request_id,
                              "cancelled" if request.status.name == "FINISHED_ABORTED" else "finished",
                              client_label=label, reason=str(reason),
                              details={"scheduler_status": request.status.name,
                                       "num_prompt_tokens": request.num_prompt_tokens,
                                       "num_output_tokens": request.num_output_tokens})
    except Exception as exc:
        session.fail(exc)
        raise


def build_rows(runner, scheduler_output, input_ids, positions):
    req_ids = runner.input_batch.req_ids
    counts = scheduler_output.num_scheduled_tokens
    if not req_ids or len(set(req_ids)) != len(req_ids) or set(counts) != set(req_ids):
        raise IntegrityError("scheduled requests differ from actual input batch")
    if any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0
           for n in counts.values()):
        raise IntegrityError("scheduled token counts must be positive integers")
    total = sum(int(n) for n in counts.values())
    if total != scheduler_output.total_num_scheduled_tokens:
        raise IntegrityError("scheduled token total mismatch")
    if (input_ids is None or positions is None or input_ids.ndim != 1 or positions.ndim != 1
            or len(input_ids) != total or len(positions) != total):
        raise IntegrityError("only unpadded one-dimensional token-ID inputs are supported")
    if len(runner.input_batch.num_computed_tokens_cpu) < len(req_ids):
        raise IntegrityError("computed token counters are missing")
    ids, actual_positions = input_ids.detach().cpu().numpy(), positions.detach().cpu().numpy()
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(actual_positions.dtype, np.integer):
        raise IntegrityError("input IDs and positions must have integer dtype")
    rows, offset = [], 0
    for index, req_id in enumerate(req_ids):
        request = runner.requests[req_id]
        start = int(runner.input_batch.num_computed_tokens_cpu[index])
        if start < 0 or request.num_prompt_tokens <= 0:
            raise IntegrityError("invalid request position or prompt length")
        for local in range(int(counts[req_id])):
            pos, source = start + local, offset + local
            token_id = int(ids[source])
            if token_id < 0 or int(actual_positions[source]) != pos or token_id != request.get_token_id(pos):
                raise IntegrityError(f"actual forward input mismatch: {req_id}, position {pos}")
            rows.append(InputRow(req_id, pos, "prefill" if pos < request.num_prompt_tokens else "decode",
                                 token_id, _request_labels.get(req_id, getattr(request, "moe_trace_label", None))))
        offset += int(counts[req_id])
    return rows


@contextmanager
def trace_execution(runner, scheduler_output, input_ids, positions):
    global _counter
    config = _get_config()
    if config is None:
        yield
        return
    session = _get_session(runner, config)
    token = None
    try:
        _check_runner(runner)
        if _active.get() is not None:
            raise IntegrityError("nested model execution is unsupported")
        if (getattr(scheduler_output, "preempted_req_ids", None)
                or getattr(scheduler_output.scheduled_cached_reqs, "resumed_req_ids", None)):
            raise ConfigError("preemption/recomputation is unsupported")
        rows = build_rows(runner, scheduler_output, input_ids, positions)
        _counter += 1
        execution = Execution(session, _counter, f"ascend-exec-{_counter:012d}", rows)
        token = _active.set(execution)
        yield
        session.record_execution(step_id=execution.step, execution_id=execution.identity,
                                 rows=rows, routes=execution.routes)
    except Exception as exc:
        session.fail(exc)
        raise
    finally:
        if token is not None:
            _active.reset(token)


def record_routes(layer_name, topk_ids, topk_weights, *, zero_experts=0,
                  force_load_balance=False, dynamic_eplb=False, comm_type="ALLGATHER"):
    execution = _active.get()
    if execution is None:
        return
    if zero_experts or force_load_balance or dynamic_eplb or comm_type != "ALLGATHER":
        raise ConfigError("only unmodified logical routing with ALLGATHER communication is supported")
    match = _layer_pattern.search(layer_name)
    if not match:
        raise IntegrityError(f"cannot identify original layer: {layer_name}")
    layer = int(match.group(1))
    if layer not in execution.session.experts_per_layer:
        raise IntegrityError(f"undeclared MoE layer: {layer}")
    if layer not in execution.session.moe_layers:
        return
    if layer in execution.routes:
        raise IntegrityError(f"duplicate layer execution: {layer}")
    ids = topk_ids.detach().cpu().numpy().copy()
    if not np.issubdtype(ids.dtype, np.integer):
        raise IntegrityError("expert IDs must have integer dtype")
    if ids.shape != (len(execution.rows), execution.session.top_k):
        raise IntegrityError("expert ID tensor shape does not match forward input rows")
    if ids.size and (ids.min() < 0 or ids.max() >= execution.session.experts_per_layer[layer]):
        raise IntegrityError("logical expert ID out of range")
    weights = (topk_weights.detach().float().cpu().numpy().copy()
               if execution.session.config.save_expert_weights else None)
    if weights is not None and (weights.shape != ids.shape or not np.isfinite(weights).all()
                                or (weights < 0).any()):
        raise IntegrityError("expert weights must have matching shape and be finite/non-negative")
    execution.routes[layer] = (ids, weights)


def close_trace():
    global _session
    if _session is not None:
        if not _session._failed:
            _session.close(require_terminal_requests=True)
        _session = None


def reset_for_tests():
    global _config, _session, _counter
    _config, _session, _counter = None, None, 0
    _request_labels.clear()
