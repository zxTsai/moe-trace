"""Version-specific hooks for vLLM 0.29.0, synchronous uni Granite execution."""
from __future__ import annotations

import contextvars
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from ..config import load_config
from ..errors import ConfigError, IntegrityError
from ..session import InputRow, TraceSession

_config = None
_session = None
_counter = 0
_active = contextvars.ContextVar("moe_trace_029_execution", default=None)
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


def _reject_unsupported(owner, config):
    cfg = owner.vllm_config
    expected = dict(enforce_eager=True, enable_prefix_caching=False,
                    speculative_model=None, tensor_parallel_size=1,
                    pipeline_parallel_size=1, data_parallel_size=1,
                    async_scheduling=False)
    for key, value in expected.items():
        if key in config.vllm and config.vllm[key] != value:
            raise ConfigError(f"0.29.0 requires vllm.{key}={value!r}")
    if not cfg.model_config.enforce_eager:
        raise ConfigError("0.29.0 tracing requires enforce_eager")
    if cfg.cache_config.enable_prefix_caching:
        raise ConfigError("0.29.0 tracing requires prefix caching disabled")
    if cfg.speculative_config is not None:
        raise ConfigError("speculative decoding is unsupported")
    if cfg.scheduler_config.async_scheduling:
        raise ConfigError("0.29.0 tracing requires --no-async-scheduling")
    if getattr(cfg, "use_v2_model_runner", False):
        raise ConfigError("0.29.0 tracing requires VLLM_USE_V2_MODEL_RUNNER=0")
    if getattr(cfg, "shutdown_timeout", 0) <= 0:
        raise ConfigError("0.29.0 tracing requires --shutdown-timeout > 0 (recommend 30)")
    parallel = cfg.parallel_config
    if parallel.distributed_executor_backend != "uni":
        raise ConfigError("0.29.0 tracing requires the uni executor")
    for attr in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, attr, 1) != 1:
            raise ConfigError(f"0.29.0 tracing requires {attr}=1")
    if getattr(parallel, "enable_expert_parallel", False) or getattr(parallel, "use_ubatching", False):
        raise ConfigError("expert parallelism and microbatching are unsupported")
    if getattr(cfg, "kv_transfer_config", None) or getattr(cfg, "ec_transfer_config", None):
        raise ConfigError("disaggregated execution is unsupported")
    if "GraniteMoeForCausalLM" not in (cfg.model_config.hf_config.architectures or []):
        raise ConfigError("0.29.0 adapter supports GraniteMoeForCausalLM only")
    if os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", ""):
        raise ConfigError("routing simulation is unsupported")


def _get_session(owner, config):
    global _session
    if _session is None:
        _reject_unsupported(owner, config)
        import vllm
        from vllm._version import __commit_id__
        if vllm.__version__ != "0.29.0":
            raise ConfigError(f"0.29.0 adapter loaded with vLLM {vllm.__version__}")
        hf = owner.vllm_config.model_config.hf_config
        layers = tuple(range(int(hf.num_hidden_layers)))
        _session = TraceSession(
            config, model={"name_or_path": str(owner.vllm_config.model_config.model),
                           "architecture": "GraniteMoeForCausalLM"},
            moe_layers=layers, experts_per_layer={i: int(hf.num_local_experts) for i in layers},
            top_k=int(hf.num_experts_per_tok),
            framework={"vllm_version": vllm.__version__, "adapter": "vllm_0_29_0",
                       "upstream_commit_id": __commit_id__,
                       "integration": "0.29.0/0001-trace-hooks-v1",
                       "vllm_import_path": str(vllm.__file__),
                       "adapter_import_path": __file__})
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
        if config.request_labels is None or label in config.request_labels:
            session.request_event(request.request_id, "started", client_label=label)
    except Exception as exc:
        session.fail(exc)
        raise


def request_finished(owner, request):
    config = _get_config()
    if config is None:
        return
    label = getattr(request, "moe_trace_label", None)
    if config.request_labels is not None and label not in config.request_labels:
        return
    session = _get_session(owner, config)
    try:
        reason = request.get_finished_reason()
        session.request_event(request.request_id,
                              "cancelled" if request.status.name == "FINISHED_ABORTED" else "finished",
                              reason=str(reason) if reason is not None else None,
                              client_label=label, details={
                                  "scheduler_status": request.status.name,
                                  "stop_reason": request.stop_reason,
                                  "num_prompt_tokens": request.num_prompt_tokens,
                                  "num_output_tokens": request.num_output_tokens})
    except Exception as exc:
        session.fail(exc)
        raise


def build_rows(runner, counts, input_ids, positions):
    """Validate GPU forward arguments against synchronous batch state."""
    req_ids = runner.input_batch.req_ids
    if len(req_ids) != len(counts) or any(int(n) <= 0 for n in counts):
        raise IntegrityError("invalid scheduled request counts")
    total = sum(int(n) for n in counts)
    if input_ids is None or input_ids.ndim != 1 or positions.ndim != 1:
        raise IntegrityError("only token-ID inputs with one-dimensional positions are supported")
    if len(input_ids) != total or len(positions) != total:
        raise IntegrityError("forward inputs include padding or missing rows")
    ids = input_ids.detach().cpu().tolist()
    actual_positions = positions.detach().cpu().tolist()
    rows, offset = [], 0
    for index, (req_id, count) in enumerate(zip(req_ids, counts)):
        request = runner.requests[req_id]
        start = int(runner.input_batch.num_computed_tokens_cpu[index])
        for local in range(int(count)):
            pos = start + local
            row = offset + local
            if actual_positions[row] != pos or ids[row] != request.get_token_id(pos):
                raise IntegrityError(f"actual model input mismatch: {req_id}, position {pos}")
            rows.append(InputRow(req_id, pos,
                                 "prefill" if pos < request.num_prompt_tokens else "decode",
                                 ids[row], getattr(request, "moe_trace_label", None)))
        offset += int(count)
    return rows


@contextmanager
def trace_execution(runner, scheduler_output, counts, input_ids, positions):
    global _counter
    config = _get_config()
    if config is None:
        yield
        return
    session = _get_session(runner, config)
    token = reference_token = None
    try:
        if getattr(scheduler_output, "preempted_req_ids", None) or getattr(
                getattr(scheduler_output, "scheduled_cached_reqs", None), "resumed_req_ids", None):
            raise ConfigError("preemption/recomputation is not yet supported")
        rows = build_rows(runner, counts, input_ids, positions)
        _counter += 1
        execution = Execution(session, _counter, f"exec-{_counter:012d}", rows)
        if os.environ.get("MOE_TRACE_REFERENCE_DIR"):
            from ..validation import vllm_0_29_0 as verification
            verification.install(runner, os.environ["MOE_TRACE_REFERENCE_DIR"])
            reference_token = verification.begin(session.run_id, execution.identity)
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
        if reference_token is not None:
            verification.end(reference_token)


def check_route_backend(monolithic):
    if _active.get() is not None and monolithic:
        raise ConfigError("monolithic MoE kernels do not expose selected routes")


def record_routes(layer_name, topk_ids, topk_weights):
    execution = _active.get()
    if execution is None:
        return
    match = _layer_pattern.search(layer_name)
    if not match:
        raise IntegrityError(f"cannot identify original layer: {layer_name}")
    layer = int(match.group(1))
    if layer not in execution.session.moe_layers:
        return
    if layer in execution.routes:
        raise IntegrityError(f"duplicate layer execution: {layer}")
    ids = topk_ids.detach().cpu().numpy().astype(np.int32, copy=True)
    weights = (topk_weights.detach().float().cpu().numpy().copy()
               if execution.session.config.save_expert_weights else None)
    execution.routes[layer] = (ids, weights)


def close_trace():
    global _session
    if _session is not None:
        if not _session._failed:
            _session.close(require_terminal_requests=True)
        _session = None


def reset_for_tests():
    global _session, _config, _counter
    _session, _config, _counter = None, None, 0
