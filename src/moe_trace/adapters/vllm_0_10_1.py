from __future__ import annotations

import contextvars
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from ..config import TraceConfig, load_config
from ..errors import ConfigError, IntegrityError
from ..session import InputRow, TraceSession

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_active: contextvars.ContextVar[_Execution | None] = contextvars.ContextVar(
    "moe_trace_execution", default=None)
_lock = threading.Lock()
_session: TraceSession | None = None
_config: TraceConfig | None = None
_counter = 0


@dataclass
class _Execution:
    session: TraceSession
    step_id: int
    execution_id: str
    rows: list[InputRow]
    routes: dict[int, list[tuple[np.ndarray, np.ndarray | None]]] = field(
        default_factory=dict)


def _get_config() -> TraceConfig | None:
    global _config
    if _config is not None:
        return _config if _config.enabled else None
    path = os.environ.get("MOE_TRACE_CONFIG")
    if not path:
        return None
    _config = load_config(path)
    return _config if _config.enabled else None


def _reject_unsupported(runner: Any, config: TraceConfig) -> None:
    v = config.vllm
    requirements = {
        "enforce_eager": True, "enable_prefix_caching": False,
        "speculative_model": None, "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1, "data_parallel_size": 1,
    }
    for key, expected in requirements.items():
        if key in v and v[key] != expected:
            raise ConfigError(f"v1 requires vllm.{key}={expected!r}")
    cfg = runner.vllm_config
    if not getattr(cfg.model_config, "enforce_eager", False):
        raise ConfigError("moe-trace v1 requires vLLM enforce_eager")
    if getattr(cfg.cache_config, "enable_prefix_caching", False):
        raise ConfigError("moe-trace v1 requires prefix caching disabled")
    if getattr(cfg, "speculative_config", None) is not None:
        raise ConfigError("moe-trace v1 does not support speculative decoding")
    parallel = cfg.parallel_config
    if getattr(parallel, "distributed_executor_backend", None) != "uni":
        raise ConfigError("v1 lifecycle collection currently requires the uni executor")
    for attr in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        if getattr(parallel, attr, 1) != 1:
            raise ConfigError(f"moe-trace v1 requires {attr}=1")
    hf = cfg.model_config.hf_config
    architectures = getattr(hf, "architectures", []) or []
    if "GraniteMoeForCausalLM" not in architectures:
        raise ConfigError("moe-trace v1 supports GraniteMoeForCausalLM only")
    if os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", ""):
        raise ConfigError("routing simulation is not a real route and is unsupported")


def _get_session(runner: Any, config: TraceConfig) -> TraceSession:
    global _session
    if _session is not None:
        return _session
    with _lock:
        if _session is None:
            _reject_unsupported(runner, config)
            hf = runner.vllm_config.model_config.hf_config
            layers = tuple(range(int(hf.num_hidden_layers)))
            experts = int(hf.num_local_experts)
            top_k = int(hf.num_experts_per_tok)
            import vllm
            _session = TraceSession(
                config,
                model={"name_or_path": str(runner.vllm_config.model_config.model),
                       "architecture": "GraniteMoeForCausalLM"},
                moe_layers=layers,
                experts_per_layer={layer: experts for layer in layers},
                top_k=top_k,
                framework={"vllm_version": vllm.__version__,
                           "adapter": "vllm_0_10_1"},
            )
    return _session


def request_started(scheduler: Any, request: Any) -> None:
    config = _get_config()
    if config is None:
        return
    label = getattr(request, "moe_trace_label", None)
    if config.request_labels is not None and label not in config.request_labels:
        return
    session = _get_session(scheduler, config)
    try:
        session.request_event(request.request_id, "started", client_label=label)
    except Exception as exc:
        session.fail(exc)
        raise


def request_finished(scheduler: Any, request: Any) -> None:
    config = _get_config()
    if config is None:
        return
    label = getattr(request, "moe_trace_label", None)
    if config.request_labels is not None and label not in config.request_labels:
        return
    session = _get_session(scheduler, config)
    reason = request.get_finished_reason()
    reason = str(reason) if reason is not None else None
    event = "cancelled" if request.status.name == "FINISHED_ABORTED" else "finished"
    try:
        session.request_event(request.request_id, event, client_label=label,
                              reason=reason, details={
                                  "scheduler_status": request.status.name,
                                  "stop_reason": request.stop_reason,
                                  "num_prompt_tokens": request.num_prompt_tokens,
                                  "num_output_tokens": request.num_output_tokens,
                              })
    except Exception as exc:
        session.fail(exc)
        raise


@contextmanager
def trace_execution(runner: Any, scheduler_output: Any,
                    num_scheduled_tokens: np.ndarray) -> Iterator[None]:
    """Scope one real model forward; a no-op when tracing is disabled."""
    global _counter
    config = _get_config()
    if config is None:
        yield
        return
    session = _get_session(runner, config)
    req_ids = runner.input_batch.req_ids
    if len(req_ids) != len(num_scheduled_tokens):
        error = IntegrityError("request and scheduled-token lengths differ")
        session.fail(error)
        raise error
    rows: list[InputRow] = []
    for req_index, (req_id, count) in enumerate(zip(req_ids, num_scheduled_tokens)):
        start = int(runner.input_batch.num_computed_tokens_cpu[req_index])
        prompt_len = len(runner.requests[req_id].prompt_token_ids)
        offset = sum(int(x) for x in num_scheduled_tokens[:req_index])
        label = getattr(runner.requests[req_id], "moe_trace_label", None)
        for local_row in range(int(count)):
            position = start + local_row
            token_id = int(runner.input_ids_cpu[offset + local_row])
            rows.append(InputRow(req_id, position,
                                 "prefill" if position < prompt_len else "decode",
                                 token_id, label))
    with _lock:
        _counter += 1
        step_id = _counter
    execution = _Execution(session, step_id, f"exec-{step_id:012d}", rows)
    reference_token = None
    if os.environ.get("MOE_TRACE_REFERENCE_DIR"):
        from . import verification
        verification.install(runner, os.environ["MOE_TRACE_REFERENCE_DIR"])
        reference_token = verification.begin(session.run_id, execution.execution_id)
    token = _active.set(execution)
    try:
        yield
        routes: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
        for layer, chunks in execution.routes.items():
            ids = np.concatenate([chunk[0] for chunk in chunks], axis=0)
            weights = None
            if config.save_expert_weights:
                weights = np.concatenate([chunk[1] for chunk in chunks if chunk[1] is not None], axis=0)
            routes[layer] = (ids, weights)
        session.record_execution(step_id=step_id, execution_id=execution.execution_id,
                                 rows=rows, routes=routes)
    except Exception as exc:
        session.fail(exc)
        raise
    finally:
        _active.reset(token)
        if reference_token is not None:
            verification.end(reference_token)


def record_routes(layer_name: str, topk_ids: Any, topk_weights: Any) -> None:
    """Capture already-selected routes immediately before expert execution."""
    execution = _active.get()
    if execution is None:
        return
    match = _LAYER_PATTERN.search(layer_name)
    if not match:
        error = IntegrityError(f"cannot derive original layer id from {layer_name!r}")
        execution.session.fail(error)
        raise error
    layer_id = int(match.group(1))
    ids = topk_ids.detach().cpu().numpy().copy()
    # Cast after the real tensor has been copied; the model tensor itself is untouched.
    ids = ids.astype(np.int32, copy=False)
    weights = None
    if execution.session.config.save_expert_weights:
        weights = topk_weights.detach().float().cpu().numpy().copy()
    execution.routes.setdefault(layer_id, []).append((ids, weights))


def close_trace() -> None:
    global _session
    if _session is not None:
        if not _session._failed:
            _session.close(require_terminal_requests=True)
        _session = None


def reset_for_tests() -> None:
    global _session, _config, _counter
    _session = None
    _config = None
    _counter = 0
