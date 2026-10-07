"""Fixed-source CPU hook integration. Not an A3/PD service adapter yet.

The CPU harness supplies verified immutable row bindings. Hooks read the actual
selector and W8A8 consumer arguments, not a second Top-K computation.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import os

from ..errors import IntegrityError, ConfigError
from ..distributed.decode import Route

_active = ContextVar("ascend_v41_source", default=None)


@dataclass
class SourceExecution:
    observer: object
    bindings: tuple
    scope: str
    layers: dict  # layer_name -> (original layer, number of logical experts)
    routes: list = field(default_factory=list)
    consumer_inputs: list = field(default_factory=list)
    selected: dict = field(default_factory=dict)
    verified: bool = False
    input_size: int = 0


@contextmanager
def execution(observer, bindings, scope, layers):
    from ..distributed.replay import InputBinding
    if _active.get() is not None:
        raise IntegrityError("nested source execution")
    bindings = tuple(bindings)
    if not bindings or not layers or any(not isinstance(b, InputBinding) for b in bindings):
        raise IntegrityError("explicit input bindings and layers required")
    if len({b.tensor_row for b in bindings}) != len(bindings):
        raise IntegrityError("duplicate tensor row")
    state = SourceExecution(observer, bindings, scope, dict(layers))
    token = _active.set(state)
    try:
        yield state
        if not state.verified or set(state.selected) != set(state.layers):
            raise IntegrityError("forward or declared MoE layers not captured")
        if {r["layer_name"] for r in state.consumer_inputs} != set(state.layers):
            raise IntegrityError("quantized consumer layers not captured")
    finally:
        _active.reset(token)


def _state():
    state = _active.get()
    if state is None and os.environ.get("MOE_TRACE_V41_SOURCE_HOOKS") == "1":
        raise ConfigError("V4.1 hooks need an explicit CPU harness; A3 service integration is unavailable")
    return state


def _cpu(tensor):
    if tensor is None or tensor.device.type != "cpu":
        raise ConfigError("V4.1 source hooks currently verify CPU only; graph/NPU capture is unavailable")


def verify_forward(input_ids, positions, graph_mode=None):
    state = _state()
    if state is None:
        return
    if graph_mode is not None and getattr(graph_mode, "name", None) != "NONE":
        raise ConfigError("source integration currently requires eager mode")
    _cpu(input_ids)
    _cpu(positions)
    import torch
    if input_ids.dtype not in (torch.int32, torch.int64) or positions.dtype not in (torch.int32, torch.int64):
        raise IntegrityError("forward IDs and positions must be integers")
    if input_ids.ndim != 1 or positions.shape != input_ids.shape or state.verified:
        raise IntegrityError("unsupported or duplicate forward inputs")
    for binding in state.bindings:
        index = binding.tensor_row
        if type(index) is not int or index < 0 or index >= len(input_ids):
            raise IntegrityError("binding outside actual forward input")
        expected = -1 if binding.row.input_role == "mask" else binding.row.token_id
        if int(input_ids[index]) != expected or int(positions[index]) != binding.row.position:
            raise IntegrityError("binding differs from actual forward input")
    state.verified = True
    state.input_size = len(input_ids)


def logical_routes(layer, ids, weights, force_load_balance=False):
    state = _state()
    if state is None:
        return
    _cpu(ids)
    _cpu(weights)
    if not state.verified or force_load_balance:
        raise ConfigError("requires verified inputs and real routing, not profiling")
    name = layer.layer_name
    if name not in state.layers or name in state.selected:
        raise IntegrityError("undeclared or repeated layer")
    if getattr(layer, "_use_v2_model_runner", False):
        raise ConfigError("V2 runner unsupported")
    # prepare/SP may change row order. CPU integration is identity-prepare only.
    if getattr(getattr(layer, "moe_config", None), "is_sequence_parallel", False):
        raise ConfigError("SP input binding is not implemented")
    layer_id, num_experts = state.layers[name]
    if ids.ndim != 2 or weights.shape != ids.shape or len(ids) != state.input_size:
        raise IntegrityError("route tensors must be aligned matrices")
    import torch
    if ids.dtype not in (torch.int32, torch.int64):
        raise IntegrityError("expert IDs must be integers")
    captured = []
    for binding in state.bindings:
        if binding.tensor_row >= len(ids):
            raise IntegrityError("binding outside selector input")
        route = Route(state.observer, binding.owner, binding.row, state.scope,
                      layer_id, num_experts, tuple(ids[binding.tensor_row].tolist()),
                      tuple(weights[binding.tensor_row].float().tolist()))
        captured.append(route)
    state.routes.extend(captured)
    state.selected[name] = len(ids)


def consumer_routes(layer, ids, weights):
    state = _state()
    if state is None:
        return
    _cpu(ids)
    _cpu(weights)
    name = layer.layer_name
    if (name not in state.selected or ids.ndim != 2 or weights.shape != ids.shape
            or len(ids) != state.selected[name]
            or any(v["layer_name"] == name for v in state.consumer_inputs)):
        raise IntegrityError("consumer does not match selected layer")
    state.consumer_inputs.append({"layer_name": name,
                                  "ids": ids.detach().clone().tolist(),
                                  "weights": weights.detach().float().clone().tolist()})
