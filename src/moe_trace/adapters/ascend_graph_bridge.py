"""Bridge ACLGraphWrapper capture/replay to execution-local route snapshots.

This module does not serialize routes or infer token ownership. The runner must
provide that association outside graph capture. Importing it enables nothing.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from .graph_routes import GraphRouteBank

_execution = ContextVar("ascend_graph_route_execution", default=None)
_bank = ContextVar("ascend_graph_route_bank", default=None)
_armed_layers = None
_before_snapshot = None


def arm_graph_capture(max_layers, before_snapshot=None):
    """Enable startup capture before the runner's first real request."""
    global _armed_layers, _before_snapshot
    if max_layers is not None and (type(max_layers) is not int or max_layers < 1):
        raise ValueError("max_layers must be positive or None")
    _armed_layers, _before_snapshot = max_layers, before_snapshot


@contextmanager
def execution_routes(max_layers):
    """Collect one forward's graphs without retaining mutable runner state."""
    if _execution.get() is not None:
        raise RuntimeError("nested route executions are unsupported")
    result = []
    token = _execution.set((max_layers, result))
    try:
        yield result
    finally:
        _execution.reset(token)


def record_graph_routes(layer, expert_ids, weights=None):
    bank = _bank.get()
    if bank is not None:
        bank.record(layer, expert_ids, weights)
        return True
    execution = _execution.get()
    if execution is not None:
        from .graph_routes import RouteSnapshot
        execution[1].append(RouteSnapshot(layer, expert_ids.clone(),
                                        None if weights is None else weights.clone()))
        return True
    return False


def trace_acl_graph(function):
    """Decorate the reference ACLGraphWrapper.__call__, keeping its return intact.

    The callback never copies to CPU inside capture. Snapshot copies are issued
    after the wrapper returns, on its calling stream. Consumer synchronization
    and host conversion remain the responsibility of the runner adapter.
    """
    @wraps(function)
    def traced(wrapper, *args, **kwargs):
        execution = _execution.get()
        if execution is None and _armed_layers is None:
            return function(wrapper, *args, **kwargs)
        # Import only during an explicitly enabled forward.
        from vllm.forward_context import get_forward_context
        context = get_forward_context()
        mode = context.cudagraph_runtime_mode
        if getattr(mode, "name", None) == "NONE" or mode != wrapper.runtime_mode:
            return function(wrapper, *args, **kwargs)
        descriptor = context.batch_descriptor
        entry = wrapper.concrete_aclgraph_entries.get(descriptor)
        capturing = entry is None or entry.aclgraph is None
        banks = getattr(wrapper, "_moe_trace_route_banks", None)
        if banks is None:
            banks = wrapper._moe_trace_route_banks = {}
        if capturing:
            bank = GraphRouteBank(execution[0] if execution is not None else _armed_layers)
            token = _bank.set(bank)
            try:
                output = function(wrapper, *args, **kwargs)
                bank.seal()
            finally:
                _bank.reset(token)
            banks[descriptor] = bank
        else:
            bank = banks.get(descriptor)
            if bank is None:
                raise RuntimeError("graph was captured without route hooks; restart with capture enabled")
            output = function(wrapper, *args, **kwargs)
        if execution is not None:
            if _before_snapshot is not None:
                _before_snapshot()
            execution[1].extend(bank.snapshot())
        return output
    return traced
