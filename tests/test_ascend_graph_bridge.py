import sys
from types import ModuleType, SimpleNamespace

import pytest

from moe_trace.adapters.ascend_graph_bridge import (
    arm_graph_capture, execution_routes, record_graph_routes, trace_acl_graph,
)
from test_graph_routes import Tensor


@pytest.fixture
def graph(monkeypatch):
    arm_graph_capture(None)
    context = SimpleNamespace(cudagraph_runtime_mode="FULL", batch_descriptor=2)
    package = ModuleType("vllm")
    module = ModuleType("vllm.forward_context")
    module.get_forward_context = lambda: context
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", module)

    class Wrapper:
        runtime_mode = "FULL"

        def __init__(self):
            self.concrete_aclgraph_entries = {}
            self.operations = []
            self.ids = Tensor([[1, 2]], self.operations)
            self.hook_calls = 0
            self.output = object()

        @trace_acl_graph
        def __call__(self):
            entry = self.concrete_aclgraph_entries.get(context.batch_descriptor)
            if entry is None:
                self.hook_calls += 1
                record_graph_routes("target.0", self.ids)
                self.concrete_aclgraph_entries[context.batch_descriptor] = SimpleNamespace(aclgraph=True)
            else:
                for operation in self.operations:
                    operation()
            return self.output

    yield Wrapper(), context
    arm_graph_capture(None)


def test_capture_replay_returns_original_output_and_fresh_execution_snapshots(graph):
    wrapper, _ = graph
    with execution_routes(1) as first:
        assert wrapper() is wrapper.output
    wrapper.ids.values[:] = [[3, 4]]
    with execution_routes(1) as second:
        assert wrapper() is wrapper.output
    assert wrapper.hook_calls == 1
    assert first[0].expert_ids.values.tolist() == [[1, 2]]
    assert second[0].expert_ids.values.tolist() == [[3, 4]]


def test_uninstrumented_graph_fails_instead_of_returning_stale_routes(graph):
    wrapper, _ = graph
    wrapper()  # Framework warmup before enabling tracing.
    with execution_routes(1):
        with pytest.raises(RuntimeError, match="restart"):
            wrapper()


def test_execution_state_is_reset_after_exception(graph):
    wrapper, _ = graph
    with pytest.raises(ValueError):
        with execution_routes(1):
            raise ValueError("forward failed")
    with execution_routes(1) as result:
        wrapper()
    assert len(result) == 1
    assert not record_graph_routes("outside", Tensor([[1]]))


def test_nested_execution_rejected(graph):
    with execution_routes(1):
        with pytest.raises(RuntimeError, match="nested"):
            with execution_routes(1):
                pass


def test_startup_warmup_retains_bank_without_writing_execution(graph):
    wrapper, _ = graph
    barriers = []
    arm_graph_capture(1, lambda: barriers.append("wait_all_streams"))
    wrapper()  # Startup capture: no execution/file emission.
    assert not barriers
    wrapper.ids.values[:] = [[7, 8]]
    with execution_routes(1) as snapshots:
        wrapper()
    assert barriers == ["wait_all_streams"]
    assert snapshots[0].expert_ids.values.tolist() == [[7, 8]]
    assert wrapper.hook_calls == 1


def test_graph_guard_covers_warmup_and_replay(graph):
    from contextlib import contextmanager
    wrapper,_=graph
    calls=[]
    @contextmanager
    def guard():
        calls.append("lock")
        try:
            yield
        finally:
            calls.append("unlock")
    arm_graph_capture(1,call_guard=guard)
    wrapper()
    with execution_routes(1):
        wrapper()
    assert calls==["lock","unlock","lock","unlock"]
