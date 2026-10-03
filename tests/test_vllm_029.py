from types import SimpleNamespace as NS

import numpy as np
import pytest

from moe_trace.adapters import vllm_0_29_0 as adapter
from moe_trace.config import TraceConfig
from moe_trace.errors import ConfigError, IntegrityError


class Tensor:
    def __init__(self, values):
        self.value = np.asarray(values)
        self.ndim = self.value.ndim

    def __len__(self):
        return len(self.value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value.tolist()


def batch():
    def request(tokens, prompt):
        return NS(get_token_id=lambda pos: tokens[pos], num_prompt_tokens=prompt)
    return NS(input_batch=NS(req_ids=["decode", "prefill"],
                             num_computed_tokens_cpu=np.array([2, 0])),
              requests={"decode": request([1, 2, 3], 2),
                        "prefill": request([10, 11], 2)})


def test_mixed_rows_follow_forward_order():
    rows = adapter.build_rows(batch(), [1, 2], Tensor([3, 10, 11]), Tensor([2, 0, 1]))
    assert [(r.request_id, r.token_position, r.phase, r.token_id) for r in rows] == [
        ("decode", 2, "decode", 3), ("prefill", 0, "prefill", 10),
        ("prefill", 1, "prefill", 11)]


@pytest.mark.parametrize("ids,positions,counts", [
    ([3, 10, 11, 0], [2, 0, 1, 0], [1, 2]),
    ([3, 10, 11], [2, 1, 0], [1, 2]),
    ([4, 10, 11], [2, 0, 1], [1, 2]),
    ([3, 10, 11], [2, 0, 1], [3]),
])
def test_invalid_forward_mapping_rejected(ids, positions, counts):
    with pytest.raises(IntegrityError):
        adapter.build_rows(batch(), counts, Tensor(ids), Tensor(positions))


def test_async_scheduling_rejected():
    owner = NS(vllm_config=NS(model_config=NS(enforce_eager=True),
                             cache_config=NS(enable_prefix_caching=False),
                             speculative_config=None,
                             scheduler_config=NS(async_scheduling=True)))
    with pytest.raises(ConfigError, match="no-async-scheduling"):
        adapter._reject_unsupported(owner, TraceConfig(enabled=True))


@pytest.mark.parametrize("v2,timeout,message", [
    (True, 30, "V2_MODEL_RUNNER"), (False, 0, "shutdown-timeout"),
])
def test_unsafe_runner_and_shutdown_rejected(v2, timeout, message):
    owner = NS(vllm_config=NS(model_config=NS(enforce_eager=True),
                             cache_config=NS(enable_prefix_caching=False),
                             speculative_config=None, use_v2_model_runner=v2,
                             shutdown_timeout=timeout,
                             scheduler_config=NS(async_scheduling=False)))
    with pytest.raises(ConfigError, match=message):
        adapter._reject_unsupported(owner, TraceConfig(enabled=True))


def test_monolithic_rejected_only_during_real_trace():
    adapter.check_route_backend(True)
    token = adapter._active.set(object())
    try:
        with pytest.raises(ConfigError, match="monolithic"):
            adapter.check_route_backend(True)
        adapter.check_route_backend(False)
    finally:
        adapter._active.reset(token)


def test_disabled_hooks_do_not_access_vllm(tmp_path, monkeypatch):
    path = tmp_path / "disabled.yaml"
    path.write_text("trace:\n  enabled: false\n")
    monkeypatch.setenv("MOE_TRACE_CONFIG", str(path))
    adapter.reset_for_tests()
    try:
        for _ in range(2):
            adapter.request_started(None, None)
            adapter.request_finished(None, None)
            with adapter.trace_execution(None, None, None, None, None):
                adapter.record_routes(None, None, None)
                adapter.check_route_backend(True)
    finally:
        adapter.reset_for_tests()
