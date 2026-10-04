import numpy as np
import pytest
from types import SimpleNamespace

from moe_trace.config import TraceConfig
from moe_trace.reader import TraceReader
from moe_trace.session import InputRow, TraceSession


@pytest.mark.parametrize("version", ["vllm_0_10_1", "vllm_0_29_0"])
def test_adapter_skips_unselected_layers_before_tensor_copy(version):
    import importlib
    adapter = importlib.import_module("moe_trace.adapters." + version)
    execution = SimpleNamespace(session=SimpleNamespace(moe_layers=(0, 1)), routes={})
    token = adapter._active.set(execution)
    try:
        # Objects deliberately have no tensor API; excluded layers must not copy.
        adapter.record_routes("model.layers.2.block_sparse_moe", object(), object())
        assert not execution.routes
    finally:
        adapter._active.reset(token)


def test_phase_and_label_filter_slice_route_rows(tmp_path):
    cfg = TraceConfig(enabled=True, output_root=tmp_path,
                      phases=("decode",), request_labels=("keep",),
                      shard_records=10, buffer_records=10)
    session = TraceSession(cfg, model={}, moe_layers=[0],
                           experts_per_layer={0: 3}, top_k=1, run_id="filtered")
    rows = [InputRow("a", 0, "prefill", client_label="keep"),
            InputRow("a", 1, "decode", client_label="keep"),
            InputRow("b", 1, "decode", client_label="drop")]
    session.record_execution(step_id=1, execution_id="e1", rows=rows,
                             routes={0: (np.asarray([[0], [1], [2]]), None)})
    session.request_event("a", "finished", reason="length")
    session.close()
    details = list(TraceReader(tmp_path / "filtered").detail_rows())
    assert [(r["request_id"], r["expert_ids"]) for r in details] == [("a", [1])]
