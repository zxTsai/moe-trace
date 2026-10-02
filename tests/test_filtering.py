import numpy as np

from moe_trace.config import TraceConfig
from moe_trace.reader import TraceReader
from moe_trace.session import InputRow, TraceSession


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
