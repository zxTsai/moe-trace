import csv

import numpy as np
import pytest

from moe_trace.analysis import expert_counts, write_details_csv
from moe_trace.config import TraceConfig
from moe_trace.errors import IntegrityError
from moe_trace.reader import TraceReader
from moe_trace.session import InputRow, TraceSession


def config(tmp_path, **kwargs):
    values = dict(enabled=True, output_root=tmp_path, buffer_records=8,
                  shard_records=3, save_token_ids=True,
                  save_expert_weights=True)
    values.update(kwargs)
    return TraceConfig(**values)


def routes(rows):
    return {
        0: (np.asarray([[0, 1], [1, 2], [2, 3]][:rows], dtype=np.int64),
            np.asarray([[.6, .4], [.7, .3], [.8, .2]][:rows], dtype=np.float16)),
        2: (np.asarray([[3, 2], [2, 1], [1, 0]][:rows], dtype=np.int32),
            np.asarray([[.5, .5], [.6, .4], [.9, .1]][:rows], dtype=np.float32)),
    }


def test_round_trip_sharding_details_and_zero_counts(tmp_path):
    session = TraceSession(config(tmp_path), model={"name": "test"},
                           moe_layers=[0, 2], experts_per_layer={0: 4, 2: 4},
                           top_k=2, run_id="run")
    session.record_execution(
        step_id=1, execution_id="e1",
        rows=[InputRow("a", 0, "prefill", 11),
              InputRow("a", 1, "prefill", 12),
              InputRow("b", 8, "decode", 13)], routes=routes(3))
    session.request_event("a", "finished", reason="stop")
    session.request_event("b", "cancelled", reason="client_abort")
    session.close(require_terminal_requests=True)

    reader = TraceReader(tmp_path / "run")
    assert reader.validate() == []
    details = list(reader.detail_rows())
    assert len(details) == 6
    assert details[0]["request_id"] == "a"
    assert details[-1]["phase"] == "decode"
    assert len(reader.status["shards"]) == 2
    counts = expert_counts(reader)
    assert any(row["count"] == 0 for row in counts)
    assert sum(row["count"] for row in counts) == 12
    output = tmp_path / "details.csv"
    write_details_csv(reader, output, request_id="b")
    with output.open() as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_chunked_prefill_positions_are_preserved(tmp_path):
    cfg = config(tmp_path, save_token_ids=False, save_expert_weights=False,
                 shard_records=8)
    session = TraceSession(cfg, model={}, moe_layers=[0, 2],
                           experts_per_layer={0: 4, 2: 4}, top_k=2,
                           run_id="chunked")
    session.record_execution(step_id=1, execution_id="e1",
                             rows=[InputRow("a", 0, "prefill"),
                                   InputRow("a", 1, "prefill")], routes=routes(2))
    session.record_execution(step_id=2, execution_id="e2",
                             rows=[InputRow("a", 2, "prefill")], routes=routes(1))
    session.request_event("a", "finished", reason="length")
    session.close()
    rows = list(TraceReader(tmp_path / "chunked").detail_rows())
    assert sorted({(r["execution_id"], r["token_position"]) for r in rows}) == [
        ("e1", 0), ("e1", 1), ("e2", 2)]


@pytest.mark.parametrize("bad_routes", [
    {0: (np.zeros((1, 2), dtype=np.int32), None)},
    {0: (np.asarray([[0, 4]], dtype=np.int32), None),
     2: (np.zeros((1, 2), dtype=np.int32), None)},
])
def test_strict_failure_marks_run_failed(tmp_path, bad_routes):
    cfg = config(tmp_path, save_token_ids=False, save_expert_weights=False)
    session = TraceSession(cfg, model={}, moe_layers=[0, 2],
                           experts_per_layer={0: 4, 2: 4}, top_k=2)
    with pytest.raises(IntegrityError):
        session.record_execution(step_id=1, execution_id="e1",
                                 rows=[InputRow("a", 0, "prefill")],
                                 routes=bad_routes)
    reader = TraceReader(session.run_dir)
    assert reader.status["state"] == "failed"
    assert not reader.status["complete"]


def test_detects_missing_shard(tmp_path):
    cfg = config(tmp_path, save_token_ids=False, save_expert_weights=False,
                 shard_records=8)
    session = TraceSession(cfg, model={}, moe_layers=[0, 2],
                           experts_per_layer={0: 4, 2: 4}, top_k=2,
                           run_id="missing")
    session.record_execution(step_id=1, execution_id="e1",
                             rows=[InputRow("a", 0, "prefill")], routes=routes(1))
    session.request_event("a", "finished", reason="stop")
    session.close()
    shard = next((tmp_path / "missing").glob("routes-*.npz"))
    shard.unlink()
    assert any("missing" in issue for issue in TraceReader(tmp_path / "missing").validate())


def test_large_multishard_validation_is_linear(tmp_path, monkeypatch):
    cfg = config(tmp_path, save_token_ids=False, save_expert_weights=False,
                 buffer_records=256, shard_records=128)
    layers = list(range(24))
    session = TraceSession(cfg, model={}, moe_layers=layers,
                           experts_per_layer={layer: 32 for layer in layers},
                           top_k=8, run_id="large")
    row_count = 40
    original_flush = session._flush
    peaks = []

    def checked_flush(count=None):
        peaks.append(len(session._buffer))
        assert len(session._buffer) <= cfg.buffer_records
        return original_flush(count)

    monkeypatch.setattr(session, "_flush", checked_flush)
    ids = np.tile(np.arange(8, dtype=np.int32), (row_count, 1))
    session.record_execution(
        step_id=1, execution_id="e1",
        rows=[InputRow("a", i, "prefill") for i in range(row_count)],
        routes={layer: (ids, None) for layer in layers})
    session.request_event("a", "finished", reason="length")
    session.close()
    reader = TraceReader(tmp_path / "large")
    assert reader.validate() == []
    assert reader.status["record_count"] == row_count * len(layers)
    assert peaks and max(peaks) <= cfg.buffer_records
