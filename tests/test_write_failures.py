"""Controlled storage failures; never fill or damage a user's disk."""
import numpy as np
import pytest

from moe_trace.config import TraceConfig
from moe_trace.errors import TraceWriteError
from moe_trace.reader import TraceReader
from moe_trace.session import InputRow, TraceSession


def session_at(tmp_path):
    return TraceSession(TraceConfig(enabled=True, output_root=tmp_path,
                                    buffer_records=2, shard_records=1),
                        model={}, moe_layers=[0], experts_per_layer={0: 2}, top_k=1)


def test_npz_write_failure_never_marks_complete(tmp_path, monkeypatch):
    session = session_at(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("injected shard write failure")
    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(TraceWriteError, match="injected shard"):
        session.record_execution(step_id=1, execution_id="e1",
                                 rows=[InputRow("r", 0, "prefill")],
                                 routes={0: (np.array([[1]]), None)})
    reader = TraceReader(session.run_dir)
    assert not reader.status["complete"]
    assert reader.status["state"] == "failed"
    assert not list(session.run_dir.glob("routes-*.npz"))
    assert reader.validate()


def test_unclosed_run_remains_incomplete(tmp_path):
    session = session_at(tmp_path)
    session.record_execution(step_id=1, execution_id="e1",
                             rows=[InputRow("r", 0, "prefill")],
                             routes={0: (np.array([[1]]), None)})
    reader = TraceReader(session.run_dir)
    assert not reader.status["complete"]
    assert reader.validate()
