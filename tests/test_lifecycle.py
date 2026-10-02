import json

import numpy as np
import pytest

from moe_trace.config import TraceConfig
from moe_trace.errors import IntegrityError
from moe_trace.reader import TraceReader
from moe_trace.session import InputRow, TraceSession


def make_session(tmp_path):
    return TraceSession(TraceConfig(enabled=True, output_root=tmp_path), model={},
                        moe_layers=[0], experts_per_layer={0: 2}, top_k=1)


def test_missing_terminal_fails_close(tmp_path):
    session = make_session(tmp_path)
    session.request_event("a", "started")
    with pytest.raises(IntegrityError, match="lack terminal"):
        session.close()
    assert TraceReader(session.run_dir).status["state"] == "failed"


def test_cancelled_run_complete_but_request_not_normal(tmp_path):
    session = make_session(tmp_path)
    session.record_execution(step_id=1, execution_id="e1",
                             rows=[InputRow("a", 0, "prefill")],
                             routes={0: (np.array([[1]]), None)})
    session.request_event("a", "cancelled", reason="abort")
    session.close()
    reader = TraceReader(session.run_dir)
    assert reader.validate() == []
    assert reader.status["complete"]
    assert not reader.status["all_requests_completed_normally"]
    assert len(list(reader.detail_rows())) == 1
    events = list(reader.jsonl("requests.jsonl"))
    (session.run_dir / "requests.jsonl").write_text(json.dumps(events[0]) + "\n")
    assert any("lacks terminal" in issue for issue in reader.validate())


def test_rejects_execution_after_cancel(tmp_path):
    session = make_session(tmp_path)
    session.request_event("a", "started")
    session.request_event("a", "cancelled", reason="abort")
    with pytest.raises(IntegrityError, match="after request termination"):
        session.record_execution(step_id=1, execution_id="e1",
                                 rows=[InputRow("a", 0, "prefill")],
                                 routes={0: (np.array([[1]]), None)})


def test_cancel_before_any_execution_is_valid(tmp_path):
    session = make_session(tmp_path)
    session.request_event("waiting", "started")
    session.request_event("waiting", "cancelled", reason="abort")
    session.close()
    reader = TraceReader(session.run_dir)
    assert reader.validate() == []
    assert reader.status["record_count"] == 0


def test_disabled_hooks_remain_noops_after_config_cached(tmp_path, monkeypatch):
    from moe_trace.adapters import vllm_0_10_1 as adapter
    path = tmp_path / "disabled.yaml"
    path.write_text("trace:\n  enabled: false\n")
    monkeypatch.setenv("MOE_TRACE_CONFIG", str(path))
    adapter.reset_for_tests()
    try:
        for _ in range(3):
            adapter.request_started(None, None)
            adapter.request_finished(None, None)
            assert adapter._get_config() is None
    finally:
        adapter.reset_for_tests()
