import importlib.util
import json
from pathlib import Path

import numpy as np

from moe_trace.config import TraceConfig
from moe_trace.session import InputRow, TraceSession

spec = importlib.util.spec_from_file_location(
    "verify_reference", Path(__file__).parents[1] / "scripts/verify_reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reference_detects_request_and_expert_corruption(tmp_path):
    config = TraceConfig(enabled=True, output_root=tmp_path, save_token_ids=True)
    session = TraceSession(config, model={}, moe_layers=[0],
                           experts_per_layer={0: 4}, top_k=2)
    session.record_execution(step_id=1, execution_id="e1",
                             rows=[InputRow("a", 0, "prefill", 7),
                                   InputRow("b", 3, "decode", 8)],
                             routes={0: (np.array([[0, 1], [2, 3]]), None)})
    session.request_event("a", "finished", reason="stop")
    session.request_event("b", "finished", reason="length")
    session.close()
    events = [dict(run_id=session.run_id, execution_id="e1", kind="inputs", rows=[
        dict(row_id=0, request_id="a", token_position=0, phase="prefill", token_id=7),
        dict(row_id=1, request_id="b", token_position=3, phase="decode", token_id=8)]),
        dict(run_id=session.run_id, execution_id="e1", kind="consumer",
             layer_name="model.layers.0.experts", consumer="fused_experts",
             expert_ids=[[0, 1], [2, 3]])]
    reference = tmp_path / "reference.jsonl"

    def save():
        reference.write_text("\n".join(json.dumps(event) for event in events))

    save()
    assert module.compare(session.run_dir, reference)["passed"]
    events[0]["rows"][0]["request_id"] = "b"
    events[1]["expert_ids"][0] = [1, 0]
    save()
    result = module.compare(session.run_dir, reference)
    assert not result["passed"]
    assert any("input mismatch" in issue for issue in result["issues"])
    assert any("consumer route mismatch" in issue for issue in result["issues"])
