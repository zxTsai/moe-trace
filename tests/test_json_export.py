import json
from types import SimpleNamespace

from moe_trace.analysis import write_details_json


def test_json_export_overwrites_and_truncates_previous_result(tmp_path):
    output = tmp_path / "details.json"
    rows = [{"request_id": "first", "expert_ids": list(range(8))}] * 100
    reader = SimpleNamespace(metadata={"schema_version": "1.0.0", "run_id": "first-run"},
                             validate=lambda: [], detail_rows=lambda: iter(rows))
    write_details_json(reader, output)
    old_size = output.stat().st_size

    rows.clear()
    reader.metadata["run_id"] = "second-run"
    write_details_json(reader, output, step_id=2)
    result = json.loads(output.read_text())
    assert result["run_id"] == "second-run"
    assert result["records"] == []
    assert result["filters"] == {"step_id": 2}
    assert result["storage_complete"] is True
    assert output.stat().st_size < old_size
    assert len(list(tmp_path.iterdir())) == 1
