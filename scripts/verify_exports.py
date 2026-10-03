"""Cross-check JSON details and CSV expert counts against stored trace arrays."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from moe_trace.reader import TraceReader


def compare(run_dir):
    root = Path(run_dir)
    reader = TraceReader(root)
    issues = reader.validate()
    rows = list(reader.detail_rows())
    exported = json.loads((root / "details.json").read_text())
    if exported["records"] != rows or exported["storage_complete"] != (not issues):
        issues.append("JSON detail records or completeness differ from stored trace")
    expected = Counter((r["phase"], r["layer_id"], int(expert))
                       for r in rows for expert in r["expert_ids"])
    with (root / "expert_counts.csv").open(newline="") as stream:
        counts = list(csv.DictReader(stream))
    actual = {(r["phase"], int(r["layer_id"]), int(r["expert_id"])): int(r["count"])
              for r in counts}
    keys = {(phase, layer, expert)
            for phase in reader.metadata["config"]["phases"]
            for layer in reader.metadata["moe_layers"]
            for expert in range(reader.metadata["experts_per_layer"][str(layer)])}
    if len(actual) != len(counts) or set(actual) != keys:
        issues.append("CSV count keys are duplicate, missing, or extra")
    if any(actual.get(key) != expected[key] for key in keys):
        issues.append("CSV expert counts differ from stored trace")
    return dict(passed=not issues, issues=issues, detail_records=len(rows),
                count_rows=len(counts), expert_choices=sum(actual.values()),
                zero_count_rows=sum(value == 0 for value in actual.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    result = compare(args.run_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
