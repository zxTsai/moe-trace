"""Compare independently observed GPU inputs/consumer arguments with saved routes."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from moe_trace.reader import TraceReader


def compare(run_dir, reference_path):
    reader = TraceReader(run_dir)
    issues = reader.validate()
    reference_inputs = {}
    reference_routes = {}
    consumers = Counter()
    for line in Path(reference_path).read_text().splitlines():
        event = json.loads(line)
        if event["run_id"] != reader.metadata["run_id"]:
            continue
        execution = event["execution_id"]
        if event["kind"] == "inputs":
            if execution in reference_inputs:
                issues.append(f"duplicate reference inputs: {execution}")
            reference_inputs[execution] = event["rows"]
        else:
            match = re.search(r"layers\.(\d+)\.", event["layer_name"] or "")
            if not match:
                issues.append("reference consumer has no layer identity")
                continue
            layer = int(match.group(1))
            consumers[event["consumer"]] += 1
            for row_id, ids in enumerate(event["expert_ids"]):
                key = (execution, row_id, layer)
                if key in reference_routes:
                    issues.append(f"duplicate reference route: {key}")
                reference_routes[key] = ids
    steps = list(reader.jsonl("steps.jsonl"))
    if {s["execution_id"] for s in steps} != set(reference_inputs):
        issues.append("saved/reference execution sets differ")
    compared_rows = 0
    mixed = 0
    concurrent = 0
    prefill_spans = {}
    membership = []
    for step in steps:
        execution = step["execution_id"]
        expected_rows = reference_inputs.get(execution, [])
        if len(step["rows"]) != len(expected_rows):
            issues.append(f"row counts differ: {execution}")
        requests = sorted({row["request_id"] for row in expected_rows})
        membership.append(requests)
        concurrent += len(requests) > 1
        mixed += {row["phase"] for row in expected_rows} == {"prefill", "decode"}
        for saved, reference in zip(step["rows"], expected_rows):
            for field in ("request_id", "token_position", "phase", "token_id"):
                if saved.get(field) != reference[field]:
                    issues.append(f"input mismatch: {execution}/{saved['row_id']}/{field}")
            compared_rows += 1
            if reference["phase"] == "prefill":
                prefill_spans.setdefault(reference["request_id"], {}).setdefault(
                    execution, []).append(reference["token_position"])
    saved_keys = set()
    route_count = 0
    for _, arrays in reader.route_arrays():
        for i, row_id in enumerate(arrays["row_ids"]):
            key = (str(arrays["execution_ids"][i]), int(row_id), int(arrays["layer_ids"][i]))
            saved_keys.add(key)
            if arrays["expert_ids"][i].tolist() != reference_routes.get(key):
                issues.append(f"consumer route mismatch: {key}")
            route_count += 1
    if saved_keys != set(reference_routes):
        issues.append("saved/reference route key sets differ")
    spans = {request: [{"execution_id": execution, "first": min(positions),
                        "last": max(positions), "count": len(positions)}
                       for execution, positions in executions.items()]
             for request, executions in prefill_spans.items()}
    for request, executions in prefill_spans.items():
        positions = sorted(p for values in executions.values() for p in values)
        if positions != list(range(len(positions))):
            issues.append(f"prefill positions not contiguous: {request}")
    return dict(run_id=reader.metadata["run_id"], passed=not issues, issues=issues,
                compared_input_rows=compared_rows, compared_route_records=route_count,
                compared_expert_ids=route_count * reader.metadata["top_k"],
                consumer_calls=dict(consumers), concurrent_executions=concurrent,
                mixed_phase_executions=mixed, request_membership=membership,
                prefill_spans=spans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("reference")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.run_dir, args.reference)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"request_membership", "prefill_spans"}}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
