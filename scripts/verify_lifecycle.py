"""Validate the actual lifecycle workload against scheduler events and routes."""
import argparse
import json
import re
from pathlib import Path

from moe_trace.reader import TraceReader


def compare(run_dir, client_file):
    reader = TraceReader(run_dir)
    issues = reader.validate()
    events = list(reader.jsonl("requests.jsonl"))
    steps = list(reader.jsonl("steps.jsonl"))
    cases = json.loads(Path(client_file).read_text())
    results = []
    terminals = {e["request_id"]: e for e in events if e["event"] != "started"}
    expected = {"stop": ("finished", "stop", "FINISHED_STOPPED"),
                "length": ("finished", "length", "FINISHED_LENGTH_CAPPED"),
                "cancel": ("cancelled", "abort", "FINISHED_ABORTED"),
                "after_cancel": ("finished", "length", "FINISHED_LENGTH_CAPPED")}
    for case in cases:
        # One completion per call. 0.29.0 additionally appends an eight-hex
        # engine suffix (input_processor.assign_request_id). Require uniqueness.
        base = case["api_id"] + "-0"
        version = reader.metadata.get("framework", {}).get("vllm_version")
        candidates = [rid for rid in terminals if (
            re.fullmatch(re.escape(base) + r"-[0-9a-f]{8}", rid)
            if version == "0.29.0" else rid == base)]
        if len(candidates) != 1:
            issues.append(f"ambiguous or absent API/internal mapping: {base}: {candidates}")
            continue
        request = candidates[0]
        terminal = terminals.get(request)
        if terminal is None:
            issues.append(f"no terminal matching this test's API ID: {request}")
            continue
        actual = (terminal["event"], terminal["reason"],
                  terminal["details"]["scheduler_status"])
        if actual != expected[case["case"]]:
            issues.append(f"unexpected terminal: {case['case']}: {actual}")
        executions = [step for step in steps
                      if any(row["request_id"] == request for row in step["rows"])]
        input_rows = [row for step in executions for row in step["rows"]
                      if row["request_id"] == request]
        if not input_rows:
            issues.append(f"executed workload has no saved rows: {request}")
        results.append(dict(case=case["case"], api_id=case["api_id"], request_id=request,
                            event=terminal["event"], reason=terminal["reason"],
                            details=terminal["details"],
                            input_rows=len(input_rows), route_records=len(input_rows) *
                            len(reader.metadata["moe_layers"]),
                            last_execution=executions[-1]["execution_id"] if executions else None))
    cancelled = next((r for r in results if r["case"] == "cancel"), None)
    after = next((r for r in results if r["case"] == "after_cancel"), None)
    if cancelled and after:
        after_first = next(i for i, step in enumerate(steps)
                           if any(row["request_id"] == after["request_id"] for row in step["rows"]))
        if any(row["request_id"] == cancelled["request_id"]
               for step in steps[after_first:] for row in step["rows"]):
            issues.append("cancelled request still executing during later request")
    else:
        issues.append("cancel/after_cancel workload evidence missing")
    if reader.status.get("all_requests_completed_normally") is not False:
        issues.append("run incorrectly says all requests completed normally")
    return dict(run_id=reader.metadata["run_id"], passed=not issues, issues=issues,
                request_count=reader.status["request_count"],
                terminal_request_count=reader.status["terminal_request_count"],
                record_count=reader.status["record_count"], cases=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("client_file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.run_dir, args.client_file)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
