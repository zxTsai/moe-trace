"""Validate all sixteen DeepSeek V4 D workers and export expert load."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import sys

from ..errors import IntegrityError
from ..io import atomic_json
from .storage import collect_run


def collect(root):
    root = Path(root)
    expected = {f"dp-{d}-tp-{t}" for d in range(16) for t in range(1)}
    if {p.name for p in root.iterdir() if p.is_dir()} != expected:
        raise IntegrityError("expected exactly sixteen worker directories")
    deployment = json.loads((root / "deployment.json").read_text())
    if (deployment.get("profile") != "deepseek-v4-dp16-tp1-debug"
            or set(deployment.get("expected_workers", [])) != expected
            or len(deployment["expected_workers"]) != 16
            or deployment.get("run_id") != root.name):
        raise IntegrityError("invalid V4 deployment manifest")
    joined, identity, layers, seen, coverage = [], None, None, set(), {}
    for d in range(16):
        for t in range(1):
            name = f"dp-{d}-tp-{t}"
            if not (root / name / "inputs-index.json").is_file():
                raise IntegrityError(f"worker has not closed input coverage: {name}")
            data = (root / name / "inputs.jsonl").read_bytes()
            index = json.loads((root / name / "inputs-index.json").read_text())
            if hashlib.sha256(data).hexdigest() != index["sha256"]:
                raise IntegrityError("input coverage hash mismatch")
            declarations = [json.loads(line) for line in data.splitlines()]
            if len(declarations) != index["executions"]:
                raise IntegrityError("input execution count mismatch")
            executions = {}
            for declaration in declarations:
                execution = declaration["execution"]
                if execution in executions:
                    raise IntegrityError("duplicate input execution")
                rows = declaration["rows"]
                positions = {r["input_row"]: (r["request"], r["position"]) for r in rows}
                if len(positions) != len(rows):
                    raise IntegrityError("duplicate input row")
                executions[execution] = positions
            if t and coverage[d] != executions:
                raise IntegrityError("TP input declarations disagree")
            coverage[d] = executions
            manifest, records = collect_run(root / name / "trace", allow_empty=True)
            if manifest["validation_level"] != "a3_unverified_debug" or len(manifest["workers"]) != 1:
                raise IntegrityError("invalid worker profile")
            w = manifest["workers"][0]
            if (w["deployment"] != "deepseek-v4-dp16-tp1-debug" or w["worker"] != name
                    or (w["dp_rank"], w["tp_rank"], w["ep_rank"]) != (d, t, d)
                    or w["run"] != root.name or w["node"] != deployment["node_id"]
                    or w["device"] != d):
                raise IntegrityError("worker identity mismatch")
            current = (w["deployment"], w["run"], w["node"])
            if identity is not None and (identity != current or layers != manifest["model_layers"]):
                raise IntegrityError("mixed runs or layer declarations")
            identity, layers = current, manifest["model_layers"]
            for route, accepted in records:
                key = (route.row.source_engine, route.row.execution, route.row.input_row, route.layer)
                if key in seen or route.row.kind != "target_decode" or accepted is not None:
                    raise IntegrityError("duplicate or non-target-decode route")
                seen.add(key)
                if (route.row.source_engine != f"D-v4-dp-{d}"
                        or executions.get(route.row.execution, {}).get(route.row.input_row)
                        != (route.row.local_request, route.row.position)):
                    raise IntegrityError("route differs from actual input declaration")
                joined.append(route)
    expected = {(f"D-v4-dp-{d}", execution, row, layer)
        for d, executions in coverage.items() for execution, rows in executions.items()
        for row in rows for layer in layers["target"]}
    if seen != expected:
        raise IntegrityError("missing or extra Decode token/layer coverage across TP workers")
    if not joined:
        raise IntegrityError("no Decode routes")
    return joined


def analyze(routes):
    """Count logical choices; this is not physical-device load or utilization."""
    sizes = {}
    counts = Counter((r.layer, expert) for r in routes for expert in r.expert_ids)
    for r in routes:
        if r.layer in sizes and sizes[r.layer] != r.num_experts:
            raise IntegrityError("inconsistent expert count for layer")
        sizes[r.layer] = r.num_experts
    summary = []
    for layer, n in sorted(sizes.items()):
        values = [counts[layer, e] for e in range(n)]
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        summary.append({"layer": layer, "num_experts": n, "expert_choices": sum(values),
                        "unused_experts": values.count(0), "max_over_mean": max(values) / mean,
                        "coefficient_of_variation": variance ** 0.5 / mean})
    return counts, summary, sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--by-execution", type=Path)
    args = parser.parse_args()
    try:
        paths = [p.resolve() for p in (args.output, args.counts, args.by_execution) if p]
        if len(set(paths)) != len(paths):
            raise IntegrityError("export paths must be distinct")
        for path in (args.output, args.counts, args.by_execution):
            if path and (path.exists() or path.resolve().is_relative_to(args.run_dir.resolve())):
                raise IntegrityError("export must be a new file outside raw run")
        routes = collect(args.run_dir)
        counts, summary, sizes = analyze(routes)
        if args.output:
            atomic_json(args.output, {"profile": "deepseek-v4-dp16-tp1-debug", "npu_verified": False,
                "requests_terminal_verified": False, "validation_level": "a3_unverified_debug",
                "load_summary": summary,
                "scope": "D Decode logical routed expert choices; excludes shared experts",
                "routes": [asdict(r) for r in routes]})
        if args.counts:
            with args.counts.open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["layer", "expert_id", "choices"])
                writer.writerows((layer, expert, counts[layer, expert])
                    for layer, n in sorted(sizes.items()) for expert in range(n))
        if args.by_execution:
            per_execution = Counter((r.row.source_engine, r.row.execution, r.layer, expert)
                                    for r in routes for expert in r.expert_ids)
            with args.by_execution.open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["source_engine", "execution", "layer", "expert_id", "choices"])
                writer.writerows((*key, n) for key, n in sorted(per_execution.items()))
        print(json.dumps({"worker_traces_complete": True, "logical_routes": len(routes),
                          "expert_choices": sum(counts.values()), "npu_verified": False}))
    except (IntegrityError, ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"worker_traces_complete": False, "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

