"""Validate all 16 Qwen worker traces and export Decode JSON/CSV."""
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
    expected = {f"dp-{d}-tp-{t}" for d in range(8) for t in range(2)}
    if {p.name for p in root.iterdir() if p.is_dir()} != expected:
        raise IntegrityError("expected exactly sixteen worker directories")
    joined, identity, layers, seen, coverage = [], None, None, set(), {}
    for d in range(8):
        for t in range(2):
            name = f"dp-{d}-tp-{t}"
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
            if (w["deployment"] != "qwen36-single-node-debug" or w["worker"] != name
                    or (w["dp_rank"], w["tp_rank"], w["ep_rank"]) != (d, t, d*2+t)
                    or w["run"] != root.name):
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
                if (route.row.source_engine != f"Qwen-dp-{d}"
                        or executions.get(route.row.execution, {}).get(route.row.input_row)
                        != (route.row.local_request, route.row.position)):
                    raise IntegrityError("route differs from actual input declaration")
                joined.append(route)
    expected = {(f"Qwen-dp-{d}", execution, row, layer)
        for d, executions in coverage.items() for execution, rows in executions.items()
        for row in rows for layer in layers["target"]}
    if seen != expected:
        raise IntegrityError("missing or extra Decode token/layer coverage across TP workers")
    if not joined:
        raise IntegrityError("no Decode routes")
    return joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--counts", type=Path)
    args = parser.parse_args()
    try:
        for path in (args.output, args.counts):
            if path and (path.exists() or path.resolve().is_relative_to(args.run_dir.resolve())):
                raise IntegrityError("export must be a new file outside raw run")
        routes = collect(args.run_dir)
        if args.output:
            atomic_json(args.output, {"profile": "qwen36-single-node-debug", "npu_verified": False,
                "requests_terminal_verified": False, "validation_level": "a3_unverified_debug",
                "routes": [asdict(r) for r in routes]})
        counts = Counter((r.layer, expert) for r in routes for expert in r.expert_ids)
        if args.counts:
            with args.counts.open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["layer", "expert_id", "choices"])
                writer.writerows((layer, expert, n) for (layer, expert), n in sorted(counts.items()))
        print(json.dumps({"worker_traces_complete": True, "logical_routes": len(routes),
                          "expert_choices": sum(counts.values()), "npu_verified": False}))
    except (IntegrityError, ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"worker_traces_complete": False, "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
