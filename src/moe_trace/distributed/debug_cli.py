"""Validate and merge all sixteen independently written D-side debug traces."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from ..errors import IntegrityError
from ..io import atomic_json
from .storage import collect_run


def collect_deployment(root):
    root = Path(root)
    deployment = json.loads((root / "deployment.json").read_text())
    expected = [f"dp-{d}-tp-{t}" for d in range(8) for t in range(2)]
    if deployment["profile"] != "1P1D-debug" or deployment["expected_workers"] != expected:
        raise IntegrityError("unsupported deployment topology")
    actual = {p.name for p in root.iterdir() if p.is_dir()}
    if actual != set(expected):
        raise IntegrityError("missing or unexpected worker directories")
    joined, layers, seen = [], None, set()
    for name in expected:
        manifest, records = collect_run(root / name / "trace", allow_empty=True)
        workers = manifest["workers"]
        if len(workers) != 1 or manifest["validation_level"] != "a3_unverified_debug":
            raise IntegrityError("worker trace has wrong profile")
        worker = workers[0]
        if (worker["worker"] != name or worker["run"] != deployment["run_id"]
                or worker["node"] != deployment["node_id"]
                or name != f'dp-{worker["dp_rank"]}-tp-{worker["tp_rank"]}'
                or worker["ep_rank"] != worker["dp_rank"] * 2 + worker["tp_rank"]
                or worker["deployment"] != "ascend-v41-debug"):
            raise IntegrityError("worker identity mismatch")
        if layers is not None and manifest["model_layers"] != layers:
            raise IntegrityError("worker model layers differ")
        layers = manifest["model_layers"]
        for route, accepted in records:
            key = (route.row.source_engine, route.row.execution, route.row.input_row,
                   route.model_scope, route.layer)
            if key in seen or route.row.kind != "target_decode":
                raise IntegrityError("duplicate or non-decode logical route")
            seen.add(key)
            joined.append((route, accepted))
    if not joined:
        raise IntegrityError("deployment contains no Decode routes")
    return deployment, joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        deployment, joined = collect_deployment(args.run_dir)
        if args.output:
            output = args.output.resolve()
            if output.is_relative_to(args.run_dir.resolve()):
                raise IntegrityError("merged output must be outside the raw run")
            if output.exists():
                raise IntegrityError("merged output already exists")
            atomic_json(output, {
                "profile": "1P1D-debug", "run_id": deployment["run_id"],
                "validation_level": "a3_unverified_debug", "npu_verified": False,
                "requests_terminal_verified": False,
                "consumer_diagnostics_validated": False,
                "request_identity": "D-local namespaced; not correlated to proxy/P",
                "routes": [asdict(route) for route, _ in joined]})
        print(json.dumps({"worker_traces_complete": True, "logical_routes": len(joined),
                          "expert_choices": sum(len(r.expert_ids) for r, _ in joined),
                          "npu_verified": False}))
    except (IntegrityError, ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"worker_traces_complete": False, "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
