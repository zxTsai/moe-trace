"""Independent small CPU fixture for distributed Decode shards, not a vLLM run."""
from dataclasses import replace
import argparse
from pathlib import Path
import uuid

from ..distributed.decode import Worker, DecodeRow, Route, CandidateResult, expert_counts
from ..distributed.storage import create_run, WorkerWriter, collect_run, export_run
from ..errors import IntegrityError
from ..io import atomic_json


def run_validation(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = output / "cpu-validation.json"
    atomic_json(report, {"state": "running", "npu_verified": False})
    try:
        run_id = str(uuid.uuid4())
        workers = [Worker("fixture", run_id, "D0", f"worker{i}", i // 2, i % 2, i, i)
                   for i in range(4)]
        root = create_run(output / "runs" / run_id, workers, {"target": [20, 21], "draft": [0]})
        writers = {w: WorkerWriter(root, w, shard_records=3) for w in workers}
        # Explicit consumer input oracle: owner, request, position, token,
        # execution, kind, candidate, per-layer IDs. Never derived from merge keys.
        oracle = [
            (0, "A", 7, 41, "verify-1", "target_verify", "c0", {20: (3, 1), 21: (1, 2)}),
            (0, "A", 8, 42, "verify-1", "target_verify", "c1", {20: (2, 0), 21: (3, 1)}),
            (0, "A", 8, 43, "recompute-2", "target_decode", None, {20: (1, 0), 21: (2, 3)}),
            (3, "B", 9, 51, "decode-1", "target_decode", None, {20: (0, 2), 21: (1, 3)}),
            (0, "A", 8, None, "draft-1", "dspark_draft", None, {0: (2, 1)}),
            (0, "A", 7, 41, "draft-1", "dspark_draft", None, {0: (0, 3)}),
        ]
        rows_by_execution = {}
        observations = []
        reference = []
        for index, (owner_index, request, pos, token, execution, kind, candidate, layers) in enumerate(oracle):
            owner = workers[owner_index]
            peer = workers[owner_index ^ 1]
            scope = "draft" if kind == "dspark_draft" else "target"
            role = "mask" if token is None else "seed" if scope == "draft" else "sequence"
            row = DecodeRow(f"engine{owner.dp_rank}", execution, 4, request, "same-local-id",
                            index, pos, token, kind, "round0" if candidate else None,
                            candidate, role)
            key = (owner, scope, execution)
            rows_by_execution.setdefault(key, []).append(row)
            for layer, ids in layers.items():
                route = Route(owner, owner, row, scope, layer, 4, ids)
                observations.extend([route, replace(route, observer=peer)])
                reference.append({"request": request, "execution": execution,
                                  "position": pos, "token_id": token, "input_role": role,
                                  "model_scope": scope, "layer": layer, "expert_ids": list(ids)})
        for (owner, scope, _), rows in rows_by_execution.items():
            writers[owner].declare_execution(scope, rows, [owner, workers[workers.index(owner) ^ 1]])
        # Reverse both worker arrival and row order; do not rely on file appearance.
        for route in reversed(observations):
            writers[route.observer].append(route)
        writers[workers[0]].append(CandidateResult("A", "engine0", "round0", "c0", True))
        writers[workers[0]].append(CandidateResult("A", "engine0", "round0", "c1", False))
        for writer in reversed(list(writers.values())):
            writer.close()
        _, joined = collect_run(root)
        actual = [{"request": r.row.global_request, "execution": r.row.execution,
                   "position": r.row.position, "token_id": r.row.token_id,
                   "input_role": r.row.input_role, "model_scope": r.model_scope,
                   "layer": r.layer, "expert_ids": list(r.expert_ids)} for r, _ in joined]
        sort = lambda records: sorted(records, key=lambda v: (v["request"], v["execution"],
                                                             v["position"], v["layer"]))
        if sort(actual) != sort(reference):
            raise IntegrityError("independent consumer fixture differs from merged routes")
        counts = expert_counts(r for r, _ in joined)
        if len(joined) != 10 or sum(counts.values()) != 20:
            raise IntegrityError("unexpected route/choice count")
        decisions = {(r.row.candidate_node, accepted) for r, accepted in joined
                     if r.row.candidate_node is not None}
        if decisions != {("c0", True), ("c1", False)}:
            raise IntegrityError("acceptance association differs from fixture")
        export_run(root, output / "details.json")
        atomic_json(output / "consumer-reference.json", {"routes": reference})
        result = {"state": "passed", "validation_level": "cpu_contract_only",
                  "npu_verified": False, "run_dir": str(root), "workers": 4,
                  "observed_routes": len(observations), "logical_routes": len(joined),
                  "expert_choices": sum(counts.values()), "comparison": "passed"}
        atomic_json(report, result)
        return result
    except Exception as exc:
        atomic_json(report, {"state": "failed", "npu_verified": False, "error": str(exc)})
        raise


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
