from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class TraceReader:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.metadata = self._json("metadata.json")
        self.status = self._json("status.json")

    def _json(self, name: str) -> dict[str, Any]:
        with (self.run_dir / name).open(encoding="utf-8") as stream:
            return json.load(stream)

    def jsonl(self, name: str) -> Iterator[dict[str, Any]]:
        path = self.run_dir / name
        if not path.exists():
            return
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{name}:{line_no}: {exc}") from exc

    def route_arrays(self) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        for shard in self.status.get("shards", []):
            name = shard["file"]
            with np.load(self.run_dir / name, allow_pickle=False) as data:
                yield name, {key: data[key] for key in data.files}

    def validate(self, *, require_complete_status: bool = True) -> list[str]:
        issues: list[str] = []
        if self.metadata.get("schema_version") != "1.0.0":
            issues.append("unsupported metadata schema_version")
        if self.status.get("run_id") != self.metadata.get("run_id"):
            issues.append("status and metadata run_id differ")
        if require_complete_status and not (self.status.get("state") == "complete"
                                            and self.status.get("complete") is True):
            issues.append("run has no successful complete marker")
        referenced = {item.get("file") for item in self.status.get("shards", [])}
        actual = {p.name for p in self.run_dir.glob("routes-*.npz")}
        for name in sorted(referenced - actual):
            issues.append(f"referenced shard is missing: {name}")
        for name in sorted(actual - referenced):
            issues.append(f"unreferenced committed shard: {name}")
        pending = sorted(p.name for p in self.run_dir.glob("*.pending"))
        if pending:
            issues.append(f"uncommitted files exist: {pending}")

        steps: dict[str, dict[int, dict[str, Any]]] = {}
        step_ids: set[tuple[str, int]] = set()
        try:
            for step in self.jsonl("steps.jsonl"):
                execution_id = step.get("execution_id")
                rows = step.get("rows", [])
                if execution_id in steps:
                    issues.append(f"duplicate execution_id in steps: {execution_id}")
                    continue
                if step.get("row_count") != len(rows):
                    issues.append(f"step row_count mismatch: {execution_id}")
                row_map = {row.get("row_id"): row for row in rows}
                if set(row_map) != set(range(len(rows))):
                    issues.append(f"non-contiguous row ids: {execution_id}")
                key = (str(step.get("engine_id")), step.get("step_id"))
                if key in step_ids:
                    issues.append(f"duplicate step id: {key}")
                step_ids.add(key)
                steps[str(execution_id)] = row_map
        except (OSError, ValueError) as exc:
            issues.append(f"cannot read steps: {exc}")

        expected_layers = set(self.metadata.get("moe_layers", []))
        top_k = self.metadata.get("top_k")
        expert_counts = {int(k): v for k, v in
                         self.metadata.get("experts_per_layer", {}).items()}
        seen: set[tuple[str, int, int]] = set()
        layers_by_row: dict[tuple[str, int], set[int]] = {}
        count = 0
        try:
            for name, arrays in self.route_arrays():
                required = {"execution_ids", "row_ids", "layer_ids", "expert_ids"}
                missing = required - set(arrays)
                if missing:
                    issues.append(f"{name} missing arrays: {sorted(missing)}")
                    continue
                n = len(arrays["row_ids"])
                if any(len(arrays[key]) != n for key in required):
                    issues.append(f"{name} leading dimensions differ")
                    continue
                if arrays["expert_ids"].shape != (n, top_k):
                    issues.append(f"{name} expert_ids shape is invalid")
                    continue
                if "expert_weights" in arrays and arrays["expert_weights"].shape != (n, top_k):
                    issues.append(f"{name} expert_weights shape is invalid")
                declared = next((s for s in self.status["shards"] if s["file"] == name), None)
                if declared and declared.get("records") != n:
                    issues.append(f"{name} manifest record count differs")
                for i in range(n):
                    execution_id = str(arrays["execution_ids"][i])
                    row_id = int(arrays["row_ids"][i])
                    layer_id = int(arrays["layer_ids"][i])
                    key = (execution_id, row_id, layer_id)
                    if key in seen:
                        issues.append(f"duplicate route key: {key}")
                    seen.add(key)
                    layers_by_row.setdefault((execution_id, row_id), set()).add(
                        layer_id)
                    if execution_id not in steps or row_id not in steps[execution_id]:
                        issues.append(f"route has no step row: {key}")
                    if layer_id not in expected_layers:
                        issues.append(f"route has undeclared layer: {key}")
                    ids = arrays["expert_ids"][i]
                    limit = expert_counts.get(layer_id, 0)
                    if ids.size and (ids.min() < 0 or ids.max() >= limit):
                        issues.append(f"expert id out of range: {key}")
                count += n
        except (OSError, ValueError) as exc:
            issues.append(f"cannot read routes: {exc}")
        if count != self.status.get("record_count", 0):
            issues.append("status record_count differs from readable shards")
        if self.metadata.get("request_lifecycle_required", False):
            lifecycle = {}
            try:
                for event in self.jsonl("requests.jsonl"):
                    request = event["request_id"]
                    kind = event.get("event")
                    previous = lifecycle.get(request)
                    if kind == "started":
                        if previous is not None:
                            issues.append(f"duplicate request start: {request}")
                        lifecycle[request] = "started"
                    elif kind in {"finished", "cancelled"}:
                        if previous != "started":
                            issues.append(f"invalid terminal event sequence: {request}")
                        if not event.get("reason"):
                            issues.append(f"terminal event lacks reason: {request}")
                        lifecycle[request] = kind
                    else:
                        issues.append(f"unknown lifecycle event: {kind}")
                for request, state in lifecycle.items():
                    if state == "started":
                        issues.append(f"request lacks terminal event: {request}")
                for row_map in steps.values():
                    for row in row_map.values():
                        if row["request_id"] not in lifecycle:
                            issues.append(f"route request lacks lifecycle: {row['request_id']}")
            except (OSError, ValueError, KeyError) as exc:
                issues.append(f"cannot read request lifecycle: {exc}")
        for execution_id, rows in steps.items():
            for row_id in rows:
                found = layers_by_row.get((execution_id, row_id), set())
                if found != expected_layers:
                    issues.append(f"missing/extra layers for {(execution_id, row_id)}")
        return issues

    @property
    def complete(self) -> bool:
        return not self.validate(require_complete_status=True)

    def detail_rows(self) -> Iterator[dict[str, Any]]:
        steps = {step["execution_id"]: step for step in self.jsonl("steps.jsonl")}
        for _, arrays in self.route_arrays():
            for i in range(len(arrays["row_ids"])):
                execution_id = str(arrays["execution_ids"][i])
                row_id = int(arrays["row_ids"][i])
                step = steps[execution_id]
                row = step["rows"][row_id]
                item = {
                    "run_id": self.metadata["run_id"],
                    "engine_id": step["engine_id"], "rank_id": step["rank_id"],
                    "step_id": step["step_id"], "execution_id": execution_id,
                    "request_id": row["request_id"], "phase": row["phase"],
                    "token_position": row["token_position"],
                    "layer_id": int(arrays["layer_ids"][i]),
                    "expert_ids": arrays["expert_ids"][i].tolist(),
                }
                if "token_id" in row:
                    item["token_id"] = row["token_id"]
                if "expert_weights" in arrays:
                    item["expert_weights"] = arrays["expert_weights"][i].tolist()
                yield item
