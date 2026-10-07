"""Experimental per-worker JSONL shards and strict offline merge.

Writer buffers are bounded. Offline merge currently materializes the run in RAM.
This schema is deliberately separate from the single-worker NPZ format.
"""
from __future__ import annotations

from dataclasses import asdict
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path

from ..errors import IntegrityError
from ..io import atomic_json
from .decode import Worker, DecodeRow, Route, CandidateResult, merge_routes, annotate_candidates

SCHEMA = "distributed-decode-experimental-1"


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"cannot read {path}: {exc}") from exc


def _manifest(root):
    manifest = _load(root / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("validation_level") != "cpu_contract_only":
        raise IntegrityError("unsupported distributed schema/validation level")
    workers = [Worker(**value) for value in manifest["workers"]]
    _check_manifest(workers, manifest["model_layers"])
    return manifest, workers


def _check_manifest(workers, layers):
    if not workers or len(set(workers)) != len(workers):
        raise IntegrityError("workers must be nonempty and unique")
    if len({(w.deployment, w.run) for w in workers}) != 1:
        raise IntegrityError("manifest workers must belong to one run")
    if len({(w.node, w.worker) for w in workers}) != len(workers):
        raise IntegrityError("worker names must be unique within node")
    if not isinstance(layers, dict) or not layers:
        raise IntegrityError("model layer declaration is required")
    for scope, values in layers.items():
        if (not isinstance(scope, str) or not scope.strip() or not isinstance(values, list)
                or not values or any(type(v) is not int or v < 0 for v in values)
                or len(set(values)) != len(values)):
            raise IntegrityError("invalid model layer declaration")


def create_run(root, workers, model_layers):
    root = Path(root)
    workers = list(workers)
    _check_manifest(workers, model_layers)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "manifest.json", {
        "schema": SCHEMA, "validation_level": "cpu_contract_only",
        "workers": [asdict(w) for w in workers], "model_layers": model_layers,
    })
    return root


def decode_route(value):
    value = dict(value)
    value["observer"] = Worker(**value["observer"])
    value["owner"] = Worker(**value["owner"])
    value["row"] = DecodeRow(**value["row"])
    value["expert_ids"] = tuple(value["expert_ids"])
    if value["weights"] is not None:
        value["weights"] = tuple(value["weights"])
    return Route(**value)


class WorkerWriter:
    """Single caller per worker. Does not resume/overwrite existing output."""

    def __init__(self, root, worker, shard_records=4096, buffer_bytes=4 * 1024 * 1024):
        if type(shard_records) is not int or shard_records < 1:
            raise IntegrityError("shard_records must be positive")
        if type(buffer_bytes) is not int or buffer_bytes < 1:
            raise IntegrityError("buffer_bytes must be positive")
        self.root = Path(root)
        self.manifest, workers = _manifest(self.root)
        if worker not in workers:
            raise IntegrityError("worker is not declared")
        self.worker = worker
        self.directory = self.root / f"worker-{workers.index(worker):05d}"
        self.directory.mkdir(exist_ok=False)
        self.limit, self.buffer, self.shards = shard_records, [], []
        self.byte_limit, self.buffer_size = buffer_bytes, 0
        self.closed = self.failed = False
        self._status("running")

    def _status(self, state):
        atomic_json(self.directory / "status.json", {
            "worker": asdict(self.worker), "state": state,
            "records": sum(s["records"] for s in self.shards), "shards": self.shards,
        })

    def append(self, event):
        if self.closed or self.failed:
            raise IntegrityError("writer is closed or failed")
        try:
            if isinstance(event, Route):
                if event.observer != self.worker:
                    raise IntegrityError("route observer differs from writer")
                value = {"type": "route", "value": asdict(event)}
            elif isinstance(event, CandidateResult):
                value = {"type": "candidate_result", "value": asdict(event)}
            else:
                raise IntegrityError("unsupported event type")
            self._append_value(value)
        except Exception:
            self.failed = True
            self._status("failed")
            raise

    def declare_execution(self, model_scope, rows, observers=None):
        """Owner declares effective inputs, so entire missing rows are detectable."""
        if self.closed or self.failed:
            raise IntegrityError("writer is closed or failed")
        try:
            rows = list(rows)
            observers = [self.worker] if observers is None else list(observers)
            known_workers = [Worker(**v) for v in self.manifest["workers"]]
            if (not observers or self.worker not in observers
                    or len(set(observers)) != len(observers)
                    or any(w not in known_workers for w in observers)):
                raise IntegrityError("invalid expected observers")
            if model_scope not in self.manifest["model_layers"] or not rows:
                raise IntegrityError("execution needs declared scope and nonempty rows")
            if (any(not isinstance(r, DecodeRow) for r in rows)
                    or len({(r.source_engine, r.execution) for r in rows}) != 1
                    or len({r.input_row for r in rows}) != len(rows)):
                raise IntegrityError("execution inputs have inconsistent or duplicate identities")
            self._append_value({"type": "execution", "value": {
                "owner": asdict(self.worker), "model_scope": model_scope,
                "rows": [asdict(r) for r in rows],
                "observers": [asdict(w) for w in observers],
            }})
        except Exception:
            self.failed = True
            self._status("failed")
            raise

    def _append_value(self, value):
        data = (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
        if len(data) > self.byte_limit:
            raise IntegrityError("event exceeds bounded buffer capacity")
        if self.buffer_size + len(data) > self.byte_limit:
            self._flush()
        self.buffer.append(data)
        self.buffer_size += len(data)
        if len(self.buffer) >= self.limit:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        name = f"events-{len(self.shards):06d}.jsonl"
        target = self.directory / name
        pending = self.directory / f".{name}.pending"
        data = b"".join(self.buffer)
        with pending.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Unique directory + single writer guarantee target cannot already exist.
        if target.exists():
            raise IntegrityError("shard would overwrite existing data")
        os.replace(pending, target)
        self.shards.append({"name": name, "records": len(self.buffer),
                            "sha256": hashlib.sha256(data).hexdigest()})
        self.buffer.clear()
        self.buffer_size = 0

    def close(self):
        if self.closed or self.failed:
            raise IntegrityError("writer is closed or failed")
        try:
            self._flush()
            self._status("complete")
            self.closed = True
        except Exception:
            self.failed = True
            self._status("failed")
            raise


def collect_run(root):
    """Reject incomplete/corrupt workers and missing layers or candidate results.

    Complete means the declared CPU event contract, not NPU execution correctness.
    Empty workers are allowed, but a run with zero routes cannot validate success.
    """
    root = Path(root)
    manifest, workers = _manifest(root)
    routes, results, declared = [], [], {}
    expected = {f"worker-{i:05d}" for i in range(len(workers))} | {"manifest.json"}
    if {p.name for p in root.iterdir()} != expected:
        raise IntegrityError("missing or unexpected run files/workers")
    for index, worker in enumerate(workers):
        directory = root / f"worker-{index:05d}"
        status = _load(directory / "status.json")
        if status["worker"] != asdict(worker) or status["state"] != "complete":
            raise IntegrityError("worker is incomplete or has wrong identity")
        total = 0
        names = {"status.json"}
        for shard_index, shard in enumerate(status["shards"]):
            name = f"events-{shard_index:06d}.jsonl"
            if shard["name"] != name:
                raise IntegrityError("invalid shard ordering/name")
            names.add(name)
            data = (directory / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != shard["sha256"]:
                raise IntegrityError("shard checksum mismatch")
            lines = data.decode("utf-8").splitlines()
            if len(lines) != shard["records"] or not lines:
                raise IntegrityError("shard count mismatch")
            total += len(lines)
            for line in lines:
                event = json.loads(line)
                if event["type"] == "execution":
                    value = event["value"]
                    if value["owner"] != asdict(worker):
                        raise IntegrityError("execution owner differs from writer")
                    scope = value["model_scope"]
                    if scope not in manifest["model_layers"] or not value["rows"]:
                        raise IntegrityError("invalid execution declaration")
                    row_values = [DecodeRow(**r) for r in value["rows"]]
                    observers = [Worker(**w) for w in value["observers"]]
                    if (not observers or worker not in observers
                            or len(set(observers)) != len(observers)
                            or any(w not in workers for w in observers)):
                        raise IntegrityError("invalid declared observers")
                    if len({(r.source_engine, r.execution) for r in row_values}) != 1:
                        raise IntegrityError("mixed execution declaration")
                    for row in row_values:
                        key = (worker.deployment, worker.run, row.source_engine,
                               row.execution, row.input_row, scope)
                        if key in declared:
                            raise IntegrityError("duplicate declared input")
                        declared[key] = (worker, row, set(observers))
                elif event["type"] == "route":
                    route = decode_route(event["value"])
                    if route.observer != worker or route.owner not in workers:
                        raise IntegrityError("undeclared owner or wrong observer")
                    routes.append(route)
                elif event["type"] == "candidate_result":
                    results.append(CandidateResult(**event["value"]))
                else:
                    raise IntegrityError("unknown event type")
        if total != status["records"] or {p.name for p in directory.iterdir()} != names:
            raise IntegrityError("worker record total or files mismatch")
    merged = merge_routes(routes)
    if not merged:
        raise IntegrityError("run has no routes")
    rows = {}
    specs = {}
    for route in merged:
        if route.model_scope not in manifest["model_layers"]:
            raise IntegrityError("undeclared model scope")
        spec = (route.num_experts, len(route.expert_ids))
        scope_layer = (route.model_scope, route.layer)
        if scope_layer in specs and specs[scope_layer] != spec:
            raise IntegrityError("inconsistent expert count or Top-K")
        specs[scope_layer] = spec
        key = route.key[:-1]
        baseline, layers = rows.setdefault(key, (route, set()))
        if baseline.row != route.row or baseline.owner != route.owner:
            raise IntegrityError("input identity differs across layers")
        layers.add(route.layer)
    for baseline, layers in rows.values():
        if layers != set(manifest["model_layers"][baseline.model_scope]):
            raise IntegrityError("missing or unexpected model layers")
    if set(rows) != set(declared):
        raise IntegrityError("declared inputs differ from captured inputs")
    observed = {}
    for route in routes:
        observed.setdefault(route.key[:-1], {}).setdefault(route.layer, set()).add(route.observer)
    for key, (owner, row, observers) in declared.items():
        if rows[key][0].owner != owner or rows[key][0].row != row:
            raise IntegrityError("route identity differs from declared input")
        if any(actual != observers for actual in observed[key].values()):
            raise IntegrityError("missing or unexpected replica observer")
    joined = annotate_candidates(merged, results)
    if any(r.row.kind == "target_verify" and r.row.candidate_node is not None
           and accepted is None for r, accepted in joined):
        raise IntegrityError("candidate acceptance result missing")
    return manifest, joined


def export_run(root, output):
    manifest, joined = collect_run(root)
    output = Path(output)
    # Do not invalidate worker integrity by writing analysis inside the input run.
    if output.resolve().is_relative_to(Path(root).resolve()):
        raise IntegrityError("analysis output must be outside the input run")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"schema": SCHEMA, "validation_level": manifest["validation_level"],
                         "routes": [{**asdict(r), "candidate_accepted": accepted}
                                    for r, accepted in joined]})
    return len(joined)


def export_counts(root, output):
    """Logical choice counts with candidate decisions; not device load."""
    _, joined = collect_run(root)
    output = Path(output)
    if output.resolve().is_relative_to(Path(root).resolve()):
        raise IntegrityError("analysis output must be outside the input run")
    counts = Counter()
    for route, accepted in joined:
        decision = "accepted" if accepted is True else "rejected" if accepted is False else "not_applicable"
        for expert in range(route.num_experts):
            counts.setdefault((route.row.kind, route.model_scope, route.layer, decision, expert), 0)
        for expert in route.expert_ids:
            counts[(route.row.kind, route.model_scope, route.layer, decision, expert)] += 1
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["execution_kind", "model_scope", "layer_id", "candidate_decision", "expert_id", "count"])
    for key, count in sorted(counts.items()):
        writer.writerow([*key, count])
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f".{output.name}.pending")
    with pending.open("w", encoding="utf-8", newline="") as target:
        target.write(stream.getvalue())
        target.flush()
        os.fsync(target.fileno())
    os.replace(pending, output)
    return sum(counts.values())
