from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import TraceConfig
from .errors import IntegrityError, TraceWriteError
from .io import append_jsonl, atomic_json

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class InputRow:
    request_id: str
    token_position: int
    phase: str
    token_id: int | None = None
    client_label: str | None = None


@dataclass
class _RouteRecord:
    execution_id: str
    row_id: int
    layer_id: int
    expert_ids: np.ndarray
    expert_weights: np.ndarray | None


class TraceSession:
    """Strict, synchronous and bounded writer for one trace run."""

    def __init__(self, config: TraceConfig, *, model: dict[str, Any],
                 moe_layers: Iterable[int], experts_per_layer: dict[int, int],
                 top_k: int, framework: dict[str, Any] | None = None,
                 run_id: str | None = None, rank_id: int = 0,
                 expert_weight_semantics: str | None = None):
        if not config.enabled:
            raise ValueError("TraceSession requires trace.enabled=true")
        self.config = config
        self.run_id = run_id or str(uuid.uuid4())
        self.rank_id = rank_id
        self.moe_layers = tuple(sorted(moe_layers if config.layers is None else config.layers))
        self.experts_per_layer = {int(k): int(v) for k, v in experts_per_layer.items()}
        self.top_k = int(top_k)
        if rank_id != 0:
            raise IntegrityError("v1 supports rank_id=0 only")
        if not self.moe_layers or self.top_k <= 0:
            raise IntegrityError("MoE layers and top_k must be non-empty/positive")
        if set(self.moe_layers) - set(self.experts_per_layer):
            raise IntegrityError("missing expert count for a configured MoE layer")
        self.run_dir = config.output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._buffer: list[_RouteRecord] = []
        self._steps: dict[str, list[InputRow]] = {}
        self._seen_keys: set[tuple[str, int, int]] = set()
        self._seen_requests: set[str] = set()
        self._request_terminal: set[str] = set()
        self._cancelled_requests: set[str] = set()
        self._shards: list[dict[str, Any]] = []
        self._record_count = 0
        self._step_count = 0
        self._closed = False
        self._failed = False
        self._status = {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id,
            "state": "running", "complete": False, "record_count": 0,
            "step_count": 0, "shards": [], "errors": [],
            "started_at": _now(),
        }
        metadata = {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id,
            "engine_id": config.engine_id, "rank_id": rank_id,
            "model": model, "framework": framework or {},
            "moe_layers": list(self.moe_layers),
            "experts_per_layer": {str(k): v for k, v in self.experts_per_layer.items()},
            "top_k": self.top_k, "config": config.to_dict(),
            "request_lifecycle_required": True,
            "expert_weight_semantics": (
                expert_weight_semantics
                or "FusedMoE select_experts output consumed by expert kernel; normalized"
                if config.save_expert_weights else None),
        }
        atomic_json(self.run_dir / "metadata.json", metadata)
        atomic_json(self.run_dir / "status.json", self._status)

    def request_event(self, request_id: str, event: str, *,
                      client_label: str | None = None,
                      reason: str | None = None,
                      details: dict[str, Any] | None = None) -> None:
        self._ensure_writable()
        if event not in {"started", "finished", "cancelled"}:
            raise IntegrityError(f"unknown request event {event!r}")
        if event == "started":
            if request_id in self._seen_requests:
                raise IntegrityError(f"duplicate request start: {request_id}")
            self._seen_requests.add(request_id)
        else:
            if request_id not in self._seen_requests:
                raise IntegrityError(f"terminal event without start: {request_id}")
            if request_id in self._request_terminal:
                raise IntegrityError(f"duplicate terminal event: {request_id}")
            if not reason:
                raise IntegrityError(f"terminal event lacks reason: {request_id}")
            self._request_terminal.add(request_id)
            if event == "cancelled":
                self._cancelled_requests.add(request_id)
        append_jsonl(self.run_dir / "requests.jsonl", {
            "run_id": self.run_id, "engine_id": self.config.engine_id,
            "request_id": request_id, "event": event,
            "client_label": client_label, "reason": reason, "at": _now(),
            "details": details or {},
        })
        self._write_status()

    def record_execution(self, *, step_id: int, execution_id: str,
                         rows: Iterable[InputRow],
                         routes: dict[int, tuple[np.ndarray, np.ndarray | None]]) -> None:
        self._ensure_writable()
        try:
            rows_list = list(rows)
            if not rows_list:
                raise IntegrityError("execution has no input rows")
            if execution_id in self._steps:
                raise IntegrityError(f"duplicate execution_id {execution_id}")
            selected = [
                (i, row) for i, row in enumerate(rows_list)
                if row.phase in self.config.phases
                and (self.config.request_labels is None
                     or row.client_label in self.config.request_labels)
            ]
            layer_set = set(routes)
            expected = set(self.moe_layers)
            if layer_set != expected:
                raise IntegrityError(
                    f"execution {execution_id} layer mismatch: expected {sorted(expected)}, "
                    f"got {sorted(layer_set)}")
            step_rows: list[dict[str, Any]] = []
            retained_rows: list[InputRow] = []
            for new_row_id, (source_row_id, row) in enumerate(selected):
                if row.request_id in self._request_terminal:
                    raise IntegrityError(f"execution after request termination: {row.request_id}")
                if row.phase not in {"prefill", "decode"} or row.token_position < 0:
                    raise IntegrityError("invalid phase or token position")
                item = {"row_id": new_row_id, "source_row_id": source_row_id,
                        "request_id": row.request_id,
                        "token_position": row.token_position, "phase": row.phase}
                if self.config.save_token_ids:
                    if row.token_id is None:
                        raise IntegrityError("token_id enabled but a row has no token_id")
                    item["token_id"] = row.token_id
                step_rows.append(item)
                retained_rows.append(row)
                if row.request_id not in self._seen_requests:
                    self.request_event(row.request_id, "started", client_label=row.client_label)
            for layer_id in self.moe_layers:
                ids, weights = routes[layer_id]
                ids = np.asarray(ids)
                if ids.shape != (len(rows_list), self.top_k):
                    raise IntegrityError(f"layer {layer_id} expert_ids shape {ids.shape} is invalid")
                if not np.issubdtype(ids.dtype, np.integer):
                    raise IntegrityError("expert_ids must have integer dtype")
                ids = ids[[source for source, _ in selected]].astype(np.int32, copy=True)
                if ids.size and (ids.min() < 0 or ids.max() >= self.experts_per_layer[layer_id]):
                    raise IntegrityError(f"layer {layer_id} expert id out of range")
                saved_weights = None
                if self.config.save_expert_weights:
                    if weights is None:
                        raise IntegrityError("expert weights enabled but missing")
                    weights = np.asarray(weights)
                    if weights.shape != (len(rows_list), self.top_k):
                        raise IntegrityError(f"layer {layer_id} weight shape is invalid")
                    saved_weights = weights[[source for source, _ in selected]].astype(
                        np.float32, copy=True)
                for row_id in range(len(selected)):
                    key = (execution_id, row_id, layer_id)
                    if key in self._seen_keys:
                        raise IntegrityError(f"duplicate route key {key}")
                    self._seen_keys.add(key)
                    self._buffer.append(_RouteRecord(execution_id, row_id, layer_id,
                                                     ids[row_id], None if saved_weights is None
                                                     else saved_weights[row_id]))
                    if len(self._buffer) >= self.config.shard_records:
                        self._flush(self.config.shard_records)
            self._steps[execution_id] = retained_rows
            append_jsonl(self.run_dir / "steps.jsonl", {
                "run_id": self.run_id, "engine_id": self.config.engine_id,
                "rank_id": self.rank_id, "step_id": step_id,
                "execution_id": execution_id, "row_count": len(selected),
                "rows": step_rows,
            })
            self._step_count += 1
            while len(self._buffer) >= self.config.shard_records:
                self._flush(self.config.shard_records)
            if len(self._buffer) > self.config.buffer_records:
                raise IntegrityError("route buffer exceeded configured hard limit")
            self._write_status()
        except Exception as exc:
            self.fail(exc)
            raise

    def _flush(self, count: int | None = None) -> None:
        if not self._buffer:
            return
        count = len(self._buffer) if count is None else min(count, len(self._buffer))
        records = self._buffer[:count]
        shard_name = f"routes-{len(self._shards) + 1:06d}.npz"
        final = self.run_dir / shard_name
        pending = self.run_dir / f".{shard_name}.pending"
        arrays: dict[str, np.ndarray] = {
            "execution_ids": np.asarray([r.execution_id for r in records]),
            "row_ids": np.asarray([r.row_id for r in records], dtype=np.int32),
            "layer_ids": np.asarray([r.layer_id for r in records], dtype=np.int32),
            "expert_ids": np.stack([r.expert_ids for r in records]).astype(np.int32, copy=False),
        }
        if self.config.save_expert_weights:
            arrays["expert_weights"] = np.stack([r.expert_weights for r in records]).astype(
                np.float32, copy=False)
        try:
            with pending.open("wb") as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, final)
        except Exception as exc:
            raise TraceWriteError(f"failed to commit {shard_name}: {exc}") from exc
        del self._buffer[:count]
        entry = {"file": shard_name, "records": count}
        self._shards.append(entry)
        self._record_count += count

    def _write_status(self) -> None:
        self._status.update(record_count=self._record_count,
                            buffered_record_count=len(self._buffer),
                            step_count=self._step_count,
                            shards=list(self._shards),
                            request_count=len(self._seen_requests),
                            terminal_request_count=len(self._request_terminal),
                            cancelled_request_count=len(self._cancelled_requests))
        atomic_json(self.run_dir / "status.json", self._status)

    def fail(self, error: BaseException) -> None:
        if self._closed:
            return
        self._failed = True
        self._status["state"] = "failed"
        self._status["complete"] = False
        self._status["errors"].append({"type": type(error).__name__, "message": str(error)})
        self._status["finished_at"] = _now()
        try:
            self._write_status()
        except Exception:
            pass

    def close(self, *, require_terminal_requests: bool = True) -> None:
        self._ensure_writable()
        try:
            if require_terminal_requests:
                missing = self._seen_requests - self._request_terminal
                if missing:
                    raise IntegrityError(f"requests lack terminal events: {sorted(missing)}")
            self._flush()
            self._write_status()
            from .reader import TraceReader
            issues = TraceReader(self.run_dir).validate(require_complete_status=False)
            if issues:
                raise IntegrityError("; ".join(issues))
            self._status.update(state="complete", complete=True, finished_at=_now())
            self._status["all_requests_completed_normally"] = not self._cancelled_requests
            self._write_status()
            self._closed = True
        except Exception as exc:
            self.fail(exc)
            raise

    def _ensure_writable(self) -> None:
        if self._closed:
            raise TraceWriteError("trace session is closed")
        if self._failed:
            raise TraceWriteError("trace session has failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
