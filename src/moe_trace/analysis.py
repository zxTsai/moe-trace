from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .reader import TraceReader


def write_details_json(reader: TraceReader, path: str | Path, **filters: Any) -> None:
    """Export native JSON arrays and explicit storage validation status."""
    issues = reader.validate()
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump({
            "schema_version": reader.metadata["schema_version"],
            "run_id": reader.metadata["run_id"],
            "storage_complete": not issues,
            "validation_issues": issues,
            "filters": filters,
            "records": list(_filtered(reader.detail_rows(), **filters)),
        }, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _filtered(rows: Iterable[dict[str, Any]], *, request_id: str | None = None,
              step_id: int | None = None) -> Iterable[dict[str, Any]]:
    for row in rows:
        if request_id is not None and row["request_id"] != request_id:
            continue
        if step_id is not None and row["step_id"] != step_id:
            continue
        yield row


def expert_counts(reader: TraceReader, *, request_id: str | None = None,
                  step_id: int | None = None) -> list[dict[str, Any]]:
    counts: dict[tuple[str, int, int], int] = {}
    for phase in reader.metadata["config"]["phases"]:
        for layer in reader.metadata["moe_layers"]:
            for expert in range(int(reader.metadata["experts_per_layer"][str(layer)])):
                counts[(phase, layer, expert)] = 0
    for row in _filtered(reader.detail_rows(), request_id=request_id, step_id=step_id):
        for expert in row["expert_ids"]:
            counts[(row["phase"], row["layer_id"], expert)] += 1
    complete = reader.complete
    return [{"run_id": reader.metadata["run_id"], "complete": complete,
             "phase": phase, "layer_id": layer, "expert_id": expert, "count": count}
            for (phase, layer, expert), count in sorted(counts.items())]


def write_counts_csv(reader: TraceReader, path: str | Path, **filters: Any) -> None:
    rows = expert_counts(reader, **filters)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_details_csv(reader: TraceReader, path: str | Path, **filters: Any) -> None:
    rows = list(_filtered(reader.detail_rows(), **filters))
    fields = ["run_id", "engine_id", "rank_id", "step_id", "execution_id",
              "request_id", "phase", "token_position", "layer_id", "expert_ids",
              "token_id", "expert_weights"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["expert_ids"] = json.dumps(row["expert_ids"])
            if "expert_weights" in row:
                row["expert_weights"] = json.dumps(row["expert_weights"])
            writer.writerow(row)
