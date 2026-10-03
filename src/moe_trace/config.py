from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

DEFAULT_OUTPUT_ROOT = Path("/home/czx/code/moe-routing/outputs")
_TRACE_KEYS = {
    "enabled", "output_root", "phases", "layers", "request_labels",
    "save_token_ids", "save_expert_weights", "strict", "buffer_records",
    "shard_records", "engine_id",
}
_VLLM_KEYS = {
    "enforce_eager", "enable_prefix_caching", "speculative_model",
    "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
    "async_scheduling",
}


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    output_root: Path = DEFAULT_OUTPUT_ROOT
    phases: tuple[str, ...] = ("prefill", "decode")
    layers: tuple[int, ...] | None = None
    request_labels: tuple[str, ...] | None = None
    save_token_ids: bool = False
    save_expert_weights: bool = False
    strict: bool = True
    buffer_records: int = 8192
    shard_records: int = 4096
    engine_id: str = "engine-0"
    vllm: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_root"] = str(self.output_root)
        value["phases"] = list(self.phases)
        if self.layers is not None:
            value["layers"] = list(self.layers)
        if self.request_labels is not None:
            value["request_labels"] = list(self.request_labels)
        return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if type(value) is not bool:
        raise ConfigError(f"trace.{key} must be a boolean")
    return value


def load_config(path: str | Path) -> TraceConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    root = _mapping(raw, "config")
    unknown_root = set(root) - {"trace", "vllm"}
    if unknown_root:
        raise ConfigError(f"unknown top-level fields: {sorted(unknown_root)}")
    trace = _mapping(root.get("trace"), "trace")
    vllm = _mapping(root.get("vllm"), "vllm")
    unknown_trace = set(trace) - _TRACE_KEYS
    unknown_vllm = set(vllm) - _VLLM_KEYS
    if unknown_trace:
        raise ConfigError(f"unknown trace fields: {sorted(unknown_trace)}")
    if unknown_vllm:
        raise ConfigError(f"unknown vllm fields: {sorted(unknown_vllm)}")

    phases = trace.get("phases", ["prefill", "decode"])
    if (not isinstance(phases, list) or not phases
            or any(p not in {"prefill", "decode"} for p in phases)
            or len(phases) != len(set(phases))):
        raise ConfigError("trace.phases must be a unique non-empty list of prefill/decode")
    layers = trace.get("layers")
    if layers is not None and (not isinstance(layers, list)
                               or any(type(x) is not int or x < 0 for x in layers)
                               or len(layers) != len(set(layers))):
        raise ConfigError("trace.layers must be null or a unique list of non-negative integers")
    labels = trace.get("request_labels")
    if labels is not None and (not isinstance(labels, list)
                               or any(not isinstance(x, str) or not x for x in labels)
                               or len(labels) != len(set(labels))):
        raise ConfigError("trace.request_labels must be null or a unique list of strings")

    buffer_records = trace.get("buffer_records", 8192)
    shard_records = trace.get("shard_records", 4096)
    if type(buffer_records) is not int or buffer_records <= 0:
        raise ConfigError("trace.buffer_records must be a positive integer")
    if type(shard_records) is not int or shard_records <= 0:
        raise ConfigError("trace.shard_records must be a positive integer")
    if shard_records > buffer_records:
        raise ConfigError("trace.shard_records cannot exceed trace.buffer_records")
    engine_id = trace.get("engine_id", "engine-0")
    if not isinstance(engine_id, str) or not engine_id:
        raise ConfigError("trace.engine_id must be a non-empty string")

    return TraceConfig(
        enabled=_bool(trace, "enabled", False),
        output_root=Path(trace.get("output_root", DEFAULT_OUTPUT_ROOT)),
        phases=tuple(phases),
        layers=None if layers is None else tuple(layers),
        request_labels=None if labels is None else tuple(labels),
        save_token_ids=_bool(trace, "save_token_ids", False),
        save_expert_weights=_bool(trace, "save_expert_weights", False),
        strict=_bool(trace, "strict", True),
        buffer_records=buffer_records,
        shard_records=shard_records,
        engine_id=engine_id,
        vllm=dict(vllm),
    )
