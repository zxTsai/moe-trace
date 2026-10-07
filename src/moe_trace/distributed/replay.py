"""Experimental static route buffers, CPU tested only.

copy_ and generation.add_ are intended to be recorded inside the device graph.
begin/drain live outside graph capture. A single buffer cannot serve concurrent
replays: caller must drain before reuse. No NPU graph integration is supplied.
"""
from dataclasses import dataclass

from ..errors import IntegrityError


@dataclass(frozen=True)
class InputBinding:
    tensor_row: int
    owner: object
    row: object


class ReplayRouteBuffer:
    def __init__(self, observer, model_scope, layer, num_experts, capacity, top_k,
                 *, device="cpu", weight_dtype=None):
        import torch
        from .decode import Worker
        if not isinstance(observer, Worker):
            raise IntegrityError("observer must be a Worker")
        if not isinstance(model_scope, str) or not model_scope.strip():
            raise IntegrityError("model scope is required")
        for name, value in (("layer", layer), ("num_experts", num_experts),
                            ("capacity", capacity), ("top_k", top_k)):
            if type(value) is not int or value < (0 if name == "layer" else 1):
                raise IntegrityError(f"invalid {name}")
        if top_k > num_experts:
            raise IntegrityError("Top-K exceeds logical expert count")
        self.observer, self.scope, self.layer = observer, model_scope, layer
        self.num_experts = num_experts
        self.ids = torch.empty((capacity, top_k), dtype=torch.int64, device=device)
        self.weights = (torch.empty((capacity, top_k), dtype=weight_dtype, device=device)
                        if weight_dtype is not None else None)
        self.generation = torch.zeros((), dtype=torch.int64, device=device)
        self.previous = 0
        self.pending = None
        self.failed = False

    def capture_copy(self, ids, weights=None):
        """Device-only operations: must run/capture once per effective layer execution.

        IDs must already be logical routed-expert IDs, without shared columns.
        This does NOT infer logical identities from final physical IDs.
        """
        import torch
        if ids.shape != self.ids.shape or ids.dtype not in (torch.int32, torch.int64):
            raise IntegrityError("captured IDs have wrong shape or dtype")
        if ids.device != self.ids.device:
            raise IntegrityError("captured IDs belong to another device")
        if self.weights is not None:
            if (weights is None or weights.shape != self.weights.shape
                    or weights.device != self.weights.device or weights.dtype != self.weights.dtype):
                raise IntegrityError("captured weights have wrong shape/device/dtype")
            self.weights.copy_(weights)
        self.ids.copy_(ids)
        self.generation.add_(1)

    def reset_after_capture(self, completion):
        """Exclude warmup/capture. Only before the first user execution."""
        if self.pending is not None or self.failed or self.previous != 0:
            raise IntegrityError("cannot reset an active/used buffer")
        completion.synchronize()
        self.generation.zero_()

    def begin(self, bindings):
        from .decode import Worker, DecodeRow
        if self.failed or self.pending is not None:
            raise IntegrityError("buffer failed or still owns a pending replay")
        bindings = tuple(bindings)
        if not bindings:
            raise IntegrityError("effective input bindings are required")
        seen = set()
        for binding in bindings:
            if not isinstance(binding, InputBinding):
                raise IntegrityError("invalid input binding")
            if (type(binding.tensor_row) is not int or binding.tensor_row < 0
                    or binding.tensor_row >= self.ids.shape[0] or binding.tensor_row in seen):
                raise IntegrityError("duplicate/out-of-range tensor row")
            if not isinstance(binding.owner, Worker) or not isinstance(binding.row, DecodeRow):
                raise IntegrityError("binding requires immutable owner and row")
            if (binding.owner.deployment, binding.owner.run) != (self.observer.deployment, self.observer.run):
                raise IntegrityError("binding belongs to another run")
            seen.add(binding.tensor_row)
        if len({(b.row.source_engine, b.row.execution) for b in bindings}) != 1:
            raise IntegrityError("binding snapshot contains multiple executions")
        if len({b.row.input_row for b in bindings}) != len(bindings):
            raise IntegrityError("duplicate effective input row")
        self.pending = bindings

    def drain(self, completion):
        """Wait for copy completion, copy to CPU, and construct independent route values."""
        from .decode import Route
        if self.pending is None or self.failed:
            raise IntegrityError("no active replay or buffer failed")
        try:
            completion.synchronize()
            generation = int(self.generation.item())
            if generation != self.previous + 1:
                raise IntegrityError("missing, stale or repeated layer copy")
            indices = [b.tensor_row for b in self.pending]
            ids = self.ids[indices].detach().cpu().tolist()
            weights = self.weights[indices].detach().float().cpu().tolist() if self.weights is not None else None
            routes = [Route(self.observer, b.owner, b.row, self.scope, self.layer,
                            self.num_experts, tuple(ids[i]), tuple(weights[i]) if weights is not None else None)
                      for i, b in enumerate(self.pending)]
            self.previous = generation
            self.pending = None
            return routes
        except Exception:
            self.failed = True
            raise
