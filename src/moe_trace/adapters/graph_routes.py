"""Device-only route slots for graph capture; host conversion belongs to callers.

One bank belongs to one graph entry. Replay executes the recorded ``copy_``
operations without invoking Python again. Snapshot must run after graph replay
on its stream, before the next replay overwrites the slots.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteSnapshot:
    layer: str
    expert_ids: object
    weights: object | None
    metadata: object | None = None


class GraphRouteBank:
    def __init__(self, max_layers):
        if type(max_layers) is not int or max_layers < 1:
            raise ValueError("max_layers must be a positive integer")
        self.max_layers = max_layers
        self._slots = {}
        self._sealed = False

    def record(self, layer, expert_ids, weights=None, metadata=None):
        """Called once per layer during capture, never during replay.

        ``clone`` allocates an owned slot and records its device copy in the
        graph. Keeping the bank alive keeps its output storage alive as well.
        """
        if self._sealed:
            raise RuntimeError("cannot record into a sealed graph route bank")
        if not isinstance(layer, str) or not layer:
            raise ValueError("layer must be a nonempty string")
        if layer in self._slots:
            raise ValueError(f"duplicate graph route layer: {layer}")
        if len(self._slots) >= self.max_layers:
            raise ValueError("graph route bank layer limit exceeded")
        if len(expert_ids.shape) != 2:
            raise ValueError("expert IDs must have shape [rows, top_k]")
        if weights is not None and weights.shape != expert_ids.shape:
            raise ValueError("route weights and IDs must have equal shapes")
        self._slots[layer] = RouteSnapshot(
            layer, expert_ids.clone(), None if weights is None else weights.clone(), metadata)

    def seal(self):
        if not self._slots:
            raise RuntimeError("graph capture recorded no routes")
        self._sealed = True

    def snapshot(self):
        """Own replay results before reuse; enqueue copies on replay's stream.

        No implicit CPU synchronization is performed. A consumer on another
        stream must wait for the producer and retain these tensors until its
        copies complete. Only a completed host copy may be serialized.
        """
        if not self._sealed:
            raise RuntimeError("graph route bank must be sealed before snapshot")
        return tuple(RouteSnapshot(s.layer, s.expert_ids.clone(),
                                   None if s.weights is None else s.weights.clone(), s.metadata)
                     for s in self._slots.values())
