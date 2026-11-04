"""CPU simulation of graph execution, not NPU graph acceptance."""
import numpy as np
import pytest

from moe_trace.adapters.graph_routes import GraphRouteBank


class Tensor:
    def __init__(self, values, operations=None):
        self.values = np.array(values)
        self.operations = operations

    @property
    def shape(self):
        return self.values.shape

    def clone(self):
        output = Tensor(self.values.copy())
        if self.operations is not None:
            self.operations.append(lambda: np.copyto(output.values, self.values))
        return output


def test_replay_changes_routes_without_reentering_python_hook():
    operations = []
    ids = Tensor([[1, 2], [3, 4]], operations)
    weights = Tensor([[.4, .6], [.2, .8]], operations)
    bank = GraphRouteBank(1)
    bank.record("target.layers.0", ids, weights)
    bank.seal()
    first = bank.snapshot()[0]
    ids.values[:] = [[5, 6], [7, 8]]
    weights.values[:] = [[.1, .9], [.3, .7]]
    for operation in operations:
        operation()
    second = bank.snapshot()[0]
    assert first.expert_ids.values.tolist() == [[1, 2], [3, 4]]
    assert second.expert_ids.values.tolist() == [[5, 6], [7, 8]]
    assert second.weights.values.tolist() == [[.1, .9], [.3, .7]]
    # A later replay cannot overwrite an execution's private snapshot.
    ids.values[:] = 0
    for operation in operations:
        operation()
    assert second.expert_ids.values.tolist() == [[5, 6], [7, 8]]


def test_bank_rejects_incomplete_or_duplicate_capture():
    bank = GraphRouteBank(1)
    with pytest.raises(RuntimeError):
        bank.snapshot()
    with pytest.raises(RuntimeError):
        bank.seal()
    bank.record("layer0", Tensor([[1, 2]]))
    with pytest.raises(ValueError, match="duplicate"):
        bank.record("layer0", Tensor([[1, 2]]))
    with pytest.raises(ValueError, match="limit"):
        bank.record("layer1", Tensor([[1, 2]]))
    bank.seal()
    assert bank.snapshot()[0].weights is None
    with pytest.raises(RuntimeError, match="sealed"):
        bank.record("layer2", Tensor([[1, 2]]))


def test_banks_isolate_graph_shapes_and_validate_weight_shape():
    small, large = GraphRouteBank(1), GraphRouteBank(1)
    with pytest.raises(ValueError, match="equal shapes"):
        small.record("layer", Tensor([[1, 2]]), Tensor([[1]]))
    small.record("layer", Tensor([[1, 2]]))
    large.record("layer", Tensor([[3, 4], [5, 6]]))
    small.seal()
    large.seal()
    assert small.snapshot()[0].expert_ids.shape == (1, 2)
    assert large.snapshot()[0].expert_ids.shape == (2, 2)
