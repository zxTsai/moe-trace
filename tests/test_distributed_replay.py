from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from moe_trace.distributed.decode import Worker, DecodeRow
from moe_trace.distributed.replay import InputBinding, ReplayRouteBuffer
from moe_trace.errors import IntegrityError

torch = pytest.importorskip("torch")


def fixture():
    owner = Worker("deployment", "run", "D0", "w0", 0, 0, 0, 0)
    row = DecodeRow("engine", "exec1", 1, "request", "local", 0, 7, 41, "target_decode")
    buffer = ReplayRouteBuffer(owner, "target", 20, 4, 3, 2, weight_dtype=torch.float32)
    return buffer, InputBinding(2, owner, row)


def test_padding_reorder_copy_and_drain_after_completion():
    buffer, binding = fixture()
    ids = torch.tensor([[99, 99], [99, 99], [3, 1]])  # Invalid padding must never be validated as user rows.
    weights = torch.tensor([[0., 0.], [0., 0.], [0.75, 0.25]])
    completed = []
    completion = NS(synchronize=lambda: completed.append(True))
    bindings = [binding]
    buffer.begin(bindings)
    bindings.clear()  # Pending snapshot is independent of mutable batch list.
    buffer.capture_copy(ids, weights)
    ids[2].fill_(0)
    routes = buffer.drain(completion)
    assert completed == [True]
    assert routes[0].expert_ids == (3, 1)
    assert routes[0].weights == (0.75, 0.25)
    buffer.begin([replace(binding, row=replace(binding.row, execution="exec2"))])
    buffer.capture_copy(torch.tensor([[0, 1], [1, 2], [2, 0]]), weights)
    assert buffer.drain(completion)[0].expert_ids == (2, 0)
    assert routes[0].expert_ids == (3, 1)


def test_buffer_cannot_be_reused_while_pending():
    buffer, binding = fixture()
    buffer.begin([binding])
    with pytest.raises(IntegrityError, match="pending"):
        buffer.begin([binding])


@pytest.mark.parametrize("copies", [0, 2])
def test_stale_or_duplicate_copy_fails(copies):
    buffer, binding = fixture()
    buffer.begin([binding])
    for _ in range(copies):
        buffer.capture_copy(torch.tensor([[0, 1], [1, 2], [2, 3]]), torch.ones(3, 2))
    with pytest.raises(IntegrityError, match="copy"):
        buffer.drain(NS(synchronize=lambda: None))
    assert buffer.failed


def test_completion_error_fails_buffer():
    buffer, binding = fixture()
    buffer.begin([binding])
    def fail():
        raise RuntimeError("device failed")
    with pytest.raises(RuntimeError):
        buffer.drain(NS(synchronize=fail))
    assert buffer.failed


def test_warmup_is_not_a_user_route():
    buffer, binding = fixture()
    buffer.capture_copy(torch.tensor([[0, 1], [1, 2], [2, 3]]), torch.ones(3, 2))
    buffer.reset_after_capture(NS(synchronize=lambda: None))
    buffer.begin([binding])
    buffer.capture_copy(torch.tensor([[0, 1], [1, 2], [3, 1]]), torch.ones(3, 2))
    assert buffer.drain(NS(synchronize=lambda: None))[0].expert_ids == (3, 1)
    with pytest.raises(IntegrityError, match="reset"):
        buffer.reset_after_capture(NS(synchronize=lambda: None))


@pytest.mark.parametrize("tensor_row", [-1, 3, True])
def test_invalid_effective_tensor_row(tensor_row):
    buffer, binding = fixture()
    with pytest.raises(IntegrityError):
        buffer.begin([replace(binding, tensor_row=tensor_row)])


def test_duplicate_input_and_wrong_dtype_fail():
    buffer, binding = fixture()
    with pytest.raises(IntegrityError):
        buffer.begin([binding, binding])
    with pytest.raises(IntegrityError, match="dtype"):
        buffer.capture_copy(torch.ones(3, 2), torch.ones(3, 2))
