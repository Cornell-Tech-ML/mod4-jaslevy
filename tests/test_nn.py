import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors

# pyright: ignore[reportCallIssue]
# pyright: ignore[reportArgumentType]


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    # Property 1: Output shape should be correct for any input tensor
    out = minitorch.max(t, 2)
    assert out.shape == (2, 3, 1), "Output shape should be (2, 3, 1)"

    # Property 2: Max value should be greater than or equal to all values in reduced dimension
    for b in range(2):
        for c in range(3):
            max_val = out[b, c, 0]
            for k in range(4):
                assert (
                    t[b, c, k] <= max_val
                ), f"Max value {max_val} should be >= {t[b, c, k]}"

    # Property 3: Test gradient computation
    t.requires_grad_(True)
    out = minitorch.max(t, 2)
    grad_output = minitorch.tensor([[[1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0]]])
    out.backward(grad_output)

    assert t.grad is not None, "Gradient should not be None"

    # Property 4: For each position, gradients should sum to 1.0 and be distributed correctly
    for b in range(2):
        for c in range(3):
            max_val = out[b, c, 0]
            grad_sum = 0.0

            # Sum up all gradients
            for k in range(4):
                grad_sum += t.grad[b, c, k]

                # Check that non-maximum values have zero gradient
                if t[b, c, k] < max_val - 1e-2:  # Using same tolerance as is_close
                    assert_close(t.grad[b, c, k], 0.0)

            # Total gradient should be 1.0
            assert_close(grad_sum, 1.0)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
