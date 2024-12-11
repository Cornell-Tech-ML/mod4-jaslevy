"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING
from typing import Optional

import numpy as np

import minitorch


from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Negation function $f(x) = -x$

        Args:
        ----
            ctx: The context for the computation.
            t1: The argument to the function.

        Returns:
        -------
            The result of the computation.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """The derivative of the negation function.

        Args:
        ----
            ctx: The context for the computation.
            grad_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Inverse function $f(x) = 1 / x$

        Args:
        ----
            ctx: The context for the computation.
            t1: The argument to the function.

        Returns:
        -------
            The result of the computation.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """The derivative of the inverse function.

        Args:
        ----
            ctx: The context for the computation.
            grad_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Addition function $f(x, y) = x + y$

        Args:
        ----
            ctx: The context for the computation.
            t1: The first argument to the function.
            t2: The second argument to the function.

        Returns:
        -------
            The result of the computation.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """The derivative of the addition function.

        Args:
        ----
            ctx: The context for the computation.
            grad_output: The derivative of the output.

        Returns:
        -------
            A tuple containing the derivatives of the input tensors.

        """
        grad_t1 = grad_output
        grad_t2 = grad_output

        return grad_t1, grad_t2


class All(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return t1.f.mul_reduce(t1, int(dim.item()))
        else:
            return t1.f.mul_reduce(
                t1.contiguous().view(int(operators.prod(t1.shape))), 0
            )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Backward pass for the All function."""
        t1, dim = ctx.saved_values

        # Check if all elements are non-zero
        all_true = (t1 != 0).all(dim)

        # If all elements are true, propagate the gradient back
        grad_input = grad_output * all_true.expand(t1.shape)

        return grad_input, None  # No gradient for the dim argument


# TODO: Implement for Task 2.3.


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise multiplication."""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for element-wise multiplication."""
        t1, t2 = ctx.saved_values

        # Compute gradients with respect to each input
        grad_t1 = grad_output * t2
        grad_t2 = grad_output * t1

        return grad_t1, grad_t2


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Sigmoid activation function."""
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for sigmoid activation."""
        (out,) = ctx.saved_values

        # Create a tensor of ones with the same shape and backend as `grad_output`
        one = minitorch.Tensor.make([1.0], (1,), backend=grad_output.backend)

        # Compute the gradient using the sigmoid derivative
        grad_input = grad_output * out * (one - out)

        return grad_input


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Element-wise ReLU activation function.

        Computes the element-wise ReLU of the input tensor `forward`.

        Args:
        ----
        t1: The input tensor.
        ctx: The context for the computation.

        Returns:
        -------
        A tensor with the same shape as `t1` containing the element-wise ReLU of `t1`.

        """
        out = t1.f.relu_map(t1)
        ctx.save_for_backward(t1)  # Save t1 for the backward pass
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for ReLU activation."""
        (t1,) = ctx.saved_values

        # ReLU derivative: 1 if t1 > 0, else 0
        grad_input = grad_output * (t1 > 0)
        return grad_input


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Logarithm function."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the log function."""
        (t1,) = ctx.saved_values

        # Gradient of log(x) is 1/x
        grad_input = grad_output / t1
        return grad_input


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Exponential function."""
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the exponential function."""
        (out,) = ctx.saved_values

        # Gradient of exp(x) is exp(x)
        grad_input = grad_output * out
        return grad_input


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim_tensor: Tensor) -> Tensor:
        """Sum over a dimension."""
        # Convert dim_tensor to an integer
        dim = int(dim_tensor.item())
        # ctx.save_for_backward(t1.shape, dim)
        # Call the appropriate backend function
        return t1.f.add_reduce(t1, dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the sum function."""
        # (shape, dim) = ctx.saved_valuea
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise less-than comparison."""
        ctx.save_for_backward(t1, t2)  # Save inputs for backward pass
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for less-than comparison."""
        t1, t2 = ctx.saved_values

        # Create zero gradients with the same shape as t1 and t2
        zero_grad_t1 = minitorch.Tensor.make(
            [0.0] * t1.size, t1.shape, backend=t1.backend
        )
        zero_grad_t2 = minitorch.Tensor.make(
            [0.0] * t2.size, t2.shape, backend=t2.backend
        )
        return zero_grad_t1, zero_grad_t2


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise equality comparison."""
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for equality comparison."""
        t1, t2 = ctx.saved_values

        # The derivative of EQ is always zero
        grad_t1 = t1.zeros(t1.shape)
        grad_t2 = t2.zeros(t2.shape)

        return grad_t1, grad_t2


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Element-wise is-close comparison."""
        return t1.f.is_close_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for is-close comparison."""
        t1, t2 = ctx.saved_values

        # The derivative of IsClose is always zero
        grad_t1 = t1.zeros(t1.shape)
        grad_t2 = t2.zeros(t2.shape)

        return grad_t1, grad_t2


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Permute tensor dimensions."""
        order_list = [int(order[i]) for i in range(order.size)]
        ctx.save_for_backward(t1, order_list)
        return t1._new(t1._tensor.permute(*order_list))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Optional[Tensor], float]:
        """Backward pass for permute."""
        t1, order = ctx.saved_values
        newT = minitorch.Tensor.make(
            grad_output._tensor._storage,
            t1.shape,
            t1._tensor.strides,
            backend=grad_output.backend,
        )
        return newT, float(0)


class View(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, shape: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1.shape)
        assert t1._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            t1._tensor._storage, tuple(shape2), backend=t1.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return t1.f.id_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(t: Tensor) -> Tensor:
            order = list(range(t.dims()))
            order[-2], order[-1] = order[-1], order[-2]
            return t._new(t._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the derivative of `f` with respect to `vals[arg]` at index `ind`
    using central difference.

    Args:
    ----
        f: function to compute derivative of
        *vals: tensor inputs to `f`
        arg: index of tensor to compute derivative of
        epsilon: small constant to compute derivative
        ind: index to compute derivative at

    Returns:
    -------
        float: derivative of `f` with respect to `vals[arg]` at index `ind`

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
