from __future__ import annotations

from typing import Union, TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given arguments.

        This method will create a new Scalar variable with the result of the computation
        and a history that contains the function and the arguments.

        Args:
        ----
            *vals: The arguments to the function.

        Returns:
        -------
            A new Scalar variable with the result of the computation.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)
        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Addition function $f(x, y) = x + y$

        Args:
        ----
            ctx: The context for the computation.
            a: The first argument.
            b: The second argument.

        Returns:
        -------
            The result of the computation.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The derivative of the addition function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Log function $f(x) = log(x)$

        Args:
        ----
            ctx: The context for the computation.
            a: The argument.

        Returns:
        -------
            The result of the computation.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the log function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    r"""Multiplication function $f(x, y) = x \times y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        r"""Multiplication function $f(x, y) = x \times y$

        Args:
        ----
            ctx: The context for the computation.
            a: The first argument.
            b: The second argument.

        Returns:
        -------
            The result of the computation.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """The derivative of the multiplication function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        a, b = ctx.saved_values
        return (d_output * b, d_output * a)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negation function $f(x) = -x$

        Args:
        ----
            ctx: The context for the computation.
            a: The argument to the function.

        Returns:
        -------
            The result of the computation.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the negation function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        return -d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Inverse function $f(x) = 1 / x$

        Args:
        ----
            ctx: The context for the computation.
            a: The argument to the function.

        Returns:
        -------
            The result of the computation.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the inverse function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Sigmoid(ScalarFunction):
    r"""Sigmoid function $f(x) =  \frac{1}{1 + e^{-x}}$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        r"""Sigmoid function $f(x) =  \frac{1}{1 + e^{-x}}$

        Args:
        ----
            ctx: The context for the computation.
            a: The argument to the function.

        Returns:
        -------
            The result of the computation.

        """
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        r"""The derivative of the sigmoid function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return d_output * operators.sigmoid(a) * (1 - operators.sigmoid(a))


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        r"""ReLU function $f(x) = max(0, x)$

        Args:
        ----
            ctx: The context for the computation.
            a: The argument to the function.

        Returns:
        -------
            The result of the computation.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        r"""The derivative of the ReLU function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        r"""Exponential function $f(x) = e^x$

        Args:
        ----
            ctx: The context for the computation.
            a: The argument to the function.

        Returns:
        -------
            The result of the computation.

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        r"""The derivative of the exponential function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class LT(ScalarFunction):
    r"""Less-than function $f(x) = 1 \text{ if } x < 0, 0 \text{ otherwise}$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        r"""Less-than function $f(x, y) = 1 \text{ if } x < y, 0 \text{ otherwise}$

        Args:
        ----
            ctx: The context for the computation.
            a: The first argument.
            b: The second argument.

        Returns:
        -------
            The result of the computation.

        """
        ctx.save_for_backward(a, b)
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Union[float, Tuple[float, float]]:
        """The derivative of the less-than function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        return 0


class EQ(ScalarFunction):
    r"""Equality function $f(x) = 1 \text{ if } x == 0, 0 \text{ otherwise}$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        r"""Equality function $f(x, y) = 1 \text{ if } x == y, 0 \text{ otherwise}$

        Args:
        ----
            ctx: The context for the computation.
            a: The first argument.
            b: The second argument.

        Returns:
        -------
            The result of the computation.

        """
        ctx.save_for_backward(a, b)
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the equality function.

        Args:
        ----
            ctx: The context for the computation.
            d_output: The derivative of the output.

        Returns:
        -------
            The derivative of the input.

        """
        (a, b) = ctx.saved_values
        return d_output if a == b else 0
