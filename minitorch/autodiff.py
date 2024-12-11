from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    result = (
        f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
        - f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])
    ) / (2 * epsilon)
    return result


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """

    @property
    def unique_id(self) -> int:
        """A unique identifier for this variable.

        The identifier is a counter that is incremented every time a new
        variable is created.

        """
        ...

    def is_leaf(self) -> bool:
        """A boolean if this variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """True if this variable was created as a constant and not as a result of
        an operation.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """A list of all variables that directly depend on this variable in the computation graph.

        This list is used to compute the derivative of the output with respect to the input of the computation graph.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the derivative of the output with respect to the input of the computation graph using the chain rule.

        Args:
        ----
            d_output (Any): derivative of the output with respect to the output of the computation graph

        Returns:
        -------
            An iterable of tuples. The first element of the tuple is the input variable and the second element is
            the derivative of the output with respect to the input variable computed using the chain rule.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    result = []

    def rsort(variable: Variable) -> None:
        if variable.unique_id not in visited and not variable.is_constant():
            visited.add(variable.unique_id)
            for parent in variable.parents:
                rsort(parent)
            result.append(variable)

    rsort(variable)
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
    variable: The right-most variable
    deriv (Any): Its derivative that we want to propagate backward to the leaves.


    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    order = list(topological_sort(variable))[::-1]
    variable_derivatives = {variable.unique_id: deriv}
    for v in order:
        if v.is_leaf():
            v.accumulate_derivative(variable_derivatives.get(v.unique_id, 0))
        else:
            dvar = variable_derivatives.get(v.unique_id, 0)
            for parent, d_parent in v.chain_rule(d_output=dvar):
                variable_derivatives.setdefault(parent.unique_id, 0)
                variable_derivatives[parent.unique_id] += d_parent


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors for the forward pass."""
        return self.saved_values
