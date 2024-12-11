"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,  # noqa: F401
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets requires_grad on this tensor.

        Args:
        ----
            x (bool): True if requires_grad is true for this tensor.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Returns True if requires_grad is true for this tensor.

        Returns
        -------
            bool: True if requires_grad is true for this tensor.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a new tensor of zeros of the same shape or a specified shape.

        Args:
        ----
            shape : shape of tensor (if None, same shape as self)

        Returns:
        -------
            New tensor of zeros

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable was created as a constant and not as a result of
        an operation.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """A list of all variables that directly depend on this variable in the computation graph.

        This list is used to compute the derivative of the output with respect to the input of the computation graph.

        Returns
        -------
            An iterable of all variables that directly depend on this variable in the computation graph.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the derivative of the output with respect to the input of the computation graph using the chain rule.

        Args:
        ----
            d_output (Any): derivative of the output with respect to the output of the computation graph

        Returns:
        -------
            An iterable of tuples. The first element of the tuple is the input variable and the second element is
            the derivative of the output with respect to the input variable computed using the chain rule.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Compute the derivatives of all leave nodes in the computation graph
        rooted at this variable.

        Args:
        ----
            grad_output: The derivative of the output with respect to this variable.
                If not provided, will be set to 1.0 if variable is a scalar, else
                an error will be raised.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    # Functions
    # TODO: Implement for Task 2.3.
    @property
    def size(self) -> int:
        """Return the total number of elements in the tensor."""
        return int(operators.prod(self.shape))

    def dims(self) -> int:
        """Return the number of dimensions of the tensor."""
        return len(self.shape)

    def __add__(self, b: TensorLike) -> Tensor:
        """Element-wise addition."""
        b = self._ensure_tensor(b)
        return Add.apply(self, b)

    def __radd__(self, b: TensorLike) -> Tensor:
        """Element-wise addition (right operand)."""
        return self.__add__(b)

    def __sub__(self, b: TensorLike) -> Tensor:
        """Element-wise subtraction."""
        b = self._ensure_tensor(b)
        return Add.apply(self, Neg.apply(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        """Element-wise multiplication."""
        b = self._ensure_tensor(b)
        return Mul.apply(self, b)

    def __rmul__(self, b: TensorLike) -> Tensor:
        """Element-wise multiplication (right operand)."""
        return self.__mul__(b)

    def __neg__(self) -> Tensor:
        """Negate the tensor."""
        return Neg.apply(self)

    def __eq__(self, b: TensorLike) -> Tensor:
        """Element-wise equality comparison."""
        b = self._ensure_tensor(b)
        return EQ.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        """Element-wise less-than comparison."""
        b = self._ensure_tensor(b)
        return LT.apply(self, b)

    def __gt__(self, b: TensorLike) -> Tensor:
        """Element-wise greater-than comparison."""
        b = self._ensure_tensor(b)
        return LT.apply(b, self)

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Check if all elements are non-zero."""
        if dim is None:
            # Reshape to a 1D tensor and reduce over all elements
            flat_tensor = self.contiguous().view(self.size)
            return All.apply(flat_tensor, Tensor.make([0], (1,), backend=self.backend))
        else:
            # Reduce over the specified dimension
            return All.apply(self, Tensor.make([dim], (1,), backend=self.backend))

    def abs(self) -> Tensor:
        """Element-wise absolute value."""
        return self.relu() + (-self).relu()

    def is_close(self, b: TensorLike, tol: float = 1e-5) -> Tensor:
        """Element-wise comparison for closeness within a tolerance."""
        b = self._ensure_tensor(b)
        diff = (self - b).abs()  # Use the subtraction operator directly
        tol_tensor = Tensor.make([tol], (1,), backend=self.backend)
        return LT.apply(diff, tol_tensor)

    def sigmoid(self) -> Tensor:
        """Sigmoid activation function."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """ReLU activation function."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Natural logarithm."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Exponential function."""
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sum over a specified dimension or all elements."""
        if dim is None:
            # Flatten the tensor to 1D and sum over all elements
            flat_tensor = self.contiguous().view(self.size)
            dim_tensor = self._ensure_tensor(0)
            # dim_tensor = Tensor.make([0], (1,), backend=self.backend)
            return Sum.apply(flat_tensor, dim_tensor)
        else:
            # Convert the dimension to a Tensor
            dim_tensor = self._ensure_tensor(dim)
            # dim_tensor = Tensor.make([dim], (1,), backend=self.backend)
            return Sum.apply(self, dim_tensor)

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Mean over a specified dimension or all elements."""
        total = self.sum(dim)
        if dim is None:
            return total / float(self.size)  # Explicitly convert to float
        return total / float(self.shape[dim])  # Explicitly convert to float

    def permute(self, *order: int, dim: Optional[int] = None) -> Tensor:
        """Permute the dimensions of the tensor.

        If `dim` is specified, permute the dimensions of the specified axis.
        """
        if dim is None:
            order_tensor = Tensor.make(list(order), (len(order),), backend=self.backend)
            return Permute.apply(self, order_tensor)
        else:
            order_tensor = Tensor.make([dim], (1,), backend=self.backend)
            return Permute.apply(self, order_tensor)

    def view(self, *shape: int, dim: Optional[int] = None) -> Tensor:
        """Return a tensor with the specified shape."""
        if dim is None:
            shape_tensor = Tensor.make(list(shape), (len(shape),), backend=self.backend)
            return View.apply(self, shape_tensor)
        dim_tensor = Tensor.make([dim], (1,), backend=self.backend)
        shape_tensor = Tensor.make(list(shape), (len(shape),), backend=self.backend)
        return View.apply(self, shape_tensor, dim_tensor)

    def zero_grad_(self) -> None:
        """Set the gradient to None."""
        self.grad = None
