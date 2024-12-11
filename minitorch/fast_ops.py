from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np

from numba import prange
from numba import njit as _njit

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    MAX_DIMS,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a given function using Numba for optimized execution.

    This function wraps the `numba.njit` decorator with default parameters to enable
    function inlining and apply any additional user-specified optimizations.

    Args:
    ----
        fn: The function to be JIT-compiled.
        **kwargs: Additional keyword arguments for customization of the JIT compilation.

    Returns:
    -------
        The JIT-compiled version of the input function, which can be executed with
        optimized performance.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        size = np.prod(out_shape)

        stride_aligned = np.array_equal(out_strides, in_strides) and np.array_equal(
            out_shape, in_shape
        )
        if stride_aligned:
            for i in prange(size):
                out[i] = fn(in_storage[i])
        else:
            for i in prange(size):
                local_out_index = np.zeros(MAX_DIMS, dtype=np.int32)
                to_index(i, out_shape, local_out_index)

                local_in_index = np.zeros(MAX_DIMS, dtype=np.int32)
                broadcast_index(local_out_index, out_shape, in_shape, local_in_index)

                out_pos = index_to_position(local_out_index, out_strides)
                in_pos = index_to_position(local_in_index, in_strides)

                out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        size = np.prod(out_shape)
        stride_aligned = (
            np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
        )

        if stride_aligned:
            for i in prange(size):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for i in prange(size):
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)
                a_i = np.zeros(MAX_DIMS, dtype=np.int32)
                b_i = np.zeros(MAX_DIMS, dtype=np.int32)

                to_index(i, out_shape, out_index)

                broadcast_index(out_index, out_shape, a_shape, a_i)
                broadcast_index(out_index, out_shape, b_shape, b_i)

                out_pos = index_to_position(out_index, out_strides)
                a_pos = index_to_position(a_i, a_strides)
                b_pos = index_to_position(b_i, b_strides)

                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        size = np.prod(out_shape)
        out_shape_np = np.asarray(out_shape, dtype=np.int32)
        a_shape_np = np.asarray(a_shape, dtype=np.int32)
        reduce_size = a_shape[reduce_dim]

        for i in prange(size):
            out_index = np.zeros_like(out_shape_np)
            a_i = np.zeros_like(a_shape_np)
            to_index(i, out_shape_np, out_index)
            a_i[:] = out_index
            a_i[reduce_dim] = 0

            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_i, a_strides)

            accumulator = a_storage[a_pos]

            for j in range(1, reduce_size):
                a_i[reduce_dim] = j
                a_pos = index_to_position(a_i, a_strides)
                accumulator = fn(accumulator, a_storage[a_pos])

            out[out_pos] = accumulator

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    out_batch_stride = out_strides[0] if len(out_shape) == 3 else 0
    batch_size = out_shape[0] if len(out_shape) == 3 else 1
    out_rows, out_cols = out_shape[-2], out_shape[-1]
    inner_dim = a_shape[-1]

    for batch in prange(batch_size):
        for i in range(out_rows):
            for j in range(out_cols):
                sum_value = 0.0
                for k in range(inner_dim):
                    a_pos = (
                        batch * a_batch_stride + i * a_strides[-2] + k * a_strides[-1]
                    )
                    b_pos = (
                        batch * b_batch_stride + k * b_strides[-2] + j * b_strides[-1]
                    )
                    sum_value += a_storage[a_pos] * b_storage[b_pos]
                out_pos = (
                    batch * out_batch_stride + i * out_strides[-2] + j * out_strides[-1]
                )
                out[out_pos] = sum_value


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
