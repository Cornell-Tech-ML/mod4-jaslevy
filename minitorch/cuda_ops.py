# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN001, ANN003
    """JIT compile the given function for CUDA device execution.

    Args:
    ----
    fn: The function to be JIT compiled.
    kwargs: Additional keyword arguments for the JIT compilation.

    Returns:
    -------
    A CUDA kernel compiled from the given function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN001, ANN003
    """JIT compile the given function for CUDA execution.

    Args:
    ----
    fn: The function to be JIT compiled.
    kwargs: Additional keyword arguments for the JIT compilation.

    Returns:
    -------
    A CUDA kernel compiled from the given function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Creates a CUDA-accelerated element-wise zip function for tensors.

        This method generates a CUDA kernel that performs an element-wise operation
        on two tensors using the provided binary function `fn`. The function
        supports broadcasting to handle tensors of different shapes.

        Args:
        ----
        fn (Callable[[float, float], float]): A binary function that defines the
            operation to perform on the elements of the two tensors (e.g., addition,
            subtraction, etc.).

        Returns:
        -------
        Callable[[Tensor, Tensor], Tensor]: A function that takes two tensors as
            input, applies the `fn` operation to each pair of elements, and returns
            a new tensor containing the results. This function utilizes CUDA for
            efficient computation.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Perform reduction along a specified dimension of a tensor using GPU parallelization.

        This function takes in a reduction function `fn` and applies it to the elements
        of the input tensor `a` along the specified dimension `dim`. The reduction is
        performed in parallel using CUDA kernels.

        The output tensor has the same shape as the input tensor, but with the specified
        dimension reduced to size 1.

        Args:
        ----
        fn: A callable that takes two floats and returns a float. This function is used
            to perform the reduction.
        start: The starting value for the reduction.
        a: The input tensor to reduce.
        dim: The dimension to reduce.

        Returns:
        -------
        A tensor with the same shape as `a`, but with the specified dimension reduced.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication between two tensors `a` and `b` using GPU parallelization.

        This function ensures that both input tensors are treated as 3-dimensional tensors
        to support batch matrix multiplication. If either input is 2-dimensional, it is
        temporarily reshaped to a 3-dimensional tensor. The output tensor is computed by
        broadcasting the batch dimensions and then performing the matrix multiplication
        along the last two dimensions.

        Args:
        ----
        a (Tensor): The first input tensor. Must have shape `[batch_size, m, n]` or `[m, n]`.
        b (Tensor): The second input tensor. Must have shape `[batch_size, n, p]` or `[n, p]`.

        Returns:
        -------
        Tensor: The resulting tensor after matrix multiplication. If both inputs are
                2-dimensional, the output will also be 2-dimensional; otherwise, it
                will be 3-dimensional with shape `[batch_size, m, p]`.

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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # Implement for Task 3.3.
        # Guard
        if i >= out_size:
            return
        # converting to flat idx to multidim idx
        to_index(i, out_shape, out_index)
        # Map the output index to the corresponding input index via broadcasting
        broadcast_index(out_index, out_shape, in_shape, in_index)

        # Convert the multidimensional indices to flat positions in storage
        out_pos = index_to_position(out_index, out_strides)
        in_pos = index_to_position(in_index, in_strides)

        # Apply the function and write the result to the output storage
        out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Implement for Task 3.3.
        # Guard against threads that are out of bounds
        if i < out_size:
            # Calculate the output index
            to_index(i, out_shape, out_index)

            # Broadcast indices for a and b
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Get storage positions
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Apply the operation and write to output
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // r'''\'''text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Implement for Task 3.3.
    cache[pos] = 0.0

    # Load elements into shared memory
    if i < size:
        cache[pos] = a[i]

    # Synchronize threads before reduction
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
            cache[pos] += cache[pos + stride]
        stride *= 2
        cuda.syncthreads()

    # Write the result of the block to the output
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """A practice sum kernel to prepare for reduce."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Implement for Task 3.3.
        if out_pos >= out_size:
            return
        to_index(out_pos, out_shape, out_index)
        cache[pos] = reduce_value
        accumulator = reduce_value
        for i in range(pos, a_shape[reduce_dim], BLOCK_DIM):
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            for j in range(len(out_index)):
                a_index[j] = out_index[j]

            a_index[reduce_dim] = i
            a_pos = index_to_position(a_index, a_strides)
            accumulator = fn(accumulator, a_storage[a_pos])

        cache[pos] = accumulator
        cuda.syncthreads()

        stride = 1
        while stride < BLOCK_DIM:
            if pos % (2 * stride) == 0 and pos + stride < BLOCK_DIM:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride *= 2

        if pos == 0:
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # Implement for Task 3.3.
    # Puts a and b into shared memory for efficient access
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Shared memory indices (within the block)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Global Matrix indices (over the whole matrix) for given thread
    row = cuda.blockIdx.y * cuda.blockDim.y + ty
    col = cuda.blockIdx.x * cuda.blockDim.x + tx

    # Accumulator
    temp = 0.0

    # iterate over k sections of the matrix.
    #  The range calculates the number of sections needed to cover the whole matrix,
    #  where each section is BLOCK_DIM in size. If the matrix size is not evenly divisible
    # by BLOCK_DIM, the last section handles the remaining elements.
    for k in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
        # Copy data from global memory to shared memory (loading) if
        # the thread's row and column indices are within the matrix bounds.
        # Tis is for the first matrix (a)
        if row < size and k * BLOCK_DIM + tx < size:
            a_shared[ty, tx] = a[row * size + k * BLOCK_DIM + tx]

        # For out-of-bounds threads, set the shared memory value to 0
        else:
            a_shared[ty, tx] = 0.0

        # Copy data from global memory to shared memory (loading) if
        # the thread's row and column indices are within the matrix bounds.
        # This is for the second matrix (b)
        if col < size and k * BLOCK_DIM + ty < size:
            b_shared[ty, tx] = b[(k * BLOCK_DIM + ty) * size + col]
        # For out-of-bounds threads, set the shared memory value to 0
        else:
            b_shared[ty, tx] = 0.0
        # Sync threads within the block to insure they have all loaded data
        # to shared memory
        cuda.syncthreads()

        # Perform the dot product for the current section of matrices 'a' and 'b'.
        # Each thread computes a partial sum for its assigned element in the output matrix.
        for n in range(BLOCK_DIM):
            temp += a_shared[ty, n] * b_shared[n, tx]

        # Sync threads within the block to insure they have made the temp calculation
        cuda.syncthreads()

    # Write the result to global memory once all sections are processed. The if statement
    # checks if the thread's row and column indices are within the matrix bounds to prevent
    # out-of-bounds writes, and the indexing insures the correct element is written to the
    # correct location in the output matrix.
    if row < size and col < size:
        out[row * size + col] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """CUDA matrix multiply.

    The following code will CUDA compile fast versions your tensor_data functions.
    If you get an error, read the docs for NUMBA as to what is allowed
    in these functions.

    Args:
    ----
        a (Tensor): tensor to be multiplied
        b (Tensor): tensor to be multiplied

    Returns:
    -------
        TensorData: result of the matrix multiplication

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # Implement for Task 3.4.

    # Accumulator
    temp = 0.0

    # Iterate over section of the shared dimension of the input
    # matrices. Each iteration processes a BLOCK_DIM x BLOCK_DIM section.
    for k in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        # Checking if the current thread is within the bounds of a.
        # Only rows of valid indices are copied into the shared memory.
        if i < a_shape[-2] and k * BLOCK_DIM + pj < a_shape[-1]:
            # Load into shared memory
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride  # Batch
                + i * a_strides[-2]  # Row
                + (k * BLOCK_DIM + pj) * a_strides[-1]  # curent section (col)
            ]
        # For out-of-bounds threads, set the shared memory value to 0
        else:
            a_shared[pi, pj] = 0.0

        # Checking if the current thread is within the bounds of b.
        # Only rows of valid indices are copied into the shared memory.
        if j < b_shape[-1] and k * BLOCK_DIM + pi < b_shape[-2]:
            # Load into shared memory with offsets for correct position
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride  # Batch
                + (k * BLOCK_DIM + pi) * b_strides[-2]  # Row
                + j * b_strides[-1]  # column
            ]
        # For out-of-bounds threads, set the shared memory value to 0
        else:
            b_shared[pi, pj] = 0.0

        # Sync threads within the block to insure they have all loaded data
        # to shared memory
        cuda.syncthreads()

        # Perform the dot product for the current section of matrices 'a' and 'b'.
        # Each thread computes a partial sum for its assigned element in the output matrix.
        for n in range(BLOCK_DIM):
            temp += a_shared[pi, n] * b_shared[n, pj]

        # Sync threads within the block to insure they have made the temp calculation
        cuda.syncthreads()

    # Write the final result for the assigned element in the output matrix to global memory.
    if (
        i < out_shape[-2] and j < out_shape[-1]
    ):  # Checking that the indices are within the bounds of the output.
        # location for correct position (batch, row, col) offset
        out[batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]] = temp


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
