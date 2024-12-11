from typing import Tuple
from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor, Exp, Log


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling


    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling


    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.


    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # Make input contiguous before view operation
    input = input.contiguous()
    new_height = height // kh
    new_width = width // kw

    # Reshape the tensor to create windows
    inner = input.view(batch, channel, new_height, kh, new_width, kw)

    # Permute and reshape to get the final shape
    out = inner.permute(0, 1, 2, 4, 3, 5)
    out = out.contiguous()
    out = out.view(batch, channel, new_height, new_width, kh * kw)

    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.


    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling


    Returns:
    -------
        Pooled tensor

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    tiled, new_height, new_width = tile(input, kernel)

    pooled = tiled.sum(4) / (kh * kw)
    pooled = pooled.view(batch, channel, new_height, new_width)

    return pooled


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Compute the maximum value along dimension dim.
        Store necessary information for backward pass.
        """
        dim_int = int(dim.item())
        output = input.backend.max_reduce(input, dim_int)
        ctx.save_for_backward(input, output, dim_int)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the max operation.
        The gradient is distributed evenly among all positions that achieved the maximum.
        """
        input, output, dim = ctx.saved_values
        shape = list(input.shape)
        shape[dim] = 1
        expanded_output = output.view(*shape)

        mask = input == expanded_output

        counts = mask.sum(dim)

        expanded_counts = counts.view(*shape)

        # Distribute gradient evenly
        grad_input = mask / expanded_counts * grad_output

        return grad_input, tensor([0.0])


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum value along dimension dim.

    Args:
    ----
        input: input tensor
        dim: dimension to reduce

    Returns:
    -------
        Tensor containing maximum values along dimension dim


    """
    dim_tensor = tensor([dim])
    return Max.apply(input, dim_tensor)


def maxpool2d(input: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    """Compute a 2D maxpool over an input tensor.


    Args:
    ----
        input: input tensor with shape (batch_size, in_channels, height, width)
        kernel_size: height and width of pooling kernel


    Returns:
    -------
        Tensor pooled with kernel_size


    """
    batch_size, nchannels, height, width = input.shape
    kh, kw = kernel_size
    assert height % kh == 0 and width % kw == 0

    out_height = height // kh
    out_width = width // kw

    out = input.zeros((batch_size, nchannels, out_height, out_width))

    for b in range(batch_size):
        for c in range(nchannels):
            for i in range(out_height):
                for j in range(out_width):
                    max_val = input[b, c, i * kh, j * kw]
                    for ki in range(kh):
                        for kj in range(kw):
                            val = input[b, c, i * kh + ki, j * kw + kj]
                            max_val = max_val * (max_val >= val) + val * (val > max_val)
                    out[b, c, i, j] = max_val

    return out


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.


    Args:
    ----
        input: input tensor
        dim: dimension to apply softmax


    Returns:
    -------
        softmax tensor


    """
    exp_x = Exp.apply(input)

    sum_exp = exp_x.sum(dim)

    sum_exp = sum_exp.view(*[1 if i == dim else s for i, s in enumerate(input.shape)])

    return exp_x / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the logsoftmax as a tensor.


    Args:
    ----
        input: input tensor
        dim: dimension to apply logsoftmax


    Returns:
    -------
        logsoftmax tensor


    """
    exp_x = Exp.apply(input)

    sum_exp = exp_x.sum(dim)

    sum_exp = sum_exp.view(*[1 if i == dim else s for i, s in enumerate(input.shape)])

    return input - Log.apply(sum_exp)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to input tensor.


    Args:
    ----
        input : input tensor
        rate : dropout rate (1 - keep probability)
        ignore : if True, ignore dropout (useful for testing)


    Returns:
    -------
        Tensor of same shape as input


    """
    if ignore:
        return input

    if rate >= 1.0:
        return input * 0.0

    mask = rand(input.shape) > rate

    scale = 1.0 / (1.0 - rate)
    return mask * input * scale
