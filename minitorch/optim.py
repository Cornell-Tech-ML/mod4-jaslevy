from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Resets the gradients of all parameters in the model to `None`.

        This method iterates over all parameters in the model and ensures that
        their gradient attributes (`derivative` or `grad`) are cleared. This is
        typically called at the beginning of each training step to ensure that
        gradients from a previous step do not interfere with the current step.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Performs a single optimization step to update the parameters.

        This method updates each parameter in the model based on its gradient and the learning rate.
        If a parameter has a `derivative` or `grad` attribute, it applies the update rule:

        - For parameters with a `derivative`: `parameter = parameter - learning_rate * derivative`
        - For parameters with a `grad`: `parameter = parameter - learning_rate * grad`

        Parameters without gradients or derivatives are skipped.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
