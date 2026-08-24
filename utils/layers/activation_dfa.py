import torch
import torch.nn as nn
import torch.nn.functional as F


class DFAActivationFunction(torch.autograd.Function):
    """Activation layer for Direct Feedback Alignment.

    Forward: applies a given activation function (e.g., F.relu, F.sigmoid).
    Backward: receives the global error `e` as `grad_output`.
              Uses autograd to compute the derivative `f'(z)` of the activation.
              Returns a single gradient for the input: a tuple `(e, f'(z))`.
              The preceding DFA linear layer will unpack this tuple.
    """

    @staticmethod
    def forward(ctx, input, activation_fn):
        ctx.save_for_backward(input)
        ctx.activation_fn = activation_fn
        return activation_fn(input)

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        e = grad_output  # global error signal

        activation_fn = ctx.activation_fn

        # Compute derivative using autograd
        with torch.enable_grad():
            z_tmp = z.detach().requires_grad_(True)
            out = activation_fn(z_tmp)
            deriv = torch.autograd.grad(
                outputs=out,
                inputs=z_tmp,
                grad_outputs=torch.ones_like(out),
            )[0]

        # Return single gradient for input: a tuple (e, deriv)
        # and None for the non-tensor activation_fn argument
        return (e, deriv)


class DFAActivation(nn.Module):
    """Non-linearity layer for DFA that passes the global error and derivative
    to the preceding linear layer.

    Args:
        activation_fn: callable from torch.nn.functional, e.g. F.relu, F.sigmoid.
    """

    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, input):
        return DFAActivationFunction.apply(input, self.activation_fn)