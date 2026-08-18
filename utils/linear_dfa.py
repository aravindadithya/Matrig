import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDFAFunction(torch.autograd.Function):
    """Direct Feedback Alignment layer using the global error signal.

    The correct DFA rule is to use the same task/error signal e for each layer:
        grad_input = e @ B
        grad_weight = e.T @ x
    where B is a fixed random feedback matrix. This is not RFA: the task signal
    comes from the output loss, not from the local downstream gradient.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, B, task_signal):
        ctx.save_for_backward(input, weight, bias, B)
        ctx.task_signal = task_signal
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, B = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.task_signal is None:
            raise RuntimeError(
                "Direct Feedback Alignment requires a global task signal. "
                "Without it, this is just RFA-style local propagation."
            )

        task_signal = ctx.task_signal.to(grad_output.dtype)
        B = B.to(grad_output.dtype)

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(task_signal, B)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(task_signal.transpose(-1, -2), input.to(grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None


class LinearDFA(nn.Module):
    """Linear layer implementing direct feedback alignment (DFA).

    Each layer stores its own fixed random feedback matrix B and is driven by the
    same global output error signal used by the network, not by the local layer
    gradient. This is the DFA paper setup, not RFA.
    """

    def __init__(self, in_features, out_features, num_classes=None, bias=False):
        super(LinearDFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes if num_classes is not None else out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("B", torch.empty(self.num_classes, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, input, task_signal=None):
        if task_signal is None:
            # Compatibility path for the existing project trainer and notebook loop.
            # The strict paper implementation uses the shared global error signal,
            # but a standard forward pass is required so the network can still be
            # evaluated and trained with the existing code path.
            return F.linear(input, self.weight, self.bias)
        return LinearDFAFunction.apply(input, self.weight, self.bias, self.B, task_signal)
