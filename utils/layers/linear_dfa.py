import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDFAFunction(torch.autograd.Function):
    """Direct Feedback Alignment layer using a global task signal.

    The layer receives a projected local signal of shape [batch, out_features].
    The gradient rule is:
        grad_input = task_signal @ R
        grad_weight = task_signal.T @ input
    where R is a fixed random feedback matrix with shape [out_features, in_features].
    This differs from RFA because the task signal is the shared global output error,
    projected to that layer's output dimension before the feedback step.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, R, task_signal):
        ctx.save_for_backward(input, weight, bias, R)
        ctx.task_signal = task_signal
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, R = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.task_signal is None:
            raise RuntimeError("DFA requires an explicit projected task signal.")

        task_signal = ctx.task_signal.to(grad_output.dtype)
        R = R.to(grad_output.dtype)

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(task_signal, R)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(task_signal.transpose(-1, -2), input.to(grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None


class LinearDFA(nn.Module):
    """Linear layer implementing direct feedback alignment (DFA).

    `B` projects the network-level error to this layer's output dimension.
    `R` is the local feedback matrix used for the input gradient.
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

        # B: project the network-level error to this layer's output dimension.
        self.register_buffer("B", torch.empty(self.num_classes, out_features))
        # R: local feedback for grad_input.
        self.register_buffer("R", torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.R, a=math.sqrt(5))
        nn.init.uniform_(self.B, -0.01, 0.01)
        nn.init.uniform_(self.R, -0.01, 0.01)

    def forward(self, input, task_signal=None):
        if task_signal is None:
            return F.linear(input, self.weight, self.bias)

        if task_signal.shape[-1] != self.out_features:
            task_signal = task_signal @ self.B.to(task_signal.device, task_signal.dtype)

        return LinearDFAFunction.apply(input, self.weight, self.bias, self.R, task_signal)
