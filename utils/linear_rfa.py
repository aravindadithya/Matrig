import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.amp import custom_fwd, custom_bwd
import math

class LinearRFAFunction(torch.autograd.Function):
    """Custom autograd function for Random Feedback Alignment layer."""

    generate_vmap_rule = True

    @staticmethod
    # @custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, bias, B):
        ctx.save_for_backward(input, weight, bias, B)
        return F.linear(input, weight, bias)

    @staticmethod
    # @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        
        input, weight, bias, B = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, B.to(grad_output.dtype))

        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.transpose(-1, -2), input.to(grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None


class LinearRFA(nn.Module):
    """Linear layer with Random Feedback Alignment for the backward pass."""
    
    def __init__(self, in_features, out_features, bias=False):
        super(LinearRFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Fixed random feedback matrix
        self.register_buffer('B', torch.empty(out_features, in_features))

    def forward(self, input):
        return LinearRFAFunction.apply(input, self.weight, self.bias, self.B)
