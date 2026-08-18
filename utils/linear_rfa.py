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
        """
        Forward pass: standard linear transformation y = Wx + b
        Saves input, weight, bias, and feedback matrix B for backward pass.
        """
        ctx.save_for_backward(input, weight, bias, B)
        return F.linear(input, weight, bias)

    @staticmethod
    # @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        """
        Backward pass for RFA layer.
        
        Standard linear layer backprop: grad_input = grad_output @ W^T
        RFA modification: grad_input = grad_output @ B (uses fixed feedback matrix B instead)
        
        grad_weight computation is unchanged from standard backprop.
        
        Dimensions:
            - grad_output: (..., out_features)
            - B: (out_features, in_features)
            - input: (batch, in_features) or just (in_features,)
            - grad_input: (..., in_features)
            - grad_weight: (out_features, in_features)
        """
        
        input, weight, bias, B = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_input = grad_output @ B (instead of grad_output @ W^T)
            grad_input = torch.matmul(grad_output, B.to(grad_output.dtype))

        if ctx.needs_input_grad[1]:
            # Standard weight gradient: outer product of grad_output and input
            grad_weight = torch.matmul(grad_output.transpose(-1, -2), input.to(grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None


class LinearRFA(nn.Module):
    """Linear layer with Random Feedback Alignment for the backward pass.
    
    In RFA, the forward pass is standard: y = Wx + b
    In the backward pass, gradients use a fixed random feedback matrix B instead of W^T:
      grad_input = grad_output @ B (instead of grad_output @ W^T)
      grad_weight = grad_output.T @ input (same as standard backprop)
    
    Note: B is left uninitialized (torch.empty) here. All random initialization happens
    in model_rfa._initialize_weights() via kaiming_uniform_ to maintain RNG flow control.
    """
    
    def __init__(self, in_features, out_features, bias=False):
        super(LinearRFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Fixed random feedback matrix (initialized in model via kaiming_uniform_)
        # Left uninitialized here to centralize all RNG calls in one place
        self.register_buffer('B', torch.empty(out_features, in_features))

    def forward(self, input):
        return LinearRFAFunction.apply(input, self.weight, self.bias, self.B)
