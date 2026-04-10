import torch
import torch.nn as nn
import math

class LinearRFAFunction(torch.autograd.Function):
    """Custom autograd function for Random Feedback Alignment layer."""

    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, B = inputs
        ctx.save_for_backward(input, weight, bias, B)
        return None

    @staticmethod
    def forward(input, weight, bias, B):
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
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
    
    def __init__(self, in_features, out_features, bias=True):
        super(LinearRFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Fixed random feedback matrix
        self.register_buffer('B', torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        #nn.init.xavier_uniform_(self.weight)
        #nn.init.xavier_uniform_(self.B)
        
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        

    def forward(self, input):
        return LinearRFAFunction.apply(input, self.weight, self.bias, self.B)
