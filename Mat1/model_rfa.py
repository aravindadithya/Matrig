import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearRFAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, B):
        ctx.save_for_backward(input, weight, bias, B)
        output = input.mm(weight.t())
        if bias:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, B = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # The essence of RFA: use B instead of weight for backward pass of input
            grad_input = grad_output.mm(B.to(grad_output.dtype))
        
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input.to(grad_output.dtype))
            
        if bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None

class LinearRFA(nn.Module):
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

class Nonlinearity(nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        return F.relu(x)
        #return F.tanh(x)

class Net(nn.Module):
    def __init__(self, dim, num_classes=2):
        super(Net, self).__init__()
        bias = False
        k = 1024
        self.dim = dim
        self.width = k

        self.features = nn.Sequential(
            LinearRFA(dim, k, bias=bias),
            Nonlinearity(),
            #LinearRFA(k, k, bias=bias),
            #Nonlinearity(),
        )

        self.classifier = nn.Sequential(           
            LinearRFA(k, num_classes, bias=bias)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x