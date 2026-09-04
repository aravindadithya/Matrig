import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDFAFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, weight, bias, B, is_classifier_layer, global_error):
        ctx.save_for_backward(input, weight, bias, B)
        ctx.is_classifier_layer = is_classifier_layer
        ctx.global_error = global_error
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, B = ctx.saved_tensors
        e = ctx.global_error
        if e is None:
            raise RuntimeError("LinearDFA requires global_error to be passed in forward.")

        # Cast for dtype consistency
        e = e.to(weight.dtype) #[bs, num_classes]
        input = input.to(weight.dtype) #[bs, in_features]
        if ctx.is_classifier_layer:
            # Classifier uses the raw global error directly.
            local_error = e
        else:
            B = B.to(weight.dtype) #[num_classes, out_features]
            local_error = torch.matmul(e, B)* grad_output # [bs, out_features]

        grad_input = grad_weight = grad_bias = None

        # Return all ones vector so that Pytorchs non-linarities be chained.
        if ctx.needs_input_grad[0]:
            grad_input = torch.ones_like(input)

        # DFA weight update
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(local_error.transpose(-1, -2), input) #[out_features, in_features]

        # DFA bias update
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = local_error.sum(dim=0)

        # No gradient for B
        return grad_input, grad_weight, grad_bias, None, None, None


class LinearDFA(nn.Module):
    """Linear layer using Direct Feedback Alignment.

    Args:
        in_features: input size
        out_features: output size
        num_classes: dimension of the global error signal (usually output size).
                     Defaults to out_features.
        bias: if True, adds a learnable bias.
        is_classifier_layer: if True, initializes feedback matrix `B` as identity.

    This layer expects `grad_output` to be either:
        - a plain tensor `e` (when no activation follows), or
        - a tuple `(e, deriv)` (when followed by a DFA activation).
    """

    def __init__(self, in_features, out_features, num_classes=None, bias=True, is_classifier_layer=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes if num_classes is not None else out_features
        self.is_classifier_layer = is_classifier_layer

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Fixed random feedback matrix
        self.register_buffer("B", torch.empty(self.num_classes, out_features))



    def forward(self, input, global_error=None):
        if global_error is None:
            return F.linear(input, self.weight, self.bias)
        return LinearDFAFunction.apply(
            input,
            self.weight,
            self.bias,
            self.B,
            self.is_classifier_layer,
            global_error,
        )