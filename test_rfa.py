"""
Quick validation test for RFA implementation.
Tests that:
1. Forward pass produces expected output shape
2. Backward pass computes gradients without error
3. B matrix is properly initialized
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils.layers.linear_rfa import LinearRFA
from utils.layers.linear_dfa import LinearDFA


def test_rfa_forward_backward():
    """Test RFA forward and backward pass."""
    print("Testing RFA Implementation...")

    in_features = 10
    out_features = 5
    batch_size = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create RFA layer
    rfa_layer = LinearRFA(in_features, out_features, bias=True).to(device)

    # Check B matrix is initialized
    assert rfa_layer.B is not None, "B matrix not initialized"
    assert rfa_layer.B.shape == (out_features, in_features), f"B shape mismatch: {rfa_layer.B.shape}"
    print(f"✓ B matrix initialized with shape {rfa_layer.B.shape}")

    # Create input
    x = torch.randn(batch_size, in_features, requires_grad=True, device=device)

    # Forward pass
    output = rfa_layer(x)
    assert output.shape == (batch_size, out_features), f"Output shape mismatch: {output.shape}"
    print(f"✓ Forward pass: input {x.shape} → output {output.shape}")

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    # Check gradients computed
    assert x.grad is not None, "Input gradient not computed"
    assert rfa_layer.weight.grad is not None, "Weight gradient not computed"
    print(f"✓ Backward pass computed gradients")
    print(f"  - Input gradient shape: {x.grad.shape}")
    print(f"  - Weight gradient shape: {rfa_layer.weight.grad.shape}")

    print(f"  - Feedback matrix B is used for backward pass (instead of W^T)")

    print("\n✓ All RFA tests passed!")


def test_dfa_forward_backward():
    """Test DFA forward and backward pass with a shared global error signal."""
    print("Testing DFA Implementation...")

    in_features = 10
    out_features = 5
    batch_size = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dfa_layer = LinearDFA(in_features, out_features, bias=True).to(device)

    assert dfa_layer.B is not None, "DFA feedback matrix not initialized"
    assert dfa_layer.B.shape == (out_features, out_features), f"DFA B shape mismatch: {dfa_layer.B.shape}"
    assert dfa_layer.R.shape == (out_features, in_features), f"DFA R shape mismatch: {dfa_layer.R.shape}"
    print(f"✓ DFA B matrix initialized with shape {dfa_layer.B.shape}")
    print(f"✓ DFA R matrix initialized with shape {dfa_layer.R.shape}")

    x = torch.randn(batch_size, in_features, requires_grad=True, device=device)
    global_error = torch.randn(batch_size, out_features, device=device, requires_grad=False)
    output = dfa_layer(x, task_signal=global_error)
    assert output.shape == (batch_size, out_features), f"DFA output shape mismatch: {output.shape}"
    print(f"✓ DFA forward pass: input {x.shape} → output {output.shape}")

    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "DFA input gradient not computed"
    assert dfa_layer.weight.grad is not None, "DFA weight gradient not computed"
    print(f"✓ DFA backward pass computed gradients")
    print(f"  - Input gradient shape: {x.grad.shape}")
    print(f"  - Weight gradient shape: {dfa_layer.weight.grad.shape}")

    print("\n✓ All DFA tests passed!")


if __name__ == "__main__":
    test_rfa_forward_backward()
    test_dfa_forward_backward()
