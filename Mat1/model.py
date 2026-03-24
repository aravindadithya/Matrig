import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, dim, num_classes=2, hidden_layers=None, bias=False, seed=None):
        """
        Fully connected neural network with configurable hidden layers.
        
        Args:
            dim: Input dimension
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes (default: None means single hidden layer of 1024)
                          Example: [1024, 512, 256] creates 3 hidden layers
            bias: Whether to use bias in linear layers (default: False)
            seed: Random seed for weight initialization (default: None)
        """
        super(Net, self).__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.dim = dim
        self.num_classes = num_classes
        self.bias = bias
        
        # Default hidden layers if not specified
        if hidden_layers is None:
            hidden_layers = [1024]
        
        self.hidden_layers = hidden_layers
        
        # Build the network: input -> hidden layers -> output
        layers = []
        prev_dim = dim
        
        # Add hidden layers (fully connected with no non-linearity between them)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            prev_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, num_classes, bias=bias))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through all layers (no non-linearity between them)."""
        x = self.network(x)
        return x