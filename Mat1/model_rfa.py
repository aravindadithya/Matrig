import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.initializer import initialize_linear_layer, arora_balanced_initialization
from utils.linear_rfa import LinearRFA

class Net(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        hidden_layers=None,
        bias=False,
        seed=None,
        init_method="arora_balanced",
        init_gain=1.0,
    ):
        """
        Fully connected neural network with random feedback alignment and configurable hidden layers.

        Args:
            dim: Input dimension
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes (default: None means single hidden layer of 1024)
                          Example: [1024, 512, 256] creates 3 hidden layers
            bias: Whether to use bias in linear layers (default: False)
            seed: Random seed for weight initialization (default: None)
            init_method: Weight initialization method for forward weights
                         (kaiming, he, glorot, arora_balanced, orthogonal)
            init_gain: Gain/scaling factor for initialization
        """
        super(Net, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.dim = dim
        self.num_classes = num_classes
        self.bias = bias
        self.init_method = init_method.lower()
        self.init_gain = init_gain

        if hidden_layers is None:
            hidden_layers = [1024]

        self.hidden_layers = hidden_layers

        layers = []
        prev_dim = dim

        for hidden_dim in hidden_layers:
            layers.append(LinearRFA(prev_dim, hidden_dim, bias=bias))
            prev_dim = hidden_dim


        self.features = nn.Sequential(*layers)
        self.classifier = LinearRFA(prev_dim, num_classes, bias=bias)
        self._initialize_weights()

    def _initialize_weights(self):
        linear_layers = [m for m in self.modules() if isinstance(m, LinearRFA)]
        print(linear_layers)
        if not linear_layers:
            return

        arora_balanced_initialization(
                linear_layers,
                distribution="normal",
                mean=0.0,
                std=self.init_gain,
                bias_value=0.0,
            )
        return
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x