import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.initializer import initialize_linear_layer, arora_balanced_initialization


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
        Fully connected neural network with configurable hidden layers.

        Args:
            dim: Input dimension
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes (default: None means single hidden layer of 1024)
                          Example: [1024, 512, 256] creates 3 hidden layers
            bias: Whether to use bias in linear layers (default: False)
            seed: Random seed for weight initialization (default: None)
            init_method: Weight initialization method
                         (kaiming, he, glorot, arora_balanced, orthogonal)
            init_gain: Gain/scaling factor for initialization
        """
        super(Net, self).__init__()

        self.seed = seed

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
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes, bias=bias)
        self._initialize_weights()

    def _initialize_weights(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        print(linear_layers)
        if not linear_layers:
            return

        if self.init_method == "arora_balanced":
            arora_balanced_initialization(
                linear_layers,
                distribution="uniform",
                mean=0.0,
                std=0.01 ** len(linear_layers),
                bias_value=0.0,
            )
        else:
            for layer in linear_layers:
                initialize_linear_layer(
                    layer,
                    method=self.init_method,
                    gain=self.init_gain,
                    bias_value=0.0,
                )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
