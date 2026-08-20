import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.initializer import initialize_linear_layer, arora_balanced_initialization
from utils.layers.linear_dfa import LinearDFA


class DFASequential(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, task_signal=None):
        if task_signal is None:
            for layer in self.layers:
                x = layer(x)
            return x

        for layer in self.layers:
            layer_task = task_signal @ layer.B.to(task_signal.device, task_signal.dtype)
            x = layer(x, task_signal=layer_task)
        return x


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
            layers.append(LinearDFA(prev_dim, hidden_dim, num_classes=num_classes, bias=bias))
            prev_dim = hidden_dim

        self.features = DFASequential(layers)
        self.classifier = LinearDFA(prev_dim, num_classes, num_classes=num_classes, bias=bias)
        self.classifier.B.data.copy_(torch.eye(num_classes, device=self.classifier.B.device, dtype=self.classifier.B.dtype))
        self._initialize_weights()

    def _initialize_weights(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        linear_layers = [m for m in self.modules() if isinstance(m, LinearDFA)]
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

        for layer in linear_layers:
            # nn.init.kaiming_uniform_(layer.B, a=math.sqrt(5))
            # nn.init.kaiming_uniform_(layer.R, a=math.sqrt(5))
            nn.init.uniform_(layer.B, -0.01, 0.01)
            nn.init.uniform_(layer.R, -0.01, 0.01)

    def forward(self, x, task_signal=None):
        if task_signal is None:
            x = self.features(x)
            x = self.classifier(x)
            return x
        x = self.features(x, task_signal=task_signal)
        x = self.classifier(x, task_signal=task_signal)
        return x
