import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.initializer import (
    initialize_linear_layer,
    arora_balanced_initialization,
    bp_adversary_initialization,
)
from utils.layers.linear_dfa import LinearDFA


class Net(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        activation,
        hidden_layers=None,
        bias=False,
        seed=None,
        init_method="arora_balanced",
        init_gain=1.0,
        learning_rate=0.01,
        c=0.5,
    ):
        super(Net, self).__init__()

        self.seed = seed
        self.dim = dim
        self.num_classes = num_classes
        self.bias = bias
        self.init_method = init_method.lower()
        self.init_gain = init_gain
        self.learning_rate = learning_rate
        self.c = c
        self.activation = activation

        if hidden_layers is None:
            hidden_layers = [1024]

        self.hidden_layers = hidden_layers

        layers = []
        prev_dim = dim
        for hidden_dim in hidden_layers:
            layers.append(LinearDFA(prev_dim, hidden_dim, num_classes=num_classes, bias=bias))
            layers.append(self.activation)
            prev_dim = hidden_dim

        self.features = nn.ModuleList(layers)
        self.classifier = LinearDFA(
            prev_dim,
            num_classes,
            num_classes=num_classes,
            bias=bias,
            is_classifier_layer=True,
        )
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
                std= 0.30,
                bias_value=0.0,
            )
        elif self.init_method == "bp_adversary":
            bp_adversary_initialization(
                linear_layers,
                learning_rate=self.learning_rate,
                c=self.c,
                bias_value=0.0,
            )
        else:
            for layer in linear_layers:
                initialize_linear_layer(
                    layer,
                    method=self.init_method,
                    gain=self.init_gain,
                    bias_value=0.0,
                    learning_rate=self.learning_rate,
                    c=self.c,
                )

        for layer in linear_layers:
            # Keep classifier feedback unconstrained; random feedback elsewhere.
            if layer.is_classifier_layer:
                layer.B.fill_(1.0)
            else:
                # nn.init.kaiming_uniform_(layer.B, a=math.sqrt(5))
                # nn.init.kaiming_uniform_(layer.R, a=math.sqrt(5))
                nn.init.uniform_(layer.B, -0.1, 0.1)

    def forward(self, x, global_error=None):
        for layer in self.features:
            if isinstance(layer, LinearDFA):
                x = layer(x, global_error=global_error)
            else:
                x = layer(x)
        x = self.classifier(x, global_error=global_error)
        return x
