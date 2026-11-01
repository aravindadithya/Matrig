import os
import random
import sys

# BioTorch recommends this setting for deterministic CUDA matrix operations.
# It must be set before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from biotorch.layers.fa import Linear as BioTorchLinear

from utils.initializer import (
    initialize_linear_layer,
    arora_balanced_initialization,
    bp_adversary_initialization,
)


def seed_everything(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch for repeatable model construction."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


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
        feedback_range=0.05,
        arora_std=0.30,
        deterministic=True,
    ):
        """Fully connected RFA network using BioTorch FA linear layers."""
        super().__init__()

        self.seed = seed
        self.deterministic = deterministic
        self.activation = activation
        self.dim = dim
        self.num_classes = num_classes
        self.bias = bias
        self.init_method = init_method.lower()
        self.init_gain = init_gain
        self.learning_rate = learning_rate
        self.c = c
        self.feedback_range = feedback_range
        self.arora_std = arora_std

        if hidden_layers is None:
            hidden_layers = [1024]
        self.hidden_layers = hidden_layers

        layers = []
        prev_dim = dim
        for hidden_dim in hidden_layers:
            layers.append(BioTorchLinear(prev_dim, hidden_dim, bias=bias))
            layers.append(self.activation)
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = BioTorchLinear(prev_dim, num_classes, bias=bias)
        self._initialize_weights()

    def _initialize_weights(self):
        seed_everything(self.seed, deterministic=self.deterministic)
        linear_layers = [m for m in self.modules() if isinstance(m, BioTorchLinear)]
        if not linear_layers:
            return

        if self.init_method == "arora_balanced":
            arora_balanced_initialization(
                linear_layers,
                distribution="uniform",
                mean=0.0,
                std=self.arora_std,
                bias_value=0.0,
                shuffle=False,
            )
        elif self.init_method == "bp_adversary":
            bp_adversary_initialization(
                linear_layers,
                learning_rate=self.learning_rate,
                c=self.c,
                bias_value=0.0,
            )
        elif self.init_method in (
            "kaiming",
            "he",
            "glorot",
            "xavier",
            "orthogonal",
            "zeros",
        ):
            for layer in linear_layers:
                initialize_linear_layer(
                    layer,
                    method=self.init_method,
                    gain=self.init_gain,
                    bias_value=0.0,
                    nonlinearity=self.activation.__class__.__name__.lower(),
                    learning_rate=self.learning_rate,
                    c=self.c,
                )
        else:
            for layer in linear_layers:
                nn.init.uniform_(layer.weight, -0.025, 0.025)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

        # BioTorch calls the fixed feedback matrix weight_backward; initialize
        # it after forward weights to preserve the experiment's RNG order.
        for layer in linear_layers:
            nn.init.uniform_(layer.weight_backward, -self.feedback_range, self.feedback_range)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)