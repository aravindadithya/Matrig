import math

import torch
import torch.nn as nn
import torch.nn.init as init


SUPPORTED_INITIALIZERS = (
    "kaiming",
    "he",
    "glorot",
    "arora_balanced",
    "bp_adversary",
    "orthogonal",
    "zeros",
)


def arora_balanced_initialization(
    layers,  # list of nn.Linear or LinearRFA layers
    distribution: str = "normal",
    mean: float = 0.0,
    std: float = 1.0,
    bias_value: float = 0.0,
) -> None:
    """Apply Arora balanced initialization to a sequence of linear layers.

    This implements balanced init for W1..WN from the specification:
      A = U Sigma V^T, W1 = Sigma^(1/N) V^T (padded),
      Wk = Sigma^(1/N) (padded) for k=2..N-1,
      WN = U Sigma^(1/N) (padded).

    Assumes layers are in the order input->hidden1, hidden1->hidden2, ..., hiddenN-1->output.

    The base matrix A is drawn from the specified distribution. For gaussian (normal),
    mean and std can be set to match paper perturbation settings.
    
    Args:
        layers: List of nn.Linear or LinearRFA layers with .weight, .bias, .in_features, .out_features
        distribution: "normal" or "uniform"
        mean: Mean for normal distribution
        std: Standard deviation for normal distribution
        bias_value: Value to initialize bias to
    """
    if len(layers) == 0:
        return

    d0 = layers[0].in_features
    dN = layers[-1].out_features
    inner_dims = [layer.out_features for layer in layers[:-1]]
    r = min(d0, dN)

    device = layers[0].weight.device
    dtype = layers[0].weight.dtype
    N = len(layers)

    if distribution == "normal":
        A = torch.randn(dN, d0, device=device, dtype=dtype) * std + mean
    elif distribution == "uniform":
        A = torch.empty(dN, d0, device=device, dtype=dtype)
        init.uniform_(A, mean - std, mean + std)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}' for Arora balanced init")

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # A = U @ diag(S) @ Vh

    root = S.pow(1.0 / N)
    # diag_root: r x r
    diag_root = torch.diag(root)

    # w1_small is r x d0
    w1_small = diag_root @ Vh
    # wn_small is dN x r
    wn_small = U @ diag_root

    # assign layer weights
    for idx, layer in enumerate(layers):
        layer.weight.data.zero_()
        if idx == 0:
            # first layer: d1 x d0
            h = layer.out_features
            layer.weight.data[:r, :d0] = w1_small[: min(r, h), :]
        elif idx == len(layers) - 1:
            # last layer: dN x d_{N-1}
            in_feat = layer.in_features
            layer.weight.data[:dN, :r] = wn_small[:, : min(r, in_feat)]
        else:
            # middle layer: d_k x d_{k-1}
            h_out = layer.out_features
            h_in = layer.in_features
            layer.weight.data[: min(r, h_out), : min(r, h_in)] = diag_root[: min(r, h_out), : min(r, h_in)]

        if layer.bias is not None:
            layer.bias.data.fill_(bias_value)


def bp_adversary_initialization(
    layers,
    learning_rate: float = 0.01,
    c: float = 0.5,
    bias_value: float = 0.0,
) -> None:
    """Initialize each layer as an almost-diagonal matrix with a single adversarial entry.

    For a network with N layers, the first diagonal entry of layer j is set to
      A * c^(1/N), 1 <= j <= N/2
      c^(1/N) / A, N/2 < j <= N
    where
      A = max{sqrt(eta * N), 2/(eta * (1-c) * c^((N-1)/N)), 2000, 20/eta,
              (20 * (10^(2N-1) / eta^(2N)))^(1/(2N-2)) }
    and eta is the learning rate.
    """
    if len(layers) == 0:
        return

    eta = float(learning_rate)
    N = len(layers)
    scale = c ** (1.0 / N)
    A = max(
        math.sqrt(eta * N),
        2.0 / (eta * (1.0 - c) * (c ** ((N - 1.0) / N))),
        2000.0,
        20.0 / eta,
        (20.0 * ((10.0 ** (2 * N - 1)) / (eta ** (2 * N)))) ** (1.0 / (2 * N - 2)),
    )

    A = float(A)
    if not math.isfinite(A):
        print("EXPLODDDEEEEEE")

    for idx, layer in enumerate(layers):
        rows, cols = layer.weight.shape
        if rows > 0 and cols > 0:
            with torch.no_grad():
                layer.weight.zero_()
                k = min(rows, cols)
                layer.weight[:k, :k].copy_(torch.eye(k, device=layer.weight.device, dtype=layer.weight.dtype))
                value = A * scale if (idx + 1) <= (N / 2.0) else scale / A
                value = float(value)
                if not math.isfinite(value):
                    print("EXPLODDDEEEEEE2")
                layer.weight[0, 0] = value

        if layer.bias is not None:
            with torch.no_grad():
                layer.bias.fill_(bias_value)


def initialize_linear_layer(
    layer: nn.Linear,
    method: str,
    gain: float = 1.0,
    bias_value: float = 0.0,
    nonlinearity: str = "relu",
    learning_rate: float = 0.01,
    c: float = 0.5,
) -> None:
    """Initialize a single Linear layer using the selected method."""
    method = method.lower()

    if method == "kaiming":
        init.kaiming_uniform_(layer.weight, a=0.0, mode="fan_in", nonlinearity=nonlinearity)
    elif method == "he":
        init.kaiming_normal_(layer.weight, a=0.0, mode="fan_in", nonlinearity=nonlinearity)
    elif method in ("glorot", "xavier"):
        init.xavier_uniform_(layer.weight, gain=gain)
    elif method == "orthogonal":
        init.orthogonal_(layer.weight, gain=gain)
    elif method == "zeros":
        init.zeros_(layer.weight)
    elif method == "bp_adversary":
        bp_adversary_initialization([layer], learning_rate=learning_rate, c=c, bias_value=bias_value)
    else:
        raise ValueError(
            f"Unsupported initialization method '{method}'. "
            f"Use one of: {', '.join(SUPPORTED_INITIALIZERS)}"
        )

    if layer.bias is not None and method != "bp_adversary":
        init.constant_(layer.bias, bias_value)
