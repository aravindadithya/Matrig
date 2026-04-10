import torch
import torch.nn as nn
import torch.nn.init as init


SUPPORTED_INITIALIZERS = (
    "kaiming",
    "he",
    "glorot",
    "arora_balanced",
    "orthogonal",
)


def arora_balanced_initialization(
    layers: list[nn.Linear],
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
    """
    if len(layers) == 0:
        return

    d0 = layers[0].in_features
    dN = layers[-1].out_features
    inner_dims = [layer.out_features for layer in layers[:-1]]
    r = min(d0, dN)

    if len(inner_dims) > 0 and min(inner_dims) < r:
        raise ValueError(
            f"Balanced initialization requires min(hidden dims) >= min(d0,dN). "
            f"Got d0={d0}, dN={dN}, inner={inner_dims}"
        )

    device = layers[0].weight.device
    dtype = layers[0].weight.dtype

    if distribution == "normal":
        A = torch.randn(dN, d0, device=device, dtype=dtype) * std + mean
    elif distribution == "uniform":
        A = (torch.rand(dN, d0, device=device, dtype=dtype) * 2.0 - 1.0) * std + mean
    else:
        raise ValueError(f"Unsupported distribution '{distribution}' for Arora balanced init")

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # A = U @ diag(S) @ Vh

    N = len(layers)
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


def initialize_linear_layer(
    layer: nn.Linear,
    method: str,
    gain: float = 1.0,
    bias_value: float = 0.0,
    nonlinearity: str = "relu",
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
    else:
        raise ValueError(
            f"Unsupported initialization method '{method}'. "
            f"Use one of: {', '.join(SUPPORTED_INITIALIZERS)}"
        )

    if layer.bias is not None:
        init.constant_(layer.bias, bias_value)
