import os
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import hickle as hkl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mat_gen import get_data_loaders
from utils.layers.linear_rfa import LinearRFA
from utils.layers.linear_dfa import LinearDFA

import model
import model_dfa
import model_rfa



def get_loaders(batch_size=128, seed=10000):
   
    config_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(config_dir, 'data', 'custom_dataset')

    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        seed=seed
    )

    return train_loader, val_loader, test_loader

def network_weight_product(net):
    """Compute the full end-to-end matrix W1:N for all linear layers in the network."""
    linear_weights = [
        m.weight.detach()
        for m in net.modules()
        if isinstance(m, (torch.nn.Linear, LinearRFA, LinearDFA))
    ]

    if not linear_weights:
        raise ValueError("Network has no linear layers.")

    W = linear_weights[0]
    for Wi in linear_weights[1:]:
        W = Wi @ W
    return W

def sigma_min(matrix):
    """Return the min{d0, dN}-th largest singular value of matrix."""
    if matrix.dim() != 2:
        raise ValueError("Expected a 2D matrix")

    s = torch.linalg.svdvals(matrix)
    k = min(matrix.shape)
    return s[k - 1]

def deficiency_margin_for_network(net, target):
    """Compute c = sigma_min(T) - ||W1:N - T||_F for the full network."""
    W_end_to_end = network_weight_product(net)

    if W_end_to_end.shape != target.shape:
        print(
            "Skipping deficiency margin due to shape mismatch: "
            f"W1:N={tuple(W_end_to_end.shape)}, target={tuple(target.shape)}"
        )
        return None

    sigma_T = sigma_min(target)
    fro_error = torch.linalg.norm(W_end_to_end - target, ord='fro')
    c = sigma_T - fro_error

    print(f"Full network W1:N shape: {tuple(W_end_to_end.shape)}")
    print(f"sigma_min(T) = {sigma_T.item():.8f}")
    print(f"||W1:N||_F = {torch.linalg.norm(W_end_to_end, ord='fro').item():.8f}")
    print(f"||W1:N - T||_F = {fro_error.item():.8f}")
    print(f"deficiency margin c = {c.item():.8f}")
    return c.item()



def get_untrained_net(
    hidden_layers=None,
    SEED=10000,
    mode="rfa",
    init_method="arora_balanced",
    init_gain=1.0,
    learning_rate=0.01,
    c=3/4,
):
    input_dim = 784
    output_dim = 784

    # Create network with consistent seed
    if mode == "RFA":
        net = model_rfa.Net(
            input_dim,
            num_classes=output_dim,
            hidden_layers=hidden_layers,
            seed=SEED,
            init_method=init_method,
            init_gain=init_gain,
            learning_rate=learning_rate,
            c=c,
        )
    elif mode == "DFA":
        net = model_dfa.Net(
            input_dim,
            num_classes=output_dim,
            hidden_layers=hidden_layers,
            seed=SEED,
            init_method=init_method,
            init_gain=init_gain,
            learning_rate=learning_rate,
            c=c,
        )
    else:
        net = model.Net(
            input_dim,
            num_classes=output_dim,
            hidden_layers=hidden_layers,
            seed=SEED,
            init_method=init_method,
            init_gain=init_gain,
            learning_rate=learning_rate,
            c=c,
        )
    return net


def get_config(
    hidden_layers,
    run_id="1",
    project="4_layer_fc_deltaarora",
    entity="ICLR_2027",
    run_name="FC",
    mode="rfa",  
    init_method="arora_balanced",
    init_gain=1.0,
    SEED=1000,
):

    depth = len(hidden_layers)
    width = hidden_layers[0]
    run_name = f"{mode}_{SEED}_{depth}_{width}"


    # Pass seed to loaders for reproducible data splitting and shuffling
    trainloader, valloader, testloader = get_loaders(seed=SEED)

    # Exhaustive seed reset to ensure global state is identical before data loading.
    # This covers cases where library-level initialization (like WandB or ONNX) 
    # might have touched various random generators.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    #scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=int(epochs/3)+1, decay=0.8)
    scheduler = None
    lfn = nn.MSELoss()
    config_dir = os.path.dirname(os.path.abspath(__file__))
    target_matrix_path = os.path.join(config_dir, 'Identity_matrix_784x784.hkl')
    target_matrix = torch.tensor(hkl.load(target_matrix_path), dtype=torch.float32)
    learning_rate = 0.01
    c= 3/4

    net = get_untrained_net(
        hidden_layers=hidden_layers,
        SEED=SEED,
        mode=mode,
        init_method=init_method,
        init_gain=init_gain,
        learning_rate=learning_rate,
        c=c,
    )

    deficiency_margin_for_network(net, target_matrix)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    

    config = {
        "project": f"{project}",
        "entity": entity,
        "run_name": run_name,
        "run_id": run_id,
        "seed": SEED,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "c": c,
        "optimizer_name": type(optimizer).__name__,
        "loss_function_name": type(lfn).__name__,
        "model_architecture": type(net).__name__,
        "model_structure": str(net),
        "num_parameters": sum(p.numel() for p in net.parameters()),
        "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
        "scheduler_name": type(scheduler).__name__ if scheduler else "None",
        "task_type": "regression",
        "hidden_layers": hidden_layers,
        "net": net,
        "train_loader": trainloader,
        "val_loader": valloader,
        "test_loader": testloader,
        "optimizer": optimizer,
        "lfn": lfn,
        "scheduler": scheduler,
        "init_method": init_method,
        "init_gain": init_gain,
        "target_matrix": target_matrix,
        "target_matrix_path": target_matrix_path,
    }
    return config
