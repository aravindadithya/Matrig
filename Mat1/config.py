import os
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from torch.utils.data import DataLoader
import model
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mat_gen import get_data_loaders


def get_loaders(batch_size=1024, seed=10000):
    """
    Load custom dataset instead of MNIST.
    Dataset path is at ../data/custom_dataset relative to this config file.
    """
    config_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(config_dir, '..', 'data', 'custom_dataset')
    
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        seed=seed
    )
    
    return train_loader, val_loader, test_loader


def get_untrained_net(choice, hidden_layers=None, seed=1000):
    output_dim = 28 * 28
    net = model.Net(28*28, num_classes=output_dim, hidden_layers=hidden_layers, seed=seed)
    return net

def get_config(
    choice,
    run_id="1",
    project="Mat1",
    entity="Matrig100",
    run_name="FC",
    hidden_layers=None,
):
    SEED = 9763
    # Set seeds for reproducibility across all libraries BEFORE creating model
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
    random.seed(SEED)
    np.random.seed(SEED)

    # Create network with consistent seed
    net = get_untrained_net(choice, hidden_layers=hidden_layers, seed=SEED)
    depth = (len(hidden_layers) if hidden_layers is not None else 1) + 1
    run_name = f"FC_{SEED}_{depth}"
    # Pass seed to loaders for reproducible data splitting and shuffling
    trainloader, valloader, testloader = get_loaders(seed=SEED)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
    #scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=int(epochs/3)+1, decay=0.8)
    scheduler = None
    lfn = nn.MSELoss()

    config = {
        "project": f"{project}",
        "entity": entity,
        "run_name": run_name,
        "run_id": run_id,
        "seed": SEED,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "optimizer_name": type(optimizer).__name__,
        "loss_function_name": type(lfn).__name__,
        "model_architecture": type(net).__name__,
        "model_structure": str(net),
        "num_parameters": sum(p.numel() for p in net.parameters()),
        "weight_decay": optimizer.param_groups[0].get('weight_decay', 0),
        "scheduler_name": type(scheduler).__name__ if scheduler else "None",
        "task_type": "regression",
        "output_dim": 28 * 28,
        "max_images": 32,
        "rotate_inputs": False,
        # Use 'channels_last' for potential performance boost with 4D tensors (e.g. CNNs)
        "memory_format": "channels_last",
        "net": net,
        "train_loader": trainloader,
        "val_loader": valloader,
        "test_loader": testloader,
        "optimizer": optimizer,
        "lfn": lfn,
        "scheduler": scheduler
    }
    return config