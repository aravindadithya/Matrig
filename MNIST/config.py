import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from MNIST import model, model_rfa, model_dfa


def one_hot_collate(batch):
    """Convert integer class labels to one-hot vectors for MSE/BCE-style training."""
    images = []
    targets = []
    for image, label in batch:
        images.append(image)
        targets.append(torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float())
    return torch.stack(images), torch.stack(targets)


def get_loaders(batch_size=128, seed=123, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.flatten()),
    ])

    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator, collate_fn=one_hot_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=one_hot_collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=one_hot_collate)

    return train_loader, val_loader, test_loader


def get_untrained_net(
    activation,
    hidden_layers=None,
    SEED=10000,
    mode="RFA",
    init_method="arora_balanced",
    init_gain=1.0,
):
    input_dim = 784
    output_dim = 10

    # Create network with consistent seed
    if mode == "RFA":
        net = model_rfa.Net(
            input_dim,
            num_classes=output_dim,
            activation=activation,
            hidden_layers=hidden_layers,
            seed=SEED,
            init_method=init_method,
            init_gain=init_gain,
        )
    elif mode == "DFA":
        net = model_dfa.Net(
            input_dim,
            num_classes=output_dim,
            activation=activation,
            hidden_layers=hidden_layers,
            seed=SEED,
            init_method=init_method,
            init_gain=init_gain,
        )
    else:
        net = model.Net(
            input_dim,
            num_classes=output_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            seed=SEED,
            init_method=init_method,
            init_gain=init_gain,
        )
    return net


def get_config(
    activation,
    hidden_layers,
    run_id="1",
    project="MNIST_fc_balancedness",
    entity="ICLR_2027",
    run_name="FC",
    mode="RFA",  
    init_method="arora_balanced",
    init_gain=1.0,
    SEED=1000,
):

    run_name = f"{mode}_{SEED}_{activation.__class__.__name__}_{len(hidden_layers)}_{hidden_layers[0]}"

    trainloader, valloader, testloader = get_loaders(batch_size= 100, seed=SEED)


    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    net = get_untrained_net(
               hidden_layers=hidden_layers,
               activation=activation,
               SEED=SEED,
               mode=mode,
               init_method=init_method,
               init_gain=init_gain
           )
        

    learning_rate = 0.01
    
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    scheduler = None
    #lfn = nn.CrossEntropyLoss()
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
            "hidden_layers": hidden_layers,
            "net": net,
            "train_loader": trainloader,
            "val_loader": valloader,
            "test_loader": testloader,
            "optimizer": optimizer,
            "lfn": lfn,
            "scheduler": scheduler,
            "init_method": init_method,
            "activation": activation,
        }
    return config



