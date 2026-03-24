import os
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import RFA_MNIST.model1.model as model, RFA_MNIST.model1.model_rfa as model_rfa

workspaces_path= os.getenv('PYTHONPATH')
print(f"Current Path: {workspaces_path}")

def get_loaders(batch_size=1024, seed=10000):
    path = workspaces_path + '/MNIST/data'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x)) 
    ])

    train_set = torchvision.datasets.MNIST(root=path, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root=path, train=False, transform=transform, download=True)
    
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    # Use a generator for reproducible random splitting
    g = torch.Generator()
    g.manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(train_set, [train_size, val_size], generator=g)
    
    # Pass the generator to the train_loader for reproducible shuffling
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, generator=g)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader

def get_untrained_net(choice):
    if choice == 0:
        net = model.Net(28*28, num_classes=10)
    else:
        net = model_rfa.Net(28*28, num_classes=10)
   
    return net

def get_config(choice, run_id="1", project="MNIST_RFA", entity="RFA100", run_name="FC_Model1"):
    SEED = 1000
    # Set seeds for reproducibility across all libraries
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
    random.seed(SEED)
    np.random.seed(SEED)

    net = get_untrained_net(choice)
    # Pass seed to loaders for reproducible data splitting and shuffling
    trainloader, valloader, testloader = get_loaders(seed=SEED)

    if choice == 0:
        run_name = "FC_Model1_Tanh"
    else:
        run_name = "FC_Model1_Tanh_RFA"

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    epochs = 10
    #scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=int(epochs/3)+1, decay=0.8)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    lfn = nn.CrossEntropyLoss()

    config = {
        "project": f"{project}_seed_{SEED}",
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
        "num_classes": 10,
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