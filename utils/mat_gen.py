"""
Matrix and dataset generation utilities.
Generate random matrices and create custom datasets for training.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import hickle as hkl


def generate_random_matrix(n, mean=0.0, std=1.0, seed=None):
    """
    Generate an n*n matrix with entries randomly sampled from normal distribution.
    
    Args:
        n: Size of the square matrix (n*n)
        mean: Mean of the normal distribution (default: 0.0)
        std: Standard deviation of the normal distribution (default: 1.0)
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        numpy array of shape (n, n)
    """
    if seed is not None:
        np.random.seed(seed)
    
    matrix = np.random.normal(loc=mean, scale=std, size=(n, n))
    return matrix


def save_matrix(matrix, filepath):
    """
    Save a matrix to disk using hickle format.
    
    Args:
        matrix: numpy array to save
        filepath: path where to save the matrix
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    hkl.dump(matrix, filepath)
    print(f"Matrix saved to {filepath}")


def load_matrix(filepath):
    """
    Load a matrix from disk using hickle format.
    
    Args:
        filepath: path to the saved matrix
    
    Returns:
        loaded numpy array
    """
    matrix = hkl.load(filepath)
    return matrix


def generate_dataset(num_train_samples=60000, num_test_samples=10000,
                     input_dim=784, matrix=None, seed=None):
    """
    Generate a regression dataset from random inputs using y = Mx.
    
    Args:
        num_train_samples: Number of training samples (default: 60000)
        num_test_samples: Number of test samples (default: 10000)
        input_dim: Dimension of input features (default: 784)
        matrix: Optional transformation matrix M of shape (input_dim, input_dim)
            If None, a random matrix is generated.
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, matrix) as numpy arrays
    """
    if seed is not None:
        np.random.seed(seed)

    if matrix is None:
        matrix = generate_random_matrix(input_dim, seed=seed).astype(np.float32)
    else:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (input_dim, input_dim):
            raise ValueError(f"Expected matrix shape {(input_dim, input_dim)}, got {matrix.shape}")
    
    # Generate random input vectors
    X_train = np.random.randn(num_train_samples, input_dim).astype(np.float32)
    X_test = np.random.randn(num_test_samples, input_dim).astype(np.float32)

    # Regression targets: y = Mx
    y_train = X_train @ matrix.T
    y_test = X_test @ matrix.T

    return X_train, y_train.astype(np.float32), X_test, y_test.astype(np.float32), matrix


def save_dataset(X_train, y_train, X_test, y_test, dataset_dir):
    """
    Save dataset components to disk using hickle format.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        dataset_dir: Directory where to save the dataset
    """
    os.makedirs(dataset_dir, exist_ok=True)
    
    hkl.dump(X_train, os.path.join(dataset_dir, 'X_train.hkl'))
    hkl.dump(y_train, os.path.join(dataset_dir, 'y_train.hkl'))
    hkl.dump(X_test, os.path.join(dataset_dir, 'X_test.hkl'))
    hkl.dump(y_test, os.path.join(dataset_dir, 'y_test.hkl'))
    
    print(f"Dataset saved to {dataset_dir}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Input dimension: {X_train.shape[1]}")
    print(f"  - Target dimension: {y_train.shape[1] if y_train.ndim > 1 else 1}")


def load_dataset(dataset_dir):
    """
    Load dataset components from disk.
    
    Args:
        dataset_dir: Directory containing the dataset files
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as numpy arrays
    """
    X_train = hkl.load(os.path.join(dataset_dir, 'X_train.hkl'))
    y_train = hkl.load(os.path.join(dataset_dir, 'y_train.hkl'))
    X_test = hkl.load(os.path.join(dataset_dir, 'X_test.hkl'))
    y_test = hkl.load(os.path.join(dataset_dir, 'y_test.hkl'))
    
    return X_train, y_train, X_test, y_test


def get_data_loaders(dataset_dir, batch_size=1024, seed=10000):
    """
    Create PyTorch DataLoaders from saved dataset.
    
    Args:
        dataset_dir: Directory containing the dataset files
        batch_size: Batch size for DataLoaders (default: 1024)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    X_train, y_train, X_test, y_test = load_dataset(dataset_dir)
    
    # Convert to PyTorch tensors (regression targets are float)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    # Split training data into train and validation
    train_size = int(0.8 * len(X_train))
    val_size = len(X_train) - train_size
    
    train_dataset = TensorDataset(X_train, y_train)
    
    g = torch.Generator()
    g.manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=g
    )
    
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True, generator=g)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    return train_loader, val_loader, test_loader
