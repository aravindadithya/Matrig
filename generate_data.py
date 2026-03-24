"""
Script to generate and save the matrix and custom dataset.
Run this script once to create the necessary files.
"""

import sys
sys.path.insert(0, '/workspaces/Matrig')

from utils.mat_gen import (
    generate_random_matrix, save_matrix,
    generate_dataset, save_dataset
)

# Generate and save a 784x784 matrix (matching flattened image dimension)
print("Generating 784x784 matrix...")
matrix = generate_random_matrix(n=784, mean=0.0, std=1.0, seed=1000)
matrix_path = '/workspaces/Matrig/Mat1/random_matrix_784x784.hkl'
save_matrix(matrix, matrix_path)

# Generate and save the custom dataset
print("\nGenerating custom dataset...")
X_train, y_train, X_test, y_test, matrix_used = generate_dataset(
    num_train_samples=60000,
    num_test_samples=10000,
    input_dim=784,
    matrix=matrix,
    seed=1000
)

# Safety check: the generated targets must come from the exact same matrix
assert matrix_used.shape == matrix.shape

dataset_dir = '/workspaces/Matrig/data/custom_dataset'
save_dataset(X_train, y_train, X_test, y_test, dataset_dir)

print("\nGeneration complete!")
print(f"Matrix saved to: {matrix_path}")
print(f"Dataset saved to: {dataset_dir}")
