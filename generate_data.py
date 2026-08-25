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

# Generate and save a new 50x100 signed matrix.
print("Generating 50x100 signed matrix...")
matrix = generate_random_matrix(rows=50, cols=100, distribution="signed", seed=1000)
matrix_path = '/workspaces/Matrig/Mat1/random_matrix_50x100_signed.hkl'
save_matrix(matrix, matrix_path)

# Generate and save the custom dataset
print("\nGenerating custom dataset...")
X_train, y_train, X_test, y_test, matrix_used = generate_dataset(
    num_train_samples=20000,
    num_test_samples=5000,
    input_dim=100,
    output_dim=50,
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
