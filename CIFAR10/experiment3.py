import argparse
import csv
import os

import hickle as hkl
import torch


def load_forward_weights(run_name, output_root="weight_dumps"):
    weights_path = os.path.join(output_root, run_name, "forward_weights.hkl")
    saved_weights = hkl.load(weights_path)
    layer_keys = sorted(
        saved_weights,
        key=lambda key: int(key.removeprefix("layer_")),
    )
    return [
        torch.as_tensor(saved_weights[key], dtype=torch.float64)
        for key in layer_keys
    ]


def average_adjacent_singular_vector_cosines(weights):
    """Compare U_i with V_(i+1), using absolute cosine values."""
    statistics = []

    for layer_index, (weight_i, weight_next) in enumerate(
        zip(weights, weights[1:]),
        start=1,
    ):
        left_vectors, _, _ = torch.linalg.svd(weight_i, full_matrices=False)
        _, _, next_right_vectors_transposed = torch.linalg.svd(
            weight_next,
            full_matrices=False,
        )
        next_right_vectors = next_right_vectors_transposed.T

        if left_vectors.shape[0] != next_right_vectors.shape[0]:
            raise ValueError(
                f"Layer {layer_index} dimensions do not match: "
                f"U_i={tuple(left_vectors.shape)}, "
                f"V_i+1={tuple(next_right_vectors.shape)}"
            )

        vector_count = min(left_vectors.shape[1], next_right_vectors.shape[1])
        cosine_values = torch.abs(
            torch.sum(
                left_vectors[:, :vector_count]
                * next_right_vectors[:, :vector_count],
                dim=0,
            )
        )
        statistics.append(float(cosine_values.mean()))

    return statistics


def run_experiment(
    run_prefix,
    output_root="weight_dumps",
    conditions=("balanced", "xavier"),
):
    rows = []

    for condition in conditions:
        run_name = f"{run_prefix}_{condition}"
        weights = load_forward_weights(run_name, output_root=output_root)
        statistics = average_adjacent_singular_vector_cosines(weights)

        print(f"\n{condition}: {run_name}")
        for layer_index, statistic in enumerate(statistics, start=1):
            print(
                f"Layer {layer_index} -> {layer_index + 1}: "
                f"{statistic:.8f}"
            )
            rows.append(
                {
                    "run_name": run_name,
                    "condition": condition,
                    "adjacent_layer_pair": f"{layer_index}->{layer_index + 1}",
                    "average_absolute_cosine_similarity": statistic,
                }
            )

    csv_path = os.path.join(
        output_root,
        f"singular_vector_alignment_{run_prefix}.csv",
    )
    with open(csv_path, "w", newline="") as csv_file:
        fieldnames = [
            "run_name",
            "condition",
            "adjacent_layer_pair",
            "average_absolute_cosine_similarity",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved Experiment 3 statistics to {csv_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Measure adjacent-layer singular-vector alignment."
    )
    parser.add_argument(
        "run_prefix",
        help="Prefix used by Experiment 2, e.g. rfa_4738_B0.050_2",
    )
    parser.add_argument(
        "--output-root",
        default="weight_dumps",
        help="Directory containing the per-run Hickle folders.",
    )
    args = parser.parse_args()
    run_experiment(args.run_prefix, output_root=args.output_root)


if __name__ == "__main__":
    main()
