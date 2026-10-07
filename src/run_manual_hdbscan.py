"""
Run the final selected manual HDBSCAN configuration.

Primary configuration
---------------------
Trajectory representation:
    100 arc-length points

Trajectory distance:
    symmetric Hausdorff distance using Haversine point distance

HDBSCAN parameters:
    min_samples = 15
    min_cluster_size = 30

This script starts from the frozen precomputed Hausdorff
distance matrix and reconstructs the entire manual HDBSCAN
pipeline from scratch.

It also verifies that the clean implementation reproduces
the previously validated experimental result.
"""

import json

import numpy as np
import pandas as pd

from src.config import (
    DISTANCE_OUTPUT_DIR,
    HDBSCAN_OUTPUT_DIR,
)

from src.hdbscan_manual import (
    fit_hdbscan_precomputed,
)


# Final primary HDBSCAN parameters

MIN_SAMPLES = 15
MIN_CLUSTER_SIZE = 30

EXPECTED_CLUSTER_COUNT = 4
EXPECTED_NOISE_COUNT = 295


# Paths

FINAL_OUTPUT_DIR = (
    HDBSCAN_OUTPUT_DIR
    / "final"
)


# Utility

def load_inputs():
    """
    Load frozen distance data and trajectory ordering.
    """

    distance_matrix = np.load(
        DISTANCE_OUTPUT_DIR
        / "hausdorff_100.npy"
    )

    trajectory_index = pd.read_csv(
        DISTANCE_OUTPUT_DIR
        / "trajectory_index.csv"
    )

    trajectory_index = (
        trajectory_index
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            trajectory_index
        )
        != distance_matrix.shape[0]
    ):
        raise RuntimeError(
            "Trajectory index and distance matrix "
            "have different sizes."
        )

    expected_indices = np.arange(
        len(
            trajectory_index
        )
    )

    actual_indices = (
        trajectory_index[
            "matrix_index"
        ]
        .to_numpy()
    )

    if not np.array_equal(
        expected_indices,
        actual_indices,
    ):
        raise RuntimeError(
            "Trajectory matrix indices are not sequential."
        )

    return (
        distance_matrix,
        trajectory_index,
    )


def build_label_table(
    trajectory_index,
    result,
):
    """
    Attach final manual HDBSCAN labels to trajectory metadata.
    """

    labels = (
        result[
            "labels"
        ]
    )

    selected_cluster_ids = (
        result[
            "selected_cluster_ids"
        ]
    )

    label_table = (
        trajectory_index
        .copy()
    )

    label_table[
        "cluster_label"
    ] = labels

    label_table[
        "selected_cluster_id"
    ] = selected_cluster_ids

    return label_table


def summarize_result(
    labels,
    result,
):
    """
    Produce final clustering summary.
    """

    cluster_mask = (
        labels >= 0
    )

    noise_mask = (
        labels == -1
    )

    unique_clusters = np.unique(
        labels[
            cluster_mask
        ]
    )

    cluster_sizes = []

    for cluster_label in (
        unique_clusters
    ):
        cluster_sizes.append(
            int(
                (
                    labels
                    == cluster_label
                ).sum()
            )
        )

    cluster_sizes = sorted(
        cluster_sizes
    )

    return {
        "min_samples": (
            MIN_SAMPLES
        ),

        "min_cluster_size": (
            MIN_CLUSTER_SIZE
        ),

        "trajectory_count": (
            int(
                len(
                    labels
                )
            )
        ),

        "selected_cluster_count": (
            int(
                len(
                    unique_clusters
                )
            )
        ),

        "noise_count": (
            int(
                noise_mask.sum()
            )
        ),

        "noise_percent": (
            float(
                noise_mask.mean()
                * 100
            )
        ),

        "cluster_sizes": (
            cluster_sizes
        ),

        "minimum_cluster_size_observed": (
            int(
                min(
                    cluster_sizes
                )
            )
        ),

        "median_cluster_size_observed": (
            float(
                np.median(
                    cluster_sizes
                )
            )
        ),

        "maximum_cluster_size_observed": (
            int(
                max(
                    cluster_sizes
                )
            )
        ),

        "selected_stability": (
            float(
                result[
                    "selected_stability"
                ]
            )
        ),
    }


# Reproduction validation

def validate_against_experimental_result(
    final_label_table,
):
    """
    Compare the clean final implementation against the
    previously validated experimental k=15 / mcs=30 result.
    """

    previous_path = (
        HDBSCAN_OUTPUT_DIR
        / "labels_k15_mcs30.csv"
    )

    if not previous_path.exists():
        raise FileNotFoundError(
            "Previous experimental labels were not found: "
            f"{previous_path}"
        )

    previous = pd.read_csv(
        previous_path
    )

    previous = (
        previous
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    current = (
        final_label_table
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    if not np.array_equal(
        previous[
            "matrix_index"
        ].to_numpy(),
        current[
            "matrix_index"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "Previous and current trajectory ordering differs."
        )

    previous_labels = (
        previous[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    current_labels = (
        current[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    exact_label_match = (
        np.array_equal(
            previous_labels,
            current_labels,
        )
    )

    previous_noise = (
        previous_labels == -1
    )

    current_noise = (
        current_labels == -1
    )

    exact_noise_match = (
        np.array_equal(
            previous_noise,
            current_noise,
        )
    )

    changed_labels = int(
        (
            previous_labels
            != current_labels
        ).sum()
    )

    changed_noise_status = int(
        (
            previous_noise
            != current_noise
        ).sum()
    )

    return {
        "exact_label_match": (
            bool(
                exact_label_match
            )
        ),

        "exact_noise_match": (
            bool(
                exact_noise_match
            )
        ),

        "changed_label_count": (
            changed_labels
        ),

        "changed_noise_status_count": (
            changed_noise_status
        ),
    }


# Saving

def save_final_outputs(
    result,
    label_table,
    summary,
    reproduction,
):
    """
    Save clean final manual-HDBSCAN artifacts.
    """

    FINAL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        FINAL_OUTPUT_DIR
        / "core_distances.npy",
        result[
            "core_distances"
        ],
    )

    result[
        "mst"
    ].to_csv(
        FINAL_OUTPUT_DIR
        / "mst.csv",
        index=False,
    )

    result[
        "single_linkage"
    ].to_csv(
        FINAL_OUTPUT_DIR
        / "single_linkage.csv",
        index=False,
    )

    result[
        "condensed_tree"
    ].to_csv(
        FINAL_OUTPUT_DIR
        / "condensed_tree.csv",
        index=False,
    )

    result[
        "cluster_table"
    ].to_csv(
        FINAL_OUTPUT_DIR
        / "cluster_stability.csv",
        index=False,
    )

    label_table.to_csv(
        FINAL_OUTPUT_DIR
        / "labels.csv",
        index=False,
    )

    combined_summary = {
        "method": (
            "manual_hdbscan"
        ),

        "distance_metric": (
            "symmetric_hausdorff_haversine"
        ),

        "trajectory_representation_points": (
            100
        ),

        **summary,

        "reproduction_validation": (
            reproduction
        ),
    }

    with open(
        FINAL_OUTPUT_DIR
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            combined_summary,
            file,
            indent=4,
        )


# Main

def main():

    print(
        "=" * 80
    )

    print(
        "FINAL MANUAL HDBSCAN RUN"
    )

    print(
        "=" * 80
    )

    print(
        "\nPrimary parameters:"
    )

    print(
        f"min_samples = "
        f"{MIN_SAMPLES}"
    )

    print(
        f"min_cluster_size = "
        f"{MIN_CLUSTER_SIZE}"
    )

    # Load frozen inputs

    (
        distance_matrix,
        trajectory_index,
    ) = load_inputs()

    print(
        "\nDistance matrix:",
        distance_matrix.shape,
    )

    print(
        "Trajectories:",
        len(
            trajectory_index
        ),
    )

    # Entire manual HDBSCAN pipeline

    print(
        "\nRunning complete manual HDBSCAN pipeline..."
    )

    result = fit_hdbscan_precomputed(
        distance_matrix=(
            distance_matrix
        ),

        min_samples=(
            MIN_SAMPLES
        ),

        min_cluster_size=(
            MIN_CLUSTER_SIZE
        ),

        allow_single_cluster=False,
    )

    label_table = build_label_table(
        trajectory_index,
        result,
    )

    labels = (
        result[
            "labels"
        ]
    )

    summary = summarize_result(
        labels,
        result,
    )

    # Expected-result validation

    if (
        summary[
            "selected_cluster_count"
        ]
        != EXPECTED_CLUSTER_COUNT
    ):
        raise RuntimeError(
            "Unexpected final cluster count: "
            f"{summary['selected_cluster_count']}"
        )

    if (
        summary[
            "noise_count"
        ]
        != EXPECTED_NOISE_COUNT
    ):
        raise RuntimeError(
            "Unexpected final noise count: "
            f"{summary['noise_count']}"
        )

    reproduction = (
        validate_against_experimental_result(
            label_table
        )
    )

    if not (
        reproduction[
            "exact_label_match"
        ]
    ):
        raise RuntimeError(
            "Clean implementation does not exactly "
            "reproduce the experimental labels."
        )

    # Save only after all validations pass

    save_final_outputs(
        result,
        label_table,
        summary,
        reproduction,
    )

    # Report

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL MANUAL HDBSCAN RESULT"
    )

    print(
        "=" * 80
    )

    print(
        "\nSelected clusters:",
        summary[
            "selected_cluster_count"
        ],
    )

    print(
        "Cluster sizes:",
        summary[
            "cluster_sizes"
        ],
    )

    print(
        "Noise trajectories:",
        summary[
            "noise_count"
        ],
    )

    print(
        "Noise percentage:",
        (
            f"{summary['noise_percent']:.3f}%"
        ),
    )

    print(
        "Selected EOM stability:",
        (
            f"{summary['selected_stability']:.6f}"
        ),
    )

    print(
        "\nReproduction validation:"
    )

    print(
        "Exact label match:",
        reproduction[
            "exact_label_match"
        ],
    )

    print(
        "Exact noise match:",
        reproduction[
            "exact_noise_match"
        ],
    )

    print(
        "Changed labels:",
        reproduction[
            "changed_label_count"
        ],
    )

    print(
        "Changed noise statuses:",
        reproduction[
            "changed_noise_status_count"
        ],
    )

    print(
        "\nFINAL REPRODUCTION: PASS"
    )

    print(
        "\nSaved:"
    )

    print(
        "outputs/clusters/final/"
    )

    print(
        "  core_distances.npy"
    )

    print(
        "  mst.csv"
    )

    print(
        "  single_linkage.csv"
    )

    print(
        "  condensed_tree.csv"
    )

    print(
        "  cluster_stability.csv"
    )

    print(
        "  labels.csv"
    )

    print(
        "  summary.json"
    )


if __name__ == "__main__":
    main()