"""
Evaluate final clustering sensitivity to trajectory representation
resolution.

Primary configuration:
    100 arc-length points

Sensitivity configuration:
    200 arc-length points

Both use:
    symmetric Hausdorff distance with Haversine point distance
    manual HDBSCAN
    min_samples = 15
    min_cluster_size = 30

The purpose is to test whether the scientific clustering result
changes materially when trajectory representation resolution is
doubled.
"""

import json

import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score

from src.config import (
    DISTANCE_OUTPUT_DIR,
    HDBSCAN_OUTPUT_DIR,
)

from src.hdbscan_manual import (
    fit_hdbscan_precomputed,
)


# Configuration

MIN_SAMPLES = 15
MIN_CLUSTER_SIZE = 30

PRIMARY_DISTANCE_PATH = (
    DISTANCE_OUTPUT_DIR
    / "hausdorff_100.npy"
)

SENSITIVITY_DISTANCE_PATH = (
    DISTANCE_OUTPUT_DIR
    / "sensitivity"
    / "hausdorff_200.npy"
)

PRIMARY_INDEX_PATH = (
    DISTANCE_OUTPUT_DIR
    / "trajectory_index.csv"
)

SENSITIVITY_INDEX_PATH = (
    DISTANCE_OUTPUT_DIR
    / "sensitivity"
    / "trajectory_index_200.csv"
)

PRIMARY_LABELS_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "final"
    / "labels.csv"
)

OUTPUT_DIR = (
    HDBSCAN_OUTPUT_DIR
    / "representation_sensitivity"
)


# Helpers

def summarize_labels(
    labels,
):
    """
    Summarize one HDBSCAN partition.
    """

    labels = np.asarray(
        labels,
        dtype="int64",
    )

    cluster_labels = np.unique(
        labels[
            labels >= 0
        ]
    )

    cluster_sizes = sorted(
        [
            int(
                (
                    labels == cluster_label
                ).sum()
            )
            for cluster_label
            in cluster_labels
        ]
    )

    noise_count = int(
        (
            labels == -1
        ).sum()
    )

    return {
        "cluster_count": int(
            len(
                cluster_labels
            )
        ),

        "cluster_sizes": (
            cluster_sizes
        ),

        "noise_count": (
            noise_count
        ),

        "noise_percent": float(
            noise_count
            / len(
                labels
            )
            * 100.0
        ),
    }


def validate_index_alignment():
    """
    Confirm that the 100-point and 200-point matrices use the
    exact same trajectory ordering.
    """

    primary = pd.read_csv(
        PRIMARY_INDEX_PATH
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_INDEX_PATH
    )

    primary = (
        primary
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    sensitivity = (
        sensitivity
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    columns = [
        "matrix_index",
        "month",
        "trajectory_id",
        "file_name",
    ]

    if not primary[
        columns
    ].equals(
        sensitivity[
            columns
        ]
    ):
        raise RuntimeError(
            "100-point and 200-point trajectory ordering differs."
        )

    return primary


# Distance-matrix comparison

def compare_distance_matrices(
    matrix_100,
    matrix_200,
):
    """
    Compare the two complete pairwise trajectory-distance matrices.
    """

    if (
        matrix_100.shape
        != matrix_200.shape
    ):
        raise ValueError(
            "Distance matrices have different shapes."
        )

    upper_indices = np.triu_indices(
        matrix_100.shape[
            0
        ],
        k=1,
    )

    distances_100 = (
        matrix_100[
            upper_indices
        ]
    )

    distances_200 = (
        matrix_200[
            upper_indices
        ]
    )

    absolute_difference = np.abs(
        distances_200
        - distances_100
    )

    pearson = float(
        np.corrcoef(
            distances_100,
            distances_200,
        )[
            0,
            1,
        ]
    )

    return {
        "pair_count": int(
            len(
                distances_100
            )
        ),

        "pearson_correlation": (
            pearson
        ),

        "absolute_difference_km": {
            "median": float(
                np.median(
                    absolute_difference
                )
            ),

            "p95": float(
                np.percentile(
                    absolute_difference,
                    95,
                )
            ),

            "maximum": float(
                np.max(
                    absolute_difference
                )
            ),
        },
    }


# Neighborhood comparison

def compare_neighbors(
    matrix_100,
    matrix_200,
):
    """
    Compare nearest-neighbor structure for every trajectory.
    """

    n = matrix_100.shape[
        0
    ]

    nearest_100 = np.argsort(
        matrix_100,
        axis=1,
    )[
        :,
        1:
    ]

    nearest_200 = np.argsort(
        matrix_200,
        axis=1,
    )[
        :,
        1:
    ]

    exact_nearest_neighbor = (
        nearest_100[
            :,
            0
        ]
        == nearest_200[
            :,
            0
        ]
    )

    top5_overlap = []

    top10_overlap = []

    for index in range(
        n
    ):

        set_100_top5 = set(
            nearest_100[
                index,
                :5,
            ]
        )

        set_200_top5 = set(
            nearest_200[
                index,
                :5,
            ]
        )

        set_100_top10 = set(
            nearest_100[
                index,
                :10,
            ]
        )

        set_200_top10 = set(
            nearest_200[
                index,
                :10,
            ]
        )

        top5_overlap.append(
            len(
                set_100_top5
                & set_200_top5
            )
            / 5.0
        )

        top10_overlap.append(
            len(
                set_100_top10
                & set_200_top10
            )
            / 10.0
        )

    return {
        "exact_nearest_neighbor_count": int(
            exact_nearest_neighbor.sum()
        ),

        "exact_nearest_neighbor_percent": float(
            exact_nearest_neighbor.mean()
            * 100.0
        ),

        "mean_top5_overlap_percent": float(
            np.mean(
                top5_overlap
            )
            * 100.0
        ),

        "minimum_top5_overlap_percent": float(
            np.min(
                top5_overlap
            )
            * 100.0
        ),

        "mean_top10_overlap_percent": float(
            np.mean(
                top10_overlap
            )
            * 100.0
        ),

        "minimum_top10_overlap_percent": float(
            np.min(
                top10_overlap
            )
            * 100.0
        ),
    }


# Partition comparison

def compare_clusterings(
    labels_100,
    labels_200,
):
    """
    Compare final HDBSCAN partitions.

    Raw numeric cluster IDs are not assumed to have the same meaning,
    so label-invariant metrics are used.
    """

    labels_100 = np.asarray(
        labels_100,
        dtype="int64",
    )

    labels_200 = np.asarray(
        labels_200,
        dtype="int64",
    )

    ari_all = float(
        adjusted_rand_score(
            labels_100,
            labels_200,
        )
    )

    common_clustered = (
        (labels_100 >= 0)
        & (labels_200 >= 0)
    )

    common_clustered_count = int(
        common_clustered.sum()
    )

    if (
        common_clustered_count >= 2
    ):
        ari_common = float(
            adjusted_rand_score(
                labels_100[
                    common_clustered
                ],
                labels_200[
                    common_clustered
                ],
            )
        )

    else:
        ari_common = (
            np.nan
        )

    noise_100 = (
        labels_100 == -1
    )

    noise_200 = (
        labels_200 == -1
    )

    noise_intersection = int(
        (
            noise_100
            & noise_200
        ).sum()
    )

    noise_union = int(
        (
            noise_100
            | noise_200
        ).sum()
    )

    if noise_union == 0:
        noise_jaccard = 1.0

    else:
        noise_jaccard = float(
            noise_intersection
            / noise_union
        )

    noise_mask_agreement = float(
        (
            noise_100
            == noise_200
        ).mean()
    )

    noise_100_clustered_200 = int(
        (
            noise_100
            & ~noise_200
        ).sum()
    )

    clustered_100_noise_200 = int(
        (
            ~noise_100
            & noise_200
        ).sum()
    )

    return {
        "ari_all": (
            ari_all
        ),

        "ari_common_clustered": (
            ari_common
        ),

        "common_clustered_count": (
            common_clustered_count
        ),

        "noise_intersection": (
            noise_intersection
        ),

        "noise_union": (
            noise_union
        ),

        "noise_jaccard": (
            noise_jaccard
        ),

        "noise_mask_agreement": (
            noise_mask_agreement
        ),

        "noise_100_but_clustered_200": (
            noise_100_clustered_200
        ),

        "clustered_100_but_noise_200": (
            clustered_100_noise_200
        ),
    }


def build_crosstab(
    labels_100,
    labels_200,
):
    """
    Build membership contingency table.
    """

    table = pd.DataFrame(
        {
            "label_100": (
                labels_100
            ),

            "label_200": (
                labels_200
            ),
        }
    )

    return pd.crosstab(
        table[
            "label_100"
        ],
        table[
            "label_200"
        ],
        margins=True,
    )


# Main

def main():

    print(
        "=" * 100
    )

    print(
        "100-POINT VS 200-POINT REPRESENTATION SENSITIVITY"
    )

    print(
        "=" * 100
    )
    # Verify ordering
    trajectory_index = (
        validate_index_alignment()
    )

    print(
        "\nTrajectory-index alignment: PASS"
    )

    print(
        "Trajectories:",
        len(
            trajectory_index
        ),
    )
    # Load matrices
    matrix_100 = np.load(
        PRIMARY_DISTANCE_PATH
    )

    matrix_200 = np.load(
        SENSITIVITY_DISTANCE_PATH
    )

    print(
        "\n100-point matrix:",
        matrix_100.shape,
    )

    print(
        "200-point matrix:",
        matrix_200.shape,
    )
    # Distance comparison
    distance_comparison = (
        compare_distance_matrices(
            matrix_100,
            matrix_200,
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PAIRWISE DISTANCE AGREEMENT"
    )

    print(
        "=" * 100
    )

    print(
        "\nUnique pairs:",
        f"{distance_comparison['pair_count']:,}",
    )

    print(
        "Pearson correlation:",
        (
            f"{distance_comparison['pearson_correlation']:.8f}"
        ),
    )

    difference = (
        distance_comparison[
            "absolute_difference_km"
        ]
    )

    print(
        "\nAbsolute 100-vs-200 distance difference:"
    )

    print(
        f"  median:  "
        f"{difference['median']:.6f} km"
    )

    print(
        f"  p95:     "
        f"{difference['p95']:.6f} km"
    )

    print(
        f"  maximum: "
        f"{difference['maximum']:.6f} km"
    )
    # Neighborhood comparison
    neighborhood = (
        compare_neighbors(
            matrix_100,
            matrix_200,
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "NEIGHBORHOOD AGREEMENT"
    )

    print(
        "=" * 100
    )

    print(
        "\nExact nearest neighbor:",
        (
            f"{neighborhood['exact_nearest_neighbor_count']} "
            f"/ {len(trajectory_index)} "
            f"({neighborhood['exact_nearest_neighbor_percent']:.2f}%)"
        ),
    )

    print(
        "Mean top-5 overlap:",
        (
            f"{neighborhood['mean_top5_overlap_percent']:.2f}%"
        ),
    )

    print(
        "Minimum top-5 overlap:",
        (
            f"{neighborhood['minimum_top5_overlap_percent']:.2f}%"
        ),
    )

    print(
        "Mean top-10 overlap:",
        (
            f"{neighborhood['mean_top10_overlap_percent']:.2f}%"
        ),
    )

    print(
        "Minimum top-10 overlap:",
        (
            f"{neighborhood['minimum_top10_overlap_percent']:.2f}%"
        ),
    )
    # Load primary 100-point labels
    primary_labels_table = pd.read_csv(
        PRIMARY_LABELS_PATH
    )

    primary_labels_table = (
        primary_labels_table
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    labels_100 = (
        primary_labels_table[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )
    # Run final HDBSCAN on 200-point matrix
    print(
        "\n"
        + "=" * 100
    )

    print(
        "RUNNING HDBSCAN ON 200-POINT MATRIX"
    )

    print(
        "=" * 100
    )

    print(
        "\nmin_samples =",
        MIN_SAMPLES,
    )

    print(
        "min_cluster_size =",
        MIN_CLUSTER_SIZE,
    )

    result_200 = (
        fit_hdbscan_precomputed(
            distance_matrix=(
                matrix_200
            ),

            min_samples=(
                MIN_SAMPLES
            ),

            min_cluster_size=(
                MIN_CLUSTER_SIZE
            ),

            allow_single_cluster=False,
        )
    )

    labels_200 = (
        result_200[
            "labels"
        ]
    )

    summary_100 = (
        summarize_labels(
            labels_100
        )
    )

    summary_200 = (
        summarize_labels(
            labels_200
        )
    )

    comparison = (
        compare_clusterings(
            labels_100,
            labels_200,
        )
    )

    crosstab = (
        build_crosstab(
            labels_100,
            labels_200,
        )
    )
    # Report
    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL CLUSTERING COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        "\n100-point primary:"
    )

    print(
        "Clusters:",
        summary_100[
            "cluster_count"
        ],
    )

    print(
        "Cluster sizes:",
        summary_100[
            "cluster_sizes"
        ],
    )

    print(
        "Noise:",
        summary_100[
            "noise_count"
        ],
        (
            f"({summary_100['noise_percent']:.3f}%)"
        ),
    )

    print(
        "\n200-point sensitivity:"
    )

    print(
        "Clusters:",
        summary_200[
            "cluster_count"
        ],
    )

    print(
        "Cluster sizes:",
        summary_200[
            "cluster_sizes"
        ],
    )

    print(
        "Noise:",
        summary_200[
            "noise_count"
        ],
        (
            f"({summary_200['noise_percent']:.3f}%)"
        ),
    )

    print(
        "\nAdjusted Rand Index:",
        f"{comparison['ari_all']:.6f}",
    )

    print(
        "ARI among trajectories clustered by both:",
        (
            f"{comparison['ari_common_clustered']:.6f}"
        ),
    )

    print(
        "Trajectories clustered by both:",
        comparison[
            "common_clustered_count"
        ],
    )

    print(
        "\nNoise-set Jaccard:",
        (
            f"{comparison['noise_jaccard']:.6f}"
        ),
    )

    print(
        "Noise-mask agreement:",
        (
            f"{comparison['noise_mask_agreement']:.6f}"
        ),
    )

    print(
        "Noise in both:",
        comparison[
            "noise_intersection"
        ],
    )

    print(
        "100-point noise but 200-point clustered:",
        comparison[
            "noise_100_but_clustered_200"
        ],
    )

    print(
        "100-point clustered but 200-point noise:",
        comparison[
            "clustered_100_but_noise_200"
        ],
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "100-POINT ↔ 200-POINT MEMBERSHIP TABLE"
    )

    print(
        "=" * 100
    )

    print(
        "\n",
        crosstab.to_string(),
        sep="",
    )
    # Save outputs
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_200_table = (
        trajectory_index[
            [
                "matrix_index",
                "month",
                "trajectory_id",
                "file_name",
            ]
        ]
        .copy()
    )

    labels_200_table[
        "cluster_label_200"
    ] = (
        labels_200
    )

    labels_200_table[
        "selected_cluster_id_200"
    ] = (
        result_200[
            "selected_cluster_ids"
        ]
    )

    labels_200_table.to_csv(
        OUTPUT_DIR
        / "labels_200_points.csv",
        index=False,
    )

    crosstab.to_csv(
        OUTPUT_DIR
        / "membership_100_vs_200.csv"
    )

    report = {
        "primary_representation_points": (
            100
        ),

        "sensitivity_representation_points": (
            200
        ),

        "hdbscan_parameters": {
            "min_samples": (
                MIN_SAMPLES
            ),

            "min_cluster_size": (
                MIN_CLUSTER_SIZE
            ),
        },

        "distance_matrix_comparison": (
            distance_comparison
        ),

        "neighborhood_comparison": (
            neighborhood
        ),

        "primary_100_summary": (
            summary_100
        ),

        "sensitivity_200_summary": (
            summary_200
        ),

        "partition_comparison": (
            comparison
        ),

        "selected_stability_200": float(
            result_200[
                "selected_stability"
            ]
        ),
    }

    with open(
        OUTPUT_DIR
        / "representation_sensitivity_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "labels_200_points.csv"
    )

    print(
        OUTPUT_DIR
        / "membership_100_vs_200.csv"
    )

    print(
        OUTPUT_DIR
        / "representation_sensitivity_summary.json"
    )


if __name__ == "__main__":
    main()