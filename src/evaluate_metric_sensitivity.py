"""
Evaluate trajectory-distance metric sensitivity.

Primary metric:
    Symmetric Hausdorff distance using Haversine point distance.

Sensitivity metric:
    Discrete Frechet distance using Haversine point distance.

Both metrics use:
    - the same 1,354 trajectories;
    - the same 100-point arc-length representations;
    - manual HDBSCAN;
    - min_samples = 15;
    - min_cluster_size = 30.

The objective is to determine whether changing the trajectory
distance metric materially changes the final route clustering.
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

HAUSDORFF_PATH = (
    DISTANCE_OUTPUT_DIR
    / "hausdorff_100.npy"
)

FRECHET_PATH = (
    DISTANCE_OUTPUT_DIR
    / "sensitivity"
    / "frechet_100.npy"
)

TRAJECTORY_INDEX_PATH = (
    DISTANCE_OUTPUT_DIR
    / "trajectory_index.csv"
)

PRIMARY_LABELS_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "final"
    / "labels.csv"
)

OUTPUT_DIR = (
    HDBSCAN_OUTPUT_DIR
    / "metric_sensitivity"
)

# Helpers
def summarize_labels(labels):
    """
    Summarize one clustering partition.
    """

    labels = np.asarray(
        labels,
        dtype="int64",
    )

    cluster_ids = np.unique(
        labels[
            labels >= 0
        ]
    )

    cluster_sizes = sorted(
        [
            int(
                (
                    labels
                    == cluster_id
                ).sum()
            )
            for cluster_id
            in cluster_ids
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
                cluster_ids
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

# Distance-matrix comparison
def compare_distance_matrices(
    hausdorff,
    frechet,
):
    """
    Compare all unique trajectory-pair distances.
    """

    if hausdorff.shape != frechet.shape:
        raise ValueError(
            "Hausdorff and Frechet matrices "
            "have different shapes."
        )

    indices = np.triu_indices(
        hausdorff.shape[
            0
        ],
        k=1,
    )

    hausdorff_values = (
        hausdorff[
            indices
        ]
    )

    frechet_values = (
        frechet[
            indices
        ]
    )

    difference = (
        frechet_values
        - hausdorff_values
    )

    absolute_difference = np.abs(
        difference
    )

    correlation = float(
        np.corrcoef(
            hausdorff_values,
            frechet_values,
        )[
            0,
            1,
        ]
    )

    tolerance = 1e-9

    frechet_less_than_hausdorff = int(
        (
            difference
            < -tolerance
        ).sum()
    )

    effectively_equal = int(
        (
            np.abs(
                difference
            )
            <= tolerance
        ).sum()
    )

    frechet_greater = int(
        (
            difference
            > tolerance
        ).sum()
    )

    return {
        "pair_count": int(
            len(
                hausdorff_values
            )
        ),

        "pearson_correlation": (
            correlation
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

        "signed_difference_frechet_minus_hausdorff_km": {
            "minimum": float(
                np.min(
                    difference
                )
            ),

            "median": float(
                np.median(
                    difference
                )
            ),

            "p95": float(
                np.percentile(
                    difference,
                    95,
                )
            ),

            "maximum": float(
                np.max(
                    difference
                )
            ),
        },

        "frechet_less_than_hausdorff_count": (
            frechet_less_than_hausdorff
        ),

        "effectively_equal_count": (
            effectively_equal
        ),

        "frechet_greater_than_hausdorff_count": (
            frechet_greater
        ),
    }

# Neighborhood comparison
def compare_neighbors(
    hausdorff,
    frechet,
):
    """
    Compare nearest-neighbor structure under the two metrics.
    """

    n = hausdorff.shape[
        0
    ]

    nearest_hausdorff = np.argsort(
        hausdorff,
        axis=1,
    )[
        :,
        1:
    ]

    nearest_frechet = np.argsort(
        frechet,
        axis=1,
    )[
        :,
        1:
    ]

    exact_nearest = (
        nearest_hausdorff[
            :,
            0
        ]
        == nearest_frechet[
            :,
            0
        ]
    )

    top5_overlap = []

    top10_overlap = []

    for index in range(
        n
    ):

        h5 = set(
            nearest_hausdorff[
                index,
                :5,
            ]
        )

        f5 = set(
            nearest_frechet[
                index,
                :5,
            ]
        )

        h10 = set(
            nearest_hausdorff[
                index,
                :10,
            ]
        )

        f10 = set(
            nearest_frechet[
                index,
                :10,
            ]
        )

        top5_overlap.append(
            len(
                h5 & f5
            )
            / 5.0
        )

        top10_overlap.append(
            len(
                h10 & f10
            )
            / 10.0
        )

    return {
        "exact_nearest_neighbor_count": int(
            exact_nearest.sum()
        ),

        "exact_nearest_neighbor_percent": float(
            exact_nearest.mean()
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
    hausdorff_labels,
    frechet_labels,
):
    """
    Compare final clustering partitions using label-invariant metrics.
    """

    hausdorff_labels = np.asarray(
        hausdorff_labels,
        dtype="int64",
    )

    frechet_labels = np.asarray(
        frechet_labels,
        dtype="int64",
    )

    ari_all = float(
        adjusted_rand_score(
            hausdorff_labels,
            frechet_labels,
        )
    )

    common_clustered = (
        (hausdorff_labels >= 0)
        & (frechet_labels >= 0)
    )

    common_clustered_count = int(
        common_clustered.sum()
    )

    if common_clustered_count >= 2:

        ari_common = float(
            adjusted_rand_score(
                hausdorff_labels[
                    common_clustered
                ],
                frechet_labels[
                    common_clustered
                ],
            )
        )

    else:

        ari_common = (
            np.nan
        )

    hausdorff_noise = (
        hausdorff_labels == -1
    )

    frechet_noise = (
        frechet_labels == -1
    )

    noise_intersection = int(
        (
            hausdorff_noise
            & frechet_noise
        ).sum()
    )

    noise_union = int(
        (
            hausdorff_noise
            | frechet_noise
        ).sum()
    )

    if noise_union == 0:

        noise_jaccard = 1.0

    else:

        noise_jaccard = float(
            noise_intersection
            / noise_union
        )

    noise_agreement = float(
        (
            hausdorff_noise
            == frechet_noise
        ).mean()
    )

    hausdorff_noise_frechet_cluster = int(
        (
            hausdorff_noise
            & ~frechet_noise
        ).sum()
    )

    hausdorff_cluster_frechet_noise = int(
        (
            ~hausdorff_noise
            & frechet_noise
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
            noise_agreement
        ),

        "hausdorff_noise_but_frechet_clustered": (
            hausdorff_noise_frechet_cluster
        ),

        "hausdorff_clustered_but_frechet_noise": (
            hausdorff_cluster_frechet_noise
        ),
    }


def build_crosstab(
    hausdorff_labels,
    frechet_labels,
):
    """
    Build membership contingency table.
    """

    table = pd.DataFrame(
        {
            "hausdorff_label": (
                hausdorff_labels
            ),

            "frechet_label": (
                frechet_labels
            ),
        }
    )

    return pd.crosstab(
        table[
            "hausdorff_label"
        ],
        table[
            "frechet_label"
        ],
        margins=True,
    )

# Main
def main():

    print(
        "=" * 100
    )

    print(
        "HAUSDORFF VS DISCRETE FRECHET METRIC SENSITIVITY"
    )

    print(
        "=" * 100
    )
    # Load matrices
    hausdorff = np.load(
        HAUSDORFF_PATH
    )

    frechet = np.load(
        FRECHET_PATH
    )

    trajectory_index = pd.read_csv(
        TRAJECTORY_INDEX_PATH
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

    print(
        "\nHausdorff matrix:",
        hausdorff.shape,
    )

    print(
        "Frechet matrix:",
        frechet.shape,
    )

    print(
        "Trajectories:",
        len(
            trajectory_index
        ),
    )

    if (
        hausdorff.shape
        != frechet.shape
    ):

        raise RuntimeError(
            "Distance matrices have different shapes."
        )

    if (
        hausdorff.shape[
            0
        ]
        != len(
            trajectory_index
        )
    ):

        raise RuntimeError(
            "Trajectory index and distance matrix "
            "sizes differ."
        )
    # Distance comparison
    distance_comparison = (
        compare_distance_matrices(
            hausdorff,
            frechet,
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

    absolute = (
        distance_comparison[
            "absolute_difference_km"
        ]
    )

    print(
        "\nAbsolute metric difference:"
    )

    print(
        f"  median:  "
        f"{absolute['median']:.6f} km"
    )

    print(
        f"  p95:     "
        f"{absolute['p95']:.6f} km"
    )

    print(
        f"  maximum: "
        f"{absolute['maximum']:.6f} km"
    )

    signed = (
        distance_comparison[
            "signed_difference_frechet_minus_hausdorff_km"
        ]
    )

    print(
        "\nFrechet - Hausdorff:"
    )

    print(
        f"  minimum: "
        f"{signed['minimum']:.9f} km"
    )

    print(
        f"  median:  "
        f"{signed['median']:.6f} km"
    )

    print(
        f"  p95:     "
        f"{signed['p95']:.6f} km"
    )

    print(
        f"  maximum: "
        f"{signed['maximum']:.6f} km"
    )

    print(
        "\nFrechet < Hausdorff pairs:",
        distance_comparison[
            "frechet_less_than_hausdorff_count"
        ],
    )

    print(
        "Effectively equal pairs:",
        distance_comparison[
            "effectively_equal_count"
        ],
    )

    print(
        "Frechet > Hausdorff pairs:",
        distance_comparison[
            "frechet_greater_than_hausdorff_count"
        ],
    )
    # Neighborhood comparison
    neighborhood = (
        compare_neighbors(
            hausdorff,
            frechet,
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
    # Primary Hausdorff labels
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

    hausdorff_labels = (
        primary_labels_table[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )
    # HDBSCAN using Frechet
    print(
        "\n"
        + "=" * 100
    )

    print(
        "RUNNING HDBSCAN ON FRECHET MATRIX"
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

    frechet_result = (
        fit_hdbscan_precomputed(
            distance_matrix=(
                frechet
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

    frechet_labels = (
        frechet_result[
            "labels"
        ]
    )

    hausdorff_summary = (
        summarize_labels(
            hausdorff_labels
        )
    )

    frechet_summary = (
        summarize_labels(
            frechet_labels
        )
    )

    comparison = (
        compare_clusterings(
            hausdorff_labels,
            frechet_labels,
        )
    )

    crosstab = (
        build_crosstab(
            hausdorff_labels,
            frechet_labels,
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
        "\nHausdorff primary:"
    )

    print(
        "Clusters:",
        hausdorff_summary[
            "cluster_count"
        ],
    )

    print(
        "Cluster sizes:",
        hausdorff_summary[
            "cluster_sizes"
        ],
    )

    print(
        "Noise:",
        hausdorff_summary[
            "noise_count"
        ],
        (
            f"({hausdorff_summary['noise_percent']:.3f}%)"
        ),
    )

    print(
        "\nDiscrete Frechet sensitivity:"
    )

    print(
        "Clusters:",
        frechet_summary[
            "cluster_count"
        ],
    )

    print(
        "Cluster sizes:",
        frechet_summary[
            "cluster_sizes"
        ],
    )

    print(
        "Noise:",
        frechet_summary[
            "noise_count"
        ],
        (
            f"({frechet_summary['noise_percent']:.3f}%)"
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
        "Hausdorff noise but Frechet clustered:",
        comparison[
            "hausdorff_noise_but_frechet_clustered"
        ],
    )

    print(
        "Hausdorff clustered but Frechet noise:",
        comparison[
            "hausdorff_clustered_but_frechet_noise"
        ],
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "HAUSDORFF ↔ FRECHET MEMBERSHIP TABLE"
    )

    print(
        "=" * 100
    )

    print(
        "\n",
        crosstab.to_string(),
        sep="",
    )
    # Save sensitivity artifacts
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frechet_label_table = (
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

    frechet_label_table[
        "frechet_cluster_label"
    ] = (
        frechet_labels
    )

    frechet_label_table[
        "frechet_selected_cluster_id"
    ] = (
        frechet_result[
            "selected_cluster_ids"
        ]
    )

    frechet_label_table.to_csv(
        OUTPUT_DIR
        / "labels_frechet_100.csv",
        index=False,
    )

    crosstab.to_csv(
        OUTPUT_DIR
        / "membership_hausdorff_vs_frechet.csv"
    )

    report = {
        "representation_points": (
            100
        ),

        "primary_metric": (
            "symmetric_hausdorff_haversine"
        ),

        "sensitivity_metric": (
            "discrete_frechet_haversine"
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

        "hausdorff_summary": (
            hausdorff_summary
        ),

        "frechet_summary": (
            frechet_summary
        ),

        "partition_comparison": (
            comparison
        ),

        "frechet_selected_stability": float(
            frechet_result[
                "selected_stability"
            ]
        ),
    }

    with open(
        OUTPUT_DIR
        / "metric_sensitivity_summary.json",
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
        / "labels_frechet_100.csv"
    )

    print(
        OUTPUT_DIR
        / "membership_hausdorff_vs_frechet.csv"
    )

    print(
        OUTPUT_DIR
        / "metric_sensitivity_summary.json"
    )


if __name__ == "__main__":
    main()