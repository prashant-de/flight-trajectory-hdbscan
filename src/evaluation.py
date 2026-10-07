"""
Final evaluation of the frozen primary HDBSCAN clustering.

Primary configuration
---------------------
Trajectory representation:
    100 arc-length points

Trajectory distance:
    symmetric Hausdorff distance with Haversine point distance

HDBSCAN:
    min_samples = 15
    min_cluster_size = 30

This module consolidates the final clustering result into
quantitative evaluation artifacts.

It evaluates:

1. overall cluster/noise structure;
2. per-cluster membership and month composition;
3. within-cluster trajectory compactness;
4. representative-route medoid statistics;
5. inter-cluster medoid separation;
6. silhouette separation among clustered trajectories;
7. geometric separation of HDBSCAN noise from route medoids.

HDBSCAN noise is treated as anomaly candidates, not automatically
as confirmed operational anomalies.
"""

import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
)

from src.config import (
    DISTANCE_OUTPUT_DIR,
    HDBSCAN_OUTPUT_DIR,
)

# Paths
DISTANCE_MATRIX_PATH = (
    DISTANCE_OUTPUT_DIR
    / "hausdorff_100.npy"
)

LABELS_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "final"
    / "labels.csv"
)

MEDOIDS_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "representative_routes_k15_mcs30.csv"
)

OUTPUT_DIR = (
    HDBSCAN_OUTPUT_DIR
    / "final_evaluation"
)

# Loading and validation
def load_inputs():
    """
    Load and validate the frozen primary clustering artifacts.
    """

    distance_matrix = np.load(
        DISTANCE_MATRIX_PATH
    )

    labels = pd.read_csv(
        LABELS_PATH
    )

    medoids = pd.read_csv(
        MEDOIDS_PATH
    )

    labels = (
        labels
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        distance_matrix.shape[
            0
        ]
        != distance_matrix.shape[
            1
        ]
    ):
        raise RuntimeError(
            "Distance matrix is not square."
        )

    if (
        distance_matrix.shape[
            0
        ]
        != len(
            labels
        )
    ):
        raise RuntimeError(
            "Distance matrix and label table have "
            "different trajectory counts."
        )

    expected_indices = np.arange(
        len(
            labels
        )
    )

    actual_indices = (
        labels[
            "matrix_index"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    if not np.array_equal(
        expected_indices,
        actual_indices,
    ):
        raise RuntimeError(
            "labels.csv is not aligned with matrix ordering."
        )

    if not np.all(
        np.isfinite(
            distance_matrix
        )
    ):
        raise RuntimeError(
            "Distance matrix contains non-finite values."
        )

    if np.any(
        distance_matrix < 0
    ):
        raise RuntimeError(
            "Distance matrix contains negative values."
        )

    if not np.allclose(
        np.diag(
            distance_matrix
        ),
        0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Distance matrix diagonal is not zero."
        )

    if not np.allclose(
        distance_matrix,
        distance_matrix.T,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Distance matrix is not symmetric."
        )

    return (
        distance_matrix,
        labels,
        medoids,
    )

# Utility functions
def pairwise_statistics(
    matrix,
    indices,
):
    """
    Compute statistics for all unique trajectory pairs within
    one cluster.
    """

    indices = np.asarray(
        indices,
        dtype="int64",
    )

    if len(
        indices
    ) < 2:
        return {
            "pair_count": 0,
            "mean_km": np.nan,
            "median_km": np.nan,
            "p95_km": np.nan,
            "maximum_km": np.nan,
        }

    submatrix = (
        matrix[
            np.ix_(
                indices,
                indices,
            )
        ]
    )

    upper = (
        submatrix[
            np.triu_indices(
                len(
                    indices
                ),
                k=1,
            )
        ]
    )

    return {
        "pair_count": int(
            len(
                upper
            )
        ),

        "mean_km": float(
            np.mean(
                upper
            )
        ),

        "median_km": float(
            np.median(
                upper
            )
        ),

        "p95_km": float(
            np.percentile(
                upper,
                95,
            )
        ),

        "maximum_km": float(
            np.max(
                upper
            )
        ),
    }


def summarize_values(
    values,
):
    """
    Return standard descriptive statistics.
    """

    values = np.asarray(
        values,
        dtype="float64",
    )

    return {
        "minimum": float(
            np.min(
                values
            )
        ),

        "mean": float(
            np.mean(
                values
            )
        ),

        "median": float(
            np.median(
                values
            )
        ),

        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),

        "maximum": float(
            np.max(
                values
            )
        ),
    }

# Overall summary
def build_overall_summary(
    labels,
):
    """
    Summarize the final partition.
    """

    cluster_labels = (
        labels.loc[
            labels[
                "cluster_label"
            ]
            >= 0,
            "cluster_label",
        ]
        .sort_values()
        .unique()
    )

    noise_count = int(
        (
            labels[
                "cluster_label"
            ]
            == -1
        ).sum()
    )

    clustered_count = (
        len(
            labels
        )
        - noise_count
    )

    return {
        "trajectory_count": int(
            len(
                labels
            )
        ),

        "cluster_count": int(
            len(
                cluster_labels
            )
        ),

        "clustered_count": int(
            clustered_count
        ),

        "clustered_percent": float(
            clustered_count
            / len(
                labels
            )
            * 100.0
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

# Cluster-level evaluation
def build_cluster_summary(
    labels,
    medoids,
    distance_matrix,
):
    """
    Build the main report-ready cluster summary table.
    """

    records = []

    cluster_ids = sorted(
        labels.loc[
            labels[
                "cluster_label"
            ]
            >= 0,
            "cluster_label",
        ]
        .unique()
        .tolist()
    )

    for cluster_id in cluster_ids:

        members = (
            labels[
                labels[
                    "cluster_label"
                ]
                == cluster_id
            ]
        )

        member_indices = (
            members[
                "matrix_index"
            ]
            .to_numpy(
                dtype="int64"
            )
        )

        january_count = int(
            (
                members[
                    "month"
                ]
                == "2026-01"
            ).sum()
        )

        july_count = int(
            (
                members[
                    "month"
                ]
                == "2026-07"
            ).sum()
        )

        cluster_size = int(
            len(
                members
            )
        )

        within = (
            pairwise_statistics(
                distance_matrix,
                member_indices,
            )
        )

        medoid_rows = (
            medoids[
                medoids[
                    "cluster_label"
                ]
                == cluster_id
            ]
        )

        if len(
            medoid_rows
        ) != 1:
            raise RuntimeError(
                f"Expected exactly one medoid for cluster "
                f"{cluster_id}, found {len(medoid_rows)}."
            )

        medoid = (
            medoid_rows
            .iloc[
                0
            ]
        )

        medoid_index = int(
            medoid[
                "medoid_matrix_index"
            ]
        )

        if medoid_index not in set(
            member_indices.tolist()
        ):
            raise RuntimeError(
                f"Medoid {medoid_index} is not a member "
                f"of cluster {cluster_id}."
            )

        path_lengths = (
            members[
                "path_length_km"
            ]
            .to_numpy(
                dtype="float64"
            )
        )

        records.append(
            {
                "cluster_label": int(
                    cluster_id
                ),

                "cluster_size": (
                    cluster_size
                ),

                "dataset_percent": float(
                    cluster_size
                    / len(
                        labels
                    )
                    * 100.0
                ),

                "january_count": (
                    january_count
                ),

                "january_percent_within_cluster": float(
                    january_count
                    / cluster_size
                    * 100.0
                ),

                "july_count": (
                    july_count
                ),

                "july_percent_within_cluster": float(
                    july_count
                    / cluster_size
                    * 100.0
                ),

                "median_path_length_km": float(
                    np.median(
                        path_lengths
                    )
                ),

                "p95_path_length_km": float(
                    np.percentile(
                        path_lengths,
                        95,
                    )
                ),

                "within_pair_count": (
                    within[
                        "pair_count"
                    ]
                ),

                "within_mean_distance_km": (
                    within[
                        "mean_km"
                    ]
                ),

                "within_median_distance_km": (
                    within[
                        "median_km"
                    ]
                ),

                "within_p95_distance_km": (
                    within[
                        "p95_km"
                    ]
                ),

                "within_maximum_distance_km": (
                    within[
                        "maximum_km"
                    ]
                ),

                "medoid_matrix_index": (
                    medoid_index
                ),

                "medoid_month": (
                    medoid[
                        "medoid_month"
                    ]
                ),

                "medoid_trajectory_id": (
                    medoid[
                        "medoid_trajectory_id"
                    ]
                ),

                "medoid_mean_distance_km": float(
                    medoid[
                        "mean_distance_to_members_km"
                    ]
                ),

                "medoid_median_distance_km": float(
                    medoid[
                        "median_distance_to_members_km"
                    ]
                ),

                "medoid_p95_distance_km": float(
                    medoid[
                        "p95_distance_to_members_km"
                    ]
                ),

                "medoid_maximum_distance_km": float(
                    medoid[
                        "maximum_distance_to_members_km"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        records
    )

# Month composition
def build_month_cluster_table(
    labels,
):
    """
    Count all cluster/noise labels separately for January and July.
    """

    table = pd.crosstab(
        labels[
            "cluster_label"
        ],
        labels[
            "month"
        ],
        margins=True,
    )

    return table

# Inter-cluster medoid separation
def build_medoid_distance_matrix(
    medoids,
    distance_matrix,
):
    """
    Build pairwise distance matrix between the four cluster medoids.
    """

    medoids_sorted = (
        medoids
        .sort_values(
            "cluster_label"
        )
        .reset_index(
            drop=True
        )
    )

    cluster_labels = (
        medoids_sorted[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    medoid_indices = (
        medoids_sorted[
            "medoid_matrix_index"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    distances = (
        distance_matrix[
            np.ix_(
                medoid_indices,
                medoid_indices,
            )
        ]
    )

    column_names = [
        f"cluster_{cluster_id}"
        for cluster_id in cluster_labels
    ]

    table = pd.DataFrame(
        distances,
        index=[
            f"cluster_{cluster_id}"
            for cluster_id in cluster_labels
        ],
        columns=(
            column_names
        ),
    )

    table.index.name = (
        "medoid_cluster"
    )

    return (
        table,
        cluster_labels,
        distances,
    )


def build_medoid_separation_summary(
    cluster_labels,
    distances,
):
    """
    Find the closest other representative route for every cluster.
    """

    records = []

    for row_position, cluster_id in enumerate(
        cluster_labels
    ):

        row = (
            distances[
                row_position
            ]
            .copy()
        )

        row[
            row_position
        ] = np.inf

        closest_position = int(
            np.argmin(
                row
            )
        )

        records.append(
            {
                "cluster_label": int(
                    cluster_id
                ),

                "closest_other_cluster": int(
                    cluster_labels[
                        closest_position
                    ]
                ),

                "closest_medoid_distance_km": float(
                    row[
                        closest_position
                    ]
                ),
            }
        )

    return pd.DataFrame(
        records
    )

# Silhouette evaluation
def evaluate_silhouette(
    labels,
    distance_matrix,
):
    """
    Evaluate route-cluster separation using silhouette scores.

    Noise trajectories are excluded because silhouette analysis
    evaluates the four selected route clusters, not the HDBSCAN
    noise population.
    """

    clustered_mask = (
        labels[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
        >= 0
    )

    clustered_indices = np.flatnonzero(
        clustered_mask
    )

    clustered_labels = (
        labels.loc[
            clustered_mask,
            "cluster_label",
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    clustered_distances = (
        distance_matrix[
            np.ix_(
                clustered_indices,
                clustered_indices,
            )
        ]
        .copy()
    )

    np.fill_diagonal(
        clustered_distances,
        0.0,
    )

    overall_score = float(
        silhouette_score(
            clustered_distances,
            clustered_labels,
            metric="precomputed",
        )
    )

    sample_scores = (
        silhouette_samples(
            clustered_distances,
            clustered_labels,
            metric="precomputed",
        )
    )

    sample_table = (
        labels.loc[
            clustered_mask,
            [
                "matrix_index",
                "month",
                "trajectory_id",
                "cluster_label",
            ],
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    sample_table[
        "silhouette_score"
    ] = (
        sample_scores
    )

    records = []

    for cluster_id in sorted(
        np.unique(
            clustered_labels
        )
    ):

        values = (
            sample_scores[
                clustered_labels
                == cluster_id
            ]
        )

        records.append(
            {
                "cluster_label": int(
                    cluster_id
                ),

                "member_count": int(
                    len(
                        values
                    )
                ),

                "mean_silhouette": float(
                    np.mean(
                        values
                    )
                ),

                "median_silhouette": float(
                    np.median(
                        values
                    )
                ),

                "minimum_silhouette": float(
                    np.min(
                        values
                    )
                ),

                "p05_silhouette": float(
                    np.percentile(
                        values,
                        5,
                    )
                ),

                "negative_silhouette_count": int(
                    (
                        values < 0
                    ).sum()
                ),

                "negative_silhouette_percent": float(
                    (
                        values < 0
                    ).mean()
                    * 100.0
                ),
            }
        )

    summary_table = pd.DataFrame(
        records
    )

    return (
        overall_score,
        summary_table,
        sample_table,
    )

# Noise geometry
def evaluate_noise_geometry(
    labels,
    medoids,
    distance_matrix,
):
    """
    Characterize the geometric separation of HDBSCAN noise
    trajectories from the four representative route medoids.
    """

    noise = (
        labels[
            labels[
                "cluster_label"
            ]
            == -1
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    noise_indices = (
        noise[
            "matrix_index"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    medoids_sorted = (
        medoids
        .sort_values(
            "cluster_label"
        )
        .reset_index(
            drop=True
        )
    )

    medoid_indices = (
        medoids_sorted[
            "medoid_matrix_index"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    medoid_labels = (
        medoids_sorted[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    noise_to_medoids = (
        distance_matrix[
            np.ix_(
                noise_indices,
                medoid_indices,
            )
        ]
    )

    nearest_positions = np.argmin(
        noise_to_medoids,
        axis=1,
    )

    nearest_distances = (
        noise_to_medoids[
            np.arange(
                len(
                    noise
                )
            ),
            nearest_positions,
        ]
    )

    nearest_cluster_labels = (
        medoid_labels[
            nearest_positions
        ]
    )

    result = (
        noise[
            [
                "matrix_index",
                "month",
                "trajectory_id",
                "file_name",
                "path_length_km",
            ]
        ]
        .copy()
    )

    result[
        "nearest_medoid_cluster"
    ] = (
        nearest_cluster_labels
    )

    result[
        "nearest_medoid_distance_km"
    ] = (
        nearest_distances
    )

    summary = summarize_values(
        nearest_distances
    )

    summary[
        "noise_count"
    ] = int(
        len(
            noise
        )
    )

    return (
        result,
        summary,
    )

# Main
def main():

    print(
        "=" * 100
    )

    print(
        "FINAL PRIMARY CLUSTER EVALUATION"
    )

    print(
        "=" * 100
    )

    (
        distance_matrix,
        labels,
        medoids,
    ) = load_inputs()

    print(
        "\nInput validation: PASS"
    )

    print(
        "Distance matrix:",
        distance_matrix.shape,
    )

    print(
        "Trajectories:",
        len(
            labels
        ),
    )
    # Overall structure
    overall = (
        build_overall_summary(
            labels
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "OVERALL PARTITION"
    )

    print(
        "=" * 100
    )

    print(
        "\nClusters:",
        overall[
            "cluster_count"
        ],
    )

    print(
        "Clustered trajectories:",
        (
            f"{overall['clustered_count']} "
            f"({overall['clustered_percent']:.3f}%)"
        ),
    )

    print(
        "Noise / anomaly candidates:",
        (
            f"{overall['noise_count']} "
            f"({overall['noise_percent']:.3f}%)"
        ),
    )
    # Per-cluster summary
    cluster_summary = (
        build_cluster_summary(
            labels,
            medoids,
            distance_matrix,
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CLUSTER SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        "\n",
        cluster_summary[
            [
                "cluster_label",
                "cluster_size",
                "january_count",
                "july_count",
                "within_mean_distance_km",
                "within_median_distance_km",
                "within_p95_distance_km",
                "medoid_trajectory_id",
            ]
        ]
        .to_string(
            index=False
        ),
        sep="",
    )
    # Month composition
    month_table = (
        build_month_cluster_table(
            labels
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MONTH × CLUSTER MEMBERSHIP"
    )

    print(
        "=" * 100
    )

    print(
        "\n",
        month_table.to_string(),
        sep="",
    )
    # Medoid separation
    (
        medoid_matrix,
        medoid_cluster_labels,
        medoid_distances,
    ) = build_medoid_distance_matrix(
        medoids,
        distance_matrix,
    )

    medoid_separation = (
        build_medoid_separation_summary(
            medoid_cluster_labels,
            medoid_distances,
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "INTER-CLUSTER MEDOID DISTANCES (KM)"
    )

    print(
        "=" * 100
    )

    print(
        "\n",
        medoid_matrix.to_string(
            float_format=lambda value: (
                f"{value:.3f}"
            )
        ),
        sep="",
    )

    print(
        "\nClosest representative route for each cluster:"
    )

    print(
        medoid_separation.to_string(
            index=False
        )
    )
    # Silhouette
    (
        silhouette_overall,
        silhouette_summary,
        silhouette_samples_table,
    ) = evaluate_silhouette(
        labels,
        distance_matrix,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SILHOUETTE EVALUATION — CLUSTERED TRAJECTORIES ONLY"
    )

    print(
        "=" * 100
    )

    print(
        "\nOverall silhouette:",
        f"{silhouette_overall:.6f}",
    )

    print(
        "\n",
        silhouette_summary.to_string(
            index=False
        ),
        sep="",
    )
    # Noise geometry
    (
        noise_geometry,
        noise_summary,
    ) = evaluate_noise_geometry(
        labels,
        medoids,
        distance_matrix,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "NOISE GEOMETRY RELATIVE TO ROUTE MEDOIDS"
    )

    print(
        "=" * 100
    )

    print(
        "\nNoise trajectories:",
        noise_summary[
            "noise_count"
        ],
    )

    print(
        "Nearest-medoid distance:"
    )

    print(
        (
            f"  minimum: "
            f"{noise_summary['minimum']:.3f} km"
        )
    )

    print(
        (
            f"  mean:    "
            f"{noise_summary['mean']:.3f} km"
        )
    )

    print(
        (
            f"  median:  "
            f"{noise_summary['median']:.3f} km"
        )
    )

    print(
        (
            f"  p95:     "
            f"{noise_summary['p95']:.3f} km"
        )
    )

    print(
        (
            f"  maximum: "
            f"{noise_summary['maximum']:.3f} km"
        )
    )
    # Save
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cluster_summary.to_csv(
        OUTPUT_DIR
        / "cluster_summary.csv",
        index=False,
    )

    month_table.to_csv(
        OUTPUT_DIR
        / "month_cluster_membership.csv"
    )

    medoid_matrix.to_csv(
        OUTPUT_DIR
        / "intercluster_medoid_distances.csv"
    )

    medoid_separation.to_csv(
        OUTPUT_DIR
        / "medoid_separation_summary.csv",
        index=False,
    )

    silhouette_summary.to_csv(
        OUTPUT_DIR
        / "silhouette_by_cluster.csv",
        index=False,
    )

    silhouette_samples_table.to_csv(
        OUTPUT_DIR
        / "silhouette_samples.csv",
        index=False,
    )

    noise_geometry.to_csv(
        OUTPUT_DIR
        / "noise_nearest_medoid.csv",
        index=False,
    )

    report = {
        "primary_configuration": {
            "representation_points": 100,
            "trajectory_metric": (
                "symmetric_hausdorff_haversine"
            ),
            "min_samples": 15,
            "min_cluster_size": 30,
        },

        "overall_partition": (
            overall
        ),

        "overall_silhouette_clustered_only": (
            silhouette_overall
        ),

        "noise_nearest_medoid_distance_km": (
            noise_summary
        ),

        "interpretation_notes": [
            (
                "Silhouette is computed only among the four "
                "selected route clusters; HDBSCAN noise is excluded."
            ),
            (
                "Silhouette is complementary evidence and is not "
                "used to select or tune the final HDBSCAN solution."
            ),
            (
                "HDBSCAN noise trajectories are anomaly candidates, "
                "not automatically confirmed operational anomalies."
            ),
            (
                "Month composition is descriptive and does not "
                "establish a seasonal causal effect."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SAVED FINAL EVALUATION ARTIFACTS"
    )

    print(
        "=" * 100
    )

    print(
        "\n",
        OUTPUT_DIR
        / "cluster_summary.csv",
        sep="",
    )

    print(
        OUTPUT_DIR
        / "month_cluster_membership.csv"
    )

    print(
        OUTPUT_DIR
        / "intercluster_medoid_distances.csv"
    )

    print(
        OUTPUT_DIR
        / "medoid_separation_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "silhouette_by_cluster.csv"
    )

    print(
        OUTPUT_DIR
        / "silhouette_samples.csv"
    )

    print(
        OUTPUT_DIR
        / "noise_nearest_medoid.csv"
    )

    print(
        OUTPUT_DIR
        / "evaluation_summary.json"
    )

    print(
        "\nFINAL CLUSTER EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()