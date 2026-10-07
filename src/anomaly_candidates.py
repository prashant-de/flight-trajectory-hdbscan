"""
Characterize HDBSCAN noise trajectories as anomaly candidates.

Primary clustering
------------------
min_samples = 15
min_cluster_size = 30

The primary HDBSCAN result contains 295 noise trajectories.

This script does NOT assume that every noise trajectory is a true
operational anomaly.

Instead, it characterizes each primary-noise trajectory using:

1. noise persistence across nearby HDBSCAN configurations:
       k=10, mcs=30
       k=15, mcs=30
       k=20, mcs=30

2. k=15 core distance;

3. Hausdorff distance to each primary cluster medoid;

4. distance to the nearest route medoid.

No arbitrary weighted anomaly score is used.

Candidates are ranked lexicographically by:

    noise persistence
        ↓
    nearest-medoid distance
        ↓
    core distance

This preserves the individual evidence components explicitly.
"""

import json

import numpy as np
import pandas as pd

from src.config import (
    DISTANCE_OUTPUT_DIR,
    HDBSCAN_OUTPUT_DIR,
)

# Paths
FINAL_LABELS_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "final"
    / "labels.csv"
)

LABELS_K10_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "labels_k10_mcs30.csv"
)

LABELS_K20_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "labels_k20_mcs30.csv"
)

CORE_DISTANCE_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "final"
    / "core_distances.npy"
)

MEDOID_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "representative_routes_k15_mcs30.csv"
)

DISTANCE_MATRIX_PATH = (
    DISTANCE_OUTPUT_DIR
    / "hausdorff_100.npy"
)

OUTPUT_DIR = (
    HDBSCAN_OUTPUT_DIR
    / "anomaly_analysis"
)

# Loading helpers
def load_label_file(
    path,
):
    """
    Load a clustering-label CSV and ensure matrix ordering.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing label file: {path}"
        )

    table = pd.read_csv(
        path
    )

    required_columns = {
        "matrix_index",
        "cluster_label",
    }

    missing = (
        required_columns
        - set(
            table.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    table = (
        table
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    expected = np.arange(
        len(
            table
        )
    )

    actual = (
        table[
            "matrix_index"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    if not np.array_equal(
        expected,
        actual,
    ):
        raise RuntimeError(
            f"Non-sequential matrix indices in {path.name}"
        )

    return table


def load_inputs():
    """
    Load clustering, sensitivity, distance and medoid artifacts.
    """

    primary = load_label_file(
        FINAL_LABELS_PATH
    )

    labels_k10 = load_label_file(
        LABELS_K10_PATH
    )

    labels_k20 = load_label_file(
        LABELS_K20_PATH
    )

    if not (
        len(primary)
        == len(labels_k10)
        == len(labels_k20)
    ):
        raise RuntimeError(
            "Sensitivity label files have different sizes."
        )

    common_indices = (
        primary[
            "matrix_index"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    for name, table in [
        (
            "k10",
            labels_k10,
        ),
        (
            "k20",
            labels_k20,
        ),
    ]:

        if not np.array_equal(
            common_indices,
            table[
                "matrix_index"
            ].to_numpy(
                dtype="int64"
            ),
        ):
            raise RuntimeError(
                f"Trajectory ordering differs for {name}."
            )

    if not DISTANCE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Missing distance matrix: {DISTANCE_MATRIX_PATH}"
        )

    distance_matrix = np.load(
        DISTANCE_MATRIX_PATH
    )

    if (
        distance_matrix.shape[0]
        != len(
            primary
        )
    ):
        raise RuntimeError(
            "Distance matrix and label table size differ."
        )

    if not CORE_DISTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing core-distance file: {CORE_DISTANCE_PATH}"
        )

    core_distances = np.load(
        CORE_DISTANCE_PATH
    )

    if (
        len(
            core_distances
        )
        != len(
            primary
        )
    ):
        raise RuntimeError(
            "Core distances and label table size differ."
        )

    if not MEDOID_PATH.exists():
        raise FileNotFoundError(
            f"Missing representative-route file: {MEDOID_PATH}"
        )

    medoids = pd.read_csv(
        MEDOID_PATH
    )

    required_medoid_columns = {
        "cluster_label",
        "medoid_matrix_index",
    }

    missing_medoid_columns = (
        required_medoid_columns
        - set(
            medoids.columns
        )
    )

    if missing_medoid_columns:
        raise RuntimeError(
            "Representative-route table is missing columns: "
            f"{sorted(missing_medoid_columns)}"
        )

    medoids = (
        medoids
        .sort_values(
            "cluster_label"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        primary,
        labels_k10,
        labels_k20,
        distance_matrix,
        core_distances,
        medoids,
    )

# Candidate characterization
def build_anomaly_candidate_table(
    primary,
    labels_k10,
    labels_k20,
    distance_matrix,
    core_distances,
    medoids,
):
    """
    Characterize all trajectories classified as noise by the
    primary k=15, mcs=30 HDBSCAN configuration.
    """

    primary_labels = (
        primary[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    noise_mask = (
        primary_labels == -1
    )

    candidate_indices = (
        primary.loc[
            noise_mask,
            "matrix_index",
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    candidates = (
        primary.loc[
            noise_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        candidates
    ) != 295:
        raise RuntimeError(
            "Expected 295 primary anomaly candidates, "
            f"found {len(candidates)}."
        )
    # Sensitivity / persistence
    k10_labels = (
        labels_k10[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    k20_labels = (
        labels_k20[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    candidates[
        "noise_k10_mcs30"
    ] = (
        k10_labels[
            candidate_indices
        ]
        == -1
    )

    # Every candidate is primary k=15 noise by construction.
    candidates[
        "noise_k15_mcs30"
    ] = True

    candidates[
        "noise_k20_mcs30"
    ] = (
        k20_labels[
            candidate_indices
        ]
        == -1
    )

    candidates[
        "persistent_noise_count"
    ] = (
        candidates[
            [
                "noise_k10_mcs30",
                "noise_k15_mcs30",
                "noise_k20_mcs30",
            ]
        ]
        .sum(
            axis=1
        )
        .astype(
            "int64"
        )
    )

    candidates[
        "persistent_noise_all_three"
    ] = (
        candidates[
            "persistent_noise_count"
        ]
        == 3
    )

    def persistence_category(
        count,
    ):
        if count == 3:
            return "noise_in_all_3"

        if count == 2:
            return "noise_in_2_of_3"

        return "noise_only_primary"

    candidates[
        "persistence_category"
    ] = (
        candidates[
            "persistent_noise_count"
        ]
        .map(
            persistence_category
        )
    )
    # Core-distance evidence
    candidates[
        "core_distance_k15_km"
    ] = (
        core_distances[
            candidate_indices
        ]
    )
    # Distance from each candidate to every primary route medoid
    cluster_labels = (
        medoids[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    medoid_indices = (
        medoids[
            "medoid_matrix_index"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    candidate_medoid_distances = (
        distance_matrix[
            np.ix_(
                candidate_indices,
                medoid_indices,
            )
        ]
    )

    for column_position, cluster_label in enumerate(
        cluster_labels
    ):

        candidates[
            f"distance_to_cluster_{cluster_label}_medoid_km"
        ] = (
            candidate_medoid_distances[
                :,
                column_position
            ]
        )

    nearest_positions = np.argmin(
        candidate_medoid_distances,
        axis=1,
    )

    candidates[
        "nearest_medoid_cluster"
    ] = (
        cluster_labels[
            nearest_positions
        ]
    )

    candidates[
        "nearest_medoid_matrix_index"
    ] = (
        medoid_indices[
            nearest_positions
        ]
    )

    candidates[
        "nearest_medoid_distance_km"
    ] = (
        candidate_medoid_distances[
            np.arange(
                len(
                    candidates
                )
            ),
            nearest_positions,
        ]
    )
    # Transparent ranking
    #
    # No weighted composite score:
    # 1. more persistent noise first
    # 2. farther from nearest medoid
    # 3. larger core distance
    candidates = (
        candidates
        .sort_values(
            [
                "persistent_noise_count",
                "nearest_medoid_distance_km",
                "core_distance_k15_km",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    candidates.insert(
        0,
        "anomaly_candidate_rank",
        np.arange(
            1,
            len(
                candidates
            )
            + 1,
        ),
    )

    return candidates

# Summary
def build_summary(
    candidates,
):
    """
    Produce descriptive anomaly-candidate summary.
    """

    persistence_counts = (
        candidates[
            "persistent_noise_count"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    persistence_counts = {
        str(
            int(
                key
            )
        ): int(
            value
        )
        for key, value
        in persistence_counts.items()
    }

    nearest_distance = (
        candidates[
            "nearest_medoid_distance_km"
        ]
        .to_numpy(
            dtype="float64"
        )
    )

    core_distance = (
        candidates[
            "core_distance_k15_km"
        ]
        .to_numpy(
            dtype="float64"
        )
    )

    return {
        "primary_configuration": {
            "min_samples": 15,
            "min_cluster_size": 30,
        },

        "primary_noise_candidate_count": (
            int(
                len(
                    candidates
                )
            )
        ),

        "persistence_configs": [
            "k10_mcs30",
            "k15_mcs30",
            "k20_mcs30",
        ],

        "persistence_count_distribution": (
            persistence_counts
        ),

        "persistent_noise_all_three_count": (
            int(
                candidates[
                    "persistent_noise_all_three"
                ]
                .sum()
            )
        ),

        "nearest_medoid_distance_km": {
            "minimum": float(
                np.min(
                    nearest_distance
                )
            ),

            "median": float(
                np.median(
                    nearest_distance
                )
            ),

            "p95": float(
                np.percentile(
                    nearest_distance,
                    95,
                )
            ),

            "maximum": float(
                np.max(
                    nearest_distance
                )
            ),
        },

        "core_distance_k15_km": {
            "minimum": float(
                np.min(
                    core_distance
                )
            ),

            "median": float(
                np.median(
                    core_distance
                )
            ),

            "p95": float(
                np.percentile(
                    core_distance,
                    95,
                )
            ),

            "maximum": float(
                np.max(
                    core_distance
                )
            ),
        },
    }

# Main
def main():

    print(
        "=" * 100
    )

    print(
        "HDBSCAN ANOMALY-CANDIDATE CHARACTERIZATION"
    )

    print(
        "=" * 100
    )

    (
        primary,
        labels_k10,
        labels_k20,
        distance_matrix,
        core_distances,
        medoids,
    ) = load_inputs()

    candidates = (
        build_anomaly_candidate_table(
            primary,
            labels_k10,
            labels_k20,
            distance_matrix,
            core_distances,
            medoids,
        )
    )

    summary = build_summary(
        candidates
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        OUTPUT_DIR
        / "anomaly_candidates_k15_mcs30.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "anomaly_candidate_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "\nPrimary HDBSCAN noise candidates:",
        summary[
            "primary_noise_candidate_count"
        ],
    )

    print(
        "\nNoise persistence across "
        "k=10, k=15, k=20 with mcs=30:"
    )

    for persistence_count in [
        1,
        2,
        3,
    ]:

        count = (
            summary[
                "persistence_count_distribution"
            ]
            .get(
                str(
                    persistence_count
                ),
                0,
            )
        )

        print(
            f"  noise in {persistence_count}/3 configs: "
            f"{count}"
        )

    print(
        "\nPersistent noise in all three:",
        summary[
            "persistent_noise_all_three_count"
        ],
    )

    nearest_stats = (
        summary[
            "nearest_medoid_distance_km"
        ]
    )

    print(
        "\nNearest-medoid Hausdorff distance:"
    )

    print(
        f"  minimum: "
        f"{nearest_stats['minimum']:.3f} km"
    )

    print(
        f"  median:  "
        f"{nearest_stats['median']:.3f} km"
    )

    print(
        f"  p95:     "
        f"{nearest_stats['p95']:.3f} km"
    )

    print(
        f"  maximum: "
        f"{nearest_stats['maximum']:.3f} km"
    )

    core_stats = (
        summary[
            "core_distance_k15_km"
        ]
    )

    print(
        "\nCore-distance distribution among candidates:"
    )

    print(
        f"  minimum: "
        f"{core_stats['minimum']:.3f} km"
    )

    print(
        f"  median:  "
        f"{core_stats['median']:.3f} km"
    )

    print(
        f"  p95:     "
        f"{core_stats['p95']:.3f} km"
    )

    print(
        f"  maximum: "
        f"{core_stats['maximum']:.3f} km"
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TOP 20 ANOMALY CANDIDATES"
    )

    print(
        "=" * 100
    )

    display_columns = [
        "anomaly_candidate_rank",
        "matrix_index",
        "month",
        "trajectory_id",
        "persistent_noise_count",
        "nearest_medoid_cluster",
        "nearest_medoid_distance_km",
        "core_distance_k15_km",
    ]

    print(
        "\n",
        candidates[
            display_columns
        ]
        .head(
            20
        )
        .to_string(
            index=False
        ),
        sep="",
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_DIR
        / "anomaly_candidates_k15_mcs30.csv"
    )

    print(
        OUTPUT_DIR
        / "anomaly_candidate_summary.json"
    )


if __name__ == "__main__":
    main()