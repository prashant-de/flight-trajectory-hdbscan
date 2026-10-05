import json
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import (
    DISTANCE_OUTPUT_DIR,
    PROCESSED_JANUARY_DIR,
    PROCESSED_JULY_DIR,
)

from src.distances import (
    EARTH_RADIUS_KM,
    build_arc_length_representation,
)


REPRESENTATION_POINTS = 100
METRIC_NAME = "symmetric_hausdorff_haversine"


# Dataset discovery

def discover_trajectories():
    """
    Build the deterministic trajectory ordering that will define
    every row/column in the distance matrix.
    """

    records = []

    january_files = sorted(
        PROCESSED_JANUARY_DIR.glob(
            "*.csv"
        )
    )

    july_files = sorted(
        PROCESSED_JULY_DIR.glob(
            "*.csv"
        )
    )

    for file_path in january_files:
        records.append(
            {
                "month": "2026-01",
                "trajectory_id": (
                    file_path.stem
                ),
                "file_name": (
                    file_path.name
                ),
                "file_path": (
                    file_path
                ),
            }
        )

    for file_path in july_files:
        records.append(
            {
                "month": "2026-07",
                "trajectory_id": (
                    file_path.stem
                ),
                "file_name": (
                    file_path.name
                ),
                "file_path": (
                    file_path
                ),
            }
        )

    return records


# Representation construction

def build_all_representations(
    trajectory_records,
):
    """
    Build the selected 100-point arc-length representation
    for every processed trajectory.

    Returns:
        representations:
            shape = (N, 100, 2)
            coordinate order = [lat, lon]

        metadata:
            per-trajectory information aligned with matrix indices
    """

    n = len(
        trajectory_records
    )

    representations = np.empty(
        (
            n,
            REPRESENTATION_POINTS,
            2,
        ),
        dtype="float64",
    )

    metadata = []

    print(
        "Building 100-point representations..."
    )

    for index, record in enumerate(
        trajectory_records
    ):
        trajectory = pd.read_csv(
            record["file_path"],
            usecols=[
                "lat",
                "lon",
            ],
        )

        (
            representation,
            _,
            path_length_km,
        ) = build_arc_length_representation(
            trajectory,
            REPRESENTATION_POINTS,
        )

        representations[
            index,
            :,
            0,
        ] = (
            representation["lat"]
            .to_numpy(
                dtype="float64",
                copy=False,
            )
        )

        representations[
            index,
            :,
            1,
        ] = (
            representation["lon"]
            .to_numpy(
                dtype="float64",
                copy=False,
            )
        )

        metadata.append(
            {
                "matrix_index": (
                    index
                ),

                "month": (
                    record["month"]
                ),

                "trajectory_id": (
                    record[
                        "trajectory_id"
                    ]
                ),

                "file_name": (
                    record[
                        "file_name"
                    ]
                ),

                "processed_rows": (
                    len(
                        trajectory
                    )
                ),

                "path_length_km": (
                    path_length_km
                ),
            }
        )

        if (
            (index + 1) % 100 == 0
            or index + 1 == n
        ):
            print(
                f"Representations: "
                f"{index + 1} / {n}",
                flush=True,
            )

    return (
        representations,
        pd.DataFrame(
            metadata
        ),
    )


# Optimized Hausdorff calculation

def hausdorff_radians_km(
    trajectory_a,
    trajectory_b,
):
    """
    Symmetric Hausdorff distance for two trajectory arrays
    already stored in radians.

    trajectory shape:
        (100, 2)

    coordinate order:
        [:, 0] = latitude
        [:, 1] = longitude
    """

    lat_a = (
        trajectory_a[
            :,
            0,
        ][
            :,
            None,
        ]
    )

    lon_a = (
        trajectory_a[
            :,
            1,
        ][
            :,
            None,
        ]
    )

    lat_b = (
        trajectory_b[
            :,
            0,
        ][
            None,
            :,
        ]
    )

    lon_b = (
        trajectory_b[
            :,
            1,
        ][
            None,
            :,
        ]
    )

    dlat = (
        lat_b - lat_a
    )

    dlon = (
        lon_b - lon_a
    )

    a = (
        np.sin(
            dlat / 2
        ) ** 2
        + np.cos(
            lat_a
        )
        * np.cos(
            lat_b
        )
        * np.sin(
            dlon / 2
        ) ** 2
    )

    a = np.clip(
        a,
        0,
        1,
    )

    angular_distance = (
        2
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(
                1 - a
            ),
        )
    )

    distance_matrix = (
        EARTH_RADIUS_KM
        * angular_distance
    )

    directed_a_to_b = (
        distance_matrix
        .min(
            axis=1
        )
        .max()
    )

    directed_b_to_a = (
        distance_matrix
        .min(
            axis=0
        )
        .max()
    )

    return max(
        directed_a_to_b,
        directed_b_to_a,
    )


# Full matrix

def build_full_distance_matrix(
    representations,
):
    """
    Build the complete symmetric Hausdorff distance matrix.
    """

    n = representations.shape[
        0
    ]

    matrix = np.zeros(
        (
            n,
            n,
        ),
        dtype="float64",
    )

    # Convert once rather than converting coordinates
    # during every pairwise comparison.
    representations_rad = (
        np.radians(
            representations
        )
    )

    total_pairs = (
        n
        * (
            n - 1
        )
        // 2
    )

    completed_pairs = 0
    report_every = 50000
    next_report = report_every

    print(
        "\nComputing full Hausdorff matrix..."
    )

    print(
        f"Trajectories: {n}"
    )

    print(
        f"Unique pairs: {total_pairs:,}"
    )

    start = perf_counter()

    for i in range(
        n
    ):
        trajectory_a = (
            representations_rad[i]
        )

        for j in range(
            i + 1,
            n,
        ):
            distance = (
                hausdorff_radians_km(
                    trajectory_a,
                    representations_rad[
                        j
                    ],
                )
            )

            matrix[
                i,
                j,
            ] = distance

            matrix[
                j,
                i,
            ] = distance

            completed_pairs += 1

            if (
                completed_pairs
                >= next_report
            ):
                percent = (
                    completed_pairs
                    / total_pairs
                    * 100
                )

                print(
                    f"Computed "
                    f"{completed_pairs:,} / "
                    f"{total_pairs:,} pairs "
                    f"({percent:.1f}%)",
                    flush=True,
                )

                next_report += (
                    report_every
                )

    elapsed = (
        perf_counter()
        - start
    )

    return matrix, elapsed


# Validation

def validate_distance_matrix(
    matrix,
):
    """
    Verify properties required by the later HDBSCAN stages.
    """

    problems = []

    if (
        matrix.ndim != 2
        or matrix.shape[0]
        != matrix.shape[1]
    ):
        problems.append(
            "matrix_not_square"
        )

        return problems

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        problems.append(
            "nonfinite_distances"
        )

    if np.any(
        matrix < 0
    ):
        problems.append(
            "negative_distances"
        )

    if not np.allclose(
        np.diag(
            matrix
        ),
        0.0,
        atol=1e-12,
    ):
        problems.append(
            "nonzero_diagonal"
        )

    if not np.allclose(
        matrix,
        matrix.T,
        atol=1e-12,
    ):
        problems.append(
            "matrix_not_symmetric"
        )

    return problems


# Save

def save_outputs(
    representations,
    trajectory_index,
    matrix,
    elapsed_seconds,
):
    """
    Save synchronized Module 3 distance artifacts.
    """

    DISTANCE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    representation_path = (
        DISTANCE_OUTPUT_DIR
        / "trajectory_representations_100.npy"
    )

    index_path = (
        DISTANCE_OUTPUT_DIR
        / "trajectory_index.csv"
    )

    matrix_path = (
        DISTANCE_OUTPUT_DIR
        / "hausdorff_100.npy"
    )

    metadata_path = (
        DISTANCE_OUTPUT_DIR
        / "hausdorff_100_metadata.json"
    )

    np.save(
        representation_path,
        representations,
    )

    trajectory_index.to_csv(
        index_path,
        index=False,
    )

    np.save(
        matrix_path,
        matrix,
    )

    n = matrix.shape[
        0
    ]

    upper = np.triu_indices(
        n,
        k=1,
    )

    pairwise_values = (
        matrix[
            upper
        ]
    )

    metadata = {
        "metric": (
            METRIC_NAME
        ),

        "representation_points": (
            REPRESENTATION_POINTS
        ),

        "trajectory_count": (
            int(
                n
            )
        ),

        "unique_pair_count": (
            int(
                len(
                    pairwise_values
                )
            )
        ),

        "distance_unit": "km",

        "matrix_shape": [
            int(
                matrix.shape[0]
            ),
            int(
                matrix.shape[1]
            ),
        ],

        "median_pairwise_distance_km": (
            float(
                np.median(
                    pairwise_values
                )
            )
        ),

        "p95_pairwise_distance_km": (
            float(
                np.percentile(
                    pairwise_values,
                    95,
                )
            )
        ),

        "maximum_pairwise_distance_km": (
            float(
                np.max(
                    pairwise_values
                )
            )
        ),

        "calculation_seconds": (
            float(
                elapsed_seconds
            )
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    return metadata


# Main

def main():
    records = (
        discover_trajectories()
    )

    print(
        "Discovered trajectories:",
        len(
            records
        ),
    )

    (
        representations,
        trajectory_index,
    ) = build_all_representations(
        records
    )

    # Save representations/index before the expensive pairwise stage.
    DISTANCE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        DISTANCE_OUTPUT_DIR
        / "trajectory_representations_100.npy",
        representations,
    )

    trajectory_index.to_csv(
        DISTANCE_OUTPUT_DIR
        / "trajectory_index.csv",
        index=False,
    )

    matrix, elapsed = (
        build_full_distance_matrix(
            representations
        )
    )

    problems = (
        validate_distance_matrix(
            matrix
        )
    )

    if problems:
        raise RuntimeError(
            "Distance matrix validation failed: "
            + ", ".join(
                problems
            )
        )

    metadata = save_outputs(
        representations,
        trajectory_index,
        matrix,
        elapsed,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FULL HAUSDORFF DISTANCE MATRIX"
    )

    print(
        "=" * 70
    )

    print(
        "\nMatrix shape:"
    )
    print(
        tuple(
            metadata[
                "matrix_shape"
            ]
        )
    )

    print(
        "\nTrajectory count:"
    )
    print(
        metadata[
            "trajectory_count"
        ]
    )

    print(
        "\nUnique trajectory pairs:"
    )
    print(
        f"{metadata['unique_pair_count']:,}"
    )

    print(
        "\nPairwise Hausdorff distance:"
    )

    print(
        "median:",
        f"{metadata['median_pairwise_distance_km']:.3f} km"
    )

    print(
        "p95:",
        f"{metadata['p95_pairwise_distance_km']:.3f} km"
    )

    print(
        "maximum:",
        f"{metadata['maximum_pairwise_distance_km']:.3f} km"
    )

    print(
        "\nMatrix validation:"
    )
    print(
        "PASS"
    )

    print(
        "\nSaved:"
    )

    print(
        "outputs/distances/"
        "trajectory_representations_100.npy"
    )

    print(
        "outputs/distances/"
        "trajectory_index.csv"
    )

    print(
        "outputs/distances/"
        "hausdorff_100.npy"
    )

    print(
        "outputs/distances/"
        "hausdorff_100_metadata.json"
    )


if __name__ == "__main__":
    main()