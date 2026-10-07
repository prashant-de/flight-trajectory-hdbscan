"""
Build the 200-point trajectory representation and full Hausdorff
distance matrix for representation-sensitivity analysis.

This script does NOT modify or replace the primary 100-point
distance artifacts.

Primary configuration:
    100 arc-length points

Sensitivity configuration:
    200 arc-length points

The same processed trajectories, ordering, geographic distance
definition, and Hausdorff implementation are used.
"""

import json
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import (
    DISTANCE_OUTPUT_DIR,
)

from src.distances import (
    build_arc_length_representation,
)

from src.build_distance_matrix import (
    discover_trajectories,
    build_full_distance_matrix,
    validate_distance_matrix,
)


# Sensitivity configuration
REPRESENTATION_POINTS = 200

SENSITIVITY_OUTPUT_DIR = (
    DISTANCE_OUTPUT_DIR
    / "sensitivity"
)

PRIMARY_INDEX_PATH = (
    DISTANCE_OUTPUT_DIR
    / "trajectory_index.csv"
)

REPRESENTATION_OUTPUT_PATH = (
    SENSITIVITY_OUTPUT_DIR
    / "trajectory_representations_200.npy"
)

INDEX_OUTPUT_PATH = (
    SENSITIVITY_OUTPUT_DIR
    / "trajectory_index_200.csv"
)

DISTANCE_OUTPUT_PATH = (
    SENSITIVITY_OUTPUT_DIR
    / "hausdorff_200.npy"
)

METADATA_OUTPUT_PATH = (
    SENSITIVITY_OUTPUT_DIR
    / "hausdorff_200_metadata.json"
)

# Representation construction
def build_all_representations_200(
    trajectory_records,
):
    """
    Build a 200-point arc-length representation for every
    processed trajectory.

    Returns
    -------
    representations
        Shape: (N, 200, 2)
        Coordinate order: [lat, lon]

    metadata
        Per-trajectory metadata aligned with matrix indices.
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
        "Building 200-point arc-length representations..."
    )

    start = perf_counter()

    for index, record in enumerate(
        trajectory_records
    ):

        trajectory = pd.read_csv(
            record[
                "file_path"
            ],
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
            representation[
                "lat"
            ]
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
            representation[
                "lon"
            ]
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
                    record[
                        "month"
                    ]
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

    elapsed = (
        perf_counter()
        - start
    )

    return (
        representations,
        pd.DataFrame(
            metadata
        ),
        elapsed,
    )

# Ordering validation
def validate_against_primary_index(
    sensitivity_index,
):
    """
    Confirm that the 200-point sensitivity matrix uses exactly
    the same trajectory ordering as the primary 100-point matrix.
    """

    if not PRIMARY_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Primary trajectory index missing: "
            f"{PRIMARY_INDEX_PATH}"
        )

    primary = pd.read_csv(
        PRIMARY_INDEX_PATH
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
        sensitivity_index
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        primary
    ) != len(
        sensitivity
    ):
        raise RuntimeError(
            "Primary and sensitivity trajectory counts differ."
        )

    columns = [
        "matrix_index",
        "month",
        "trajectory_id",
        "file_name",
    ]

    exact_match = (
        primary[
            columns
        ]
        .equals(
            sensitivity[
                columns
            ]
        )
    )

    if not exact_match:
        comparison = (
            primary[
                columns
            ]
            != sensitivity[
                columns
            ]
        )

        mismatch_rows = (
            comparison
            .any(
                axis=1
            )
            .sum()
        )

        raise RuntimeError(
            "200-point trajectory ordering does not match "
            f"the primary ordering. "
            f"Mismatched rows: {mismatch_rows}"
        )

    return True

# Distance summary
def summarize_distance_matrix(
    matrix,
):
    """
    Summarize unique off-diagonal trajectory distances.
    """

    upper_triangle = (
        matrix[
            np.triu_indices(
                matrix.shape[
                    0
                ],
                k=1,
            )
        ]
    )

    return {
        "minimum_km": float(
            np.min(
                upper_triangle
            )
        ),

        "median_km": float(
            np.median(
                upper_triangle
            )
        ),

        "p95_km": float(
            np.percentile(
                upper_triangle,
                95,
            )
        ),

        "maximum_km": float(
            np.max(
                upper_triangle
            )
        ),
    }

# Main
def main():

    print(
        "=" * 100
    )

    print(
        "200-POINT REPRESENTATION SENSITIVITY"
    )

    print(
        "=" * 100
    )

    print(
        "\nRepresentation points:",
        REPRESENTATION_POINTS,
    )
    # Discover trajectories using the exact same primary logic
    trajectory_records = (
        discover_trajectories()
    )

    print(
        "Trajectories discovered:",
        len(
            trajectory_records
        ),
    )

    if len(
        trajectory_records
    ) != 1354:
        raise RuntimeError(
            "Expected 1,354 processed trajectories, "
            f"found {len(trajectory_records)}."
        )
    # Build representations
    (
        representations,
        trajectory_index,
        representation_elapsed,
    ) = build_all_representations_200(
        trajectory_records
    )

    print(
        "\nRepresentation array shape:",
        representations.shape,
    )

    print(
        "Representation build time:",
        f"{representation_elapsed:.3f} s",
    )
    # Verify matrix ordering before expensive pairwise computation
    validate_against_primary_index(
        trajectory_index
    )

    print(
        "\nPrimary-index alignment: PASS"
    )
    # Save representation/index first
    SENSITIVITY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        REPRESENTATION_OUTPUT_PATH,
        representations,
    )

    trajectory_index.to_csv(
        INDEX_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved representation:"
    )

    print(
        REPRESENTATION_OUTPUT_PATH
    )
    # Full pairwise Hausdorff matrix
    (
        distance_matrix,
        distance_elapsed,
    ) = build_full_distance_matrix(
        representations
    )
    # Matrix validation
    validation_result = (
        validate_distance_matrix(
            distance_matrix
        )
    )

    print(
        "\nDistance-matrix validation result:"
    )

    print(
        validation_result
    )

    # The existing validator may return None on success depending
    # on its implementation, so independently enforce the key
    # structural conditions as well.

    if (
        distance_matrix.shape
        != (
            1354,
            1354,
        )
    ):
        raise RuntimeError(
            "Unexpected distance-matrix shape."
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

    print(
        "Independent structural validation: PASS"
    )
    # Summary
    distance_summary = (
        summarize_distance_matrix(
            distance_matrix
        )
    )

    print(
        "\n200-point Hausdorff distribution:"
    )

    print(
        f"Minimum: "
        f"{distance_summary['minimum_km']:.3f} km"
    )

    print(
        f"Median:  "
        f"{distance_summary['median_km']:.3f} km"
    )

    print(
        f"P95:     "
        f"{distance_summary['p95_km']:.3f} km"
    )

    print(
        f"Maximum: "
        f"{distance_summary['maximum_km']:.3f} km"
    )

    print(
        "\nFull distance computation time:",
        f"{distance_elapsed:.3f} s",
    )
    # Save
    np.save(
        DISTANCE_OUTPUT_PATH,
        distance_matrix,
    )

    metadata = {
        "purpose": (
            "representation_sensitivity"
        ),

        "representation_points": (
            REPRESENTATION_POINTS
        ),

        "trajectory_count": (
            int(
                distance_matrix.shape[
                    0
                ]
            )
        ),

        "unique_pair_count": (
            int(
                distance_matrix.shape[
                    0
                ]
                * (
                    distance_matrix.shape[
                        0
                    ]
                    - 1
                )
                // 2
            )
        ),

        "trajectory_order_matches_primary": (
            True
        ),

        "representation_build_seconds": (
            float(
                representation_elapsed
            )
        ),

        "distance_matrix_build_seconds": (
            float(
                distance_elapsed
            )
        ),

        "distance_summary_km": (
            distance_summary
        ),
    }

    with open(
        METADATA_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        "\nSaved:"
    )

    print(
        REPRESENTATION_OUTPUT_PATH
    )

    print(
        INDEX_OUTPUT_PATH
    )

    print(
        DISTANCE_OUTPUT_PATH
    )

    print(
        METADATA_OUTPUT_PATH
    )

    print(
        "\n200-POINT DISTANCE MATRIX: PASS"
    )


if __name__ == "__main__":
    main()