"""
Build the full discrete Frechet trajectory-distance matrix.

Sensitivity configuration
-------------------------
Trajectory representation:
    100 arc-length points

Primary metric:
    symmetric Hausdorff distance

Sensitivity metric:
    discrete Frechet distance

The same 1,354 trajectories and the same 100-point representations
are used. Only the trajectory-distance definition changes.

This runner is checkpointed and resumable because the full
915,981-pair Frechet calculation is substantially more expensive
than Hausdorff distance.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import (
    DISTANCE_OUTPUT_DIR,
)

from src.distances import (
    discrete_frechet_distance_km,
)

# Paths
REPRESENTATION_PATH = (
    DISTANCE_OUTPUT_DIR
    / "trajectory_representations_100.npy"
)

PRIMARY_INDEX_PATH = (
    DISTANCE_OUTPUT_DIR
    / "trajectory_index.csv"
)

SENSITIVITY_OUTPUT_DIR = (
    DISTANCE_OUTPUT_DIR
    / "sensitivity"
)

FRECHET_MATRIX_PATH = (
    SENSITIVITY_OUTPUT_DIR
    / "frechet_100.npy"
)

STATE_PATH = (
    SENSITIVITY_OUTPUT_DIR
    / "frechet_100_state.json"
)

METADATA_PATH = (
    SENSITIVITY_OUTPUT_DIR
    / "frechet_100_metadata.json"
)


EXPECTED_TRAJECTORIES = 1354
EXPECTED_POINTS = 100

REPORT_EVERY_PAIRS = 10000
CHECKPOINT_EVERY_ROWS = 10

# Input validation
def load_inputs():
    """
    Load the frozen primary 100-point representations and
    trajectory index.
    """

    if not REPRESENTATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing representation file: {REPRESENTATION_PATH}"
        )

    if not PRIMARY_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing trajectory index: {PRIMARY_INDEX_PATH}"
        )

    representations = np.load(
        REPRESENTATION_PATH,
        mmap_mode="r",
    )

    trajectory_index = pd.read_csv(
        PRIMARY_INDEX_PATH
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

    expected_shape = (
        EXPECTED_TRAJECTORIES,
        EXPECTED_POINTS,
        2,
    )

    if representations.shape != expected_shape:
        raise RuntimeError(
            "Unexpected representation shape. "
            f"Expected {expected_shape}, "
            f"found {representations.shape}."
        )

    if len(
        trajectory_index
    ) != EXPECTED_TRAJECTORIES:
        raise RuntimeError(
            "Unexpected trajectory-index size."
        )

    expected_indices = np.arange(
        EXPECTED_TRAJECTORIES
    )

    actual_indices = (
        trajectory_index[
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
            "Trajectory index is not sequential."
        )

    return (
        representations,
        trajectory_index,
    )

# Checkpoint handling
def save_state(
    next_row,
    completed_pairs,
    cumulative_seconds,
):
    """
    Save resumable calculation state.
    """

    state = {
        "next_row": int(
            next_row
        ),

        "completed_pairs": int(
            completed_pairs
        ),

        "cumulative_seconds": float(
            cumulative_seconds
        ),
    }

    temporary_path = Path(
        str(
            STATE_PATH
        )
        + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=4,
        )

    temporary_path.replace(
        STATE_PATH
    )


def load_state():
    """
    Load previous progress if a checkpoint exists.
    """

    if not STATE_PATH.exists():
        return {
            "next_row": 0,
            "completed_pairs": 0,
            "cumulative_seconds": 0.0,
        }

    with open(
        STATE_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )

# Matrix creation / resume
def open_matrix(
    n,
    resume,
):
    """
    Create or reopen the Frechet matrix as a NumPy memory map.
    """

    if resume:

        if not FRECHET_MATRIX_PATH.exists():
            raise FileNotFoundError(
                "Checkpoint exists but Frechet matrix is missing."
            )

        matrix = np.load(
            FRECHET_MATRIX_PATH,
            mmap_mode="r+",
        )

        if matrix.shape != (
            n,
            n,
        ):
            raise RuntimeError(
                "Saved Frechet matrix has unexpected shape."
            )

        return matrix

    matrix = np.lib.format.open_memmap(
        FRECHET_MATRIX_PATH,
        mode="w+",
        dtype="float64",
        shape=(
            n,
            n,
        ),
    )

    matrix[:] = 0.0

    matrix.flush()

    return matrix

# Frechet computation
def build_frechet_matrix(
    representations,
):
    """
    Compute or resume the complete symmetric Frechet matrix.
    """

    n = representations.shape[
        0
    ]

    total_pairs = (
        n
        * (
            n - 1
        )
        // 2
    )

    state = load_state()

    next_row = int(
        state[
            "next_row"
        ]
    )

    completed_pairs = int(
        state[
            "completed_pairs"
        ]
    )

    previous_seconds = float(
        state[
            "cumulative_seconds"
        ]
    )

    resume = (
        next_row > 0
    )

    matrix = open_matrix(
        n,
        resume,
    )

        # -------------------------------------------------------------
    # The existing discrete_frechet_distance_km() implementation
    # expects DataFrames with named "lat" and "lon" columns.
    #
    # The frozen representation file is a NumPy array with shape
    # (N, 100, 2), so convert each trajectory once here rather
    # than rebuilding a DataFrame for every trajectory pair.
    # -------------------------------------------------------------

    representation_tables = [
        pd.DataFrame(
            np.asarray(
                representations[index]
            ),
            columns=[
                "lat",
                "lon",
            ],
        )
        for index in range(
            n
        )
    ]

    print(
        "Prepared DataFrame trajectory cache:",
        len(
            representation_tables
        ),
    )

    if resume:

        print(
            "\nResuming previous Frechet calculation."
        )

        print(
            "Starting row:",
            next_row,
        )

        print(
            "Previously completed pairs:",
            f"{completed_pairs:,}",
        )

    else:

        print(
            "\nStarting new Frechet calculation."
        )

    print(
        "\nTrajectories:",
        n,
    )

    print(
        "Unique pairs:",
        f"{total_pairs:,}",
    )

    print(
        "Representation points:",
        representations.shape[
            1
        ],
    )

    session_start = (
        perf_counter()
    )

    next_report = (
        (
            completed_pairs
            // REPORT_EVERY_PAIRS
        )
        + 1
    ) * REPORT_EVERY_PAIRS

    for i in range(
        next_row,
        n,
    ):

        trajectory_a = (
            representation_tables[
                i
            ]
        )

        for j in range(
            i + 1,
            n,
        ):

            distance = (
                discrete_frechet_distance_km(
                    trajectory_a,
                    representation_tables[
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
                    * 100.0
                )

                elapsed_session = (
                    perf_counter()
                    - session_start
                )

                elapsed_total = (
                    previous_seconds
                    + elapsed_session
                )

                if completed_pairs > 0:

                    seconds_per_pair = (
                        elapsed_total
                        / completed_pairs
                    )

                    remaining_pairs = (
                        total_pairs
                        - completed_pairs
                    )

                    estimated_remaining = (
                        remaining_pairs
                        * seconds_per_pair
                    )

                else:

                    estimated_remaining = (
                        float(
                            "nan"
                        )
                    )

                print(
                    f"Computed "
                    f"{completed_pairs:,} / "
                    f"{total_pairs:,} pairs "
                    f"({percent:.1f}%) | "
                    f"estimated remaining "
                    f"{estimated_remaining / 60.0:.1f} min",
                    flush=True,
                )

                next_report += (
                    REPORT_EVERY_PAIRS
                )

        # ---------------------------------------------------------
        # A checkpoint is written only after a complete matrix row.
        # If interruption occurs mid-row, that row is safely
        # recomputed on the next run.
        # ---------------------------------------------------------

        if (
            (i + 1) % CHECKPOINT_EVERY_ROWS == 0
            or i + 1 == n
        ):

            matrix.flush()

            elapsed_session = (
                perf_counter()
                - session_start
            )

            cumulative_seconds = (
                previous_seconds
                + elapsed_session
            )

            save_state(
                next_row=(
                    i + 1
                ),

                completed_pairs=(
                    completed_pairs
                ),

                cumulative_seconds=(
                    cumulative_seconds
                ),
            )

    matrix.flush()

    session_elapsed = (
        perf_counter()
        - session_start
    )

    total_elapsed = (
        previous_seconds
        + session_elapsed
    )

    return (
        matrix,
        total_elapsed,
        completed_pairs,
        total_pairs,
    )

# Validation
def validate_matrix(
    matrix,
):
    """
    Validate completed Frechet matrix.
    """

    problems = []

    if matrix.shape != (
        EXPECTED_TRAJECTORIES,
        EXPECTED_TRAJECTORIES,
    ):

        problems.append(
            "incorrect_shape"
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):

        problems.append(
            "nonfinite_values"
        )

    if np.any(
        matrix < 0
    ):

        problems.append(
            "negative_values"
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
            "not_symmetric"
        )

    return problems

# Summary
def summarize_matrix(
    matrix,
):
    """
    Summarize all unique off-diagonal distances.
    """

    upper = (
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

# Main
def main():

    print(
        "=" * 100
    )

    print(
        "FULL DISCRETE FRECHET DISTANCE MATRIX"
    )

    print(
        "=" * 100
    )

    SENSITIVITY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        representations,
        trajectory_index,
    ) = load_inputs()

    print(
        "\nRepresentation array:",
        representations.shape,
    )

    print(
        "Trajectory index:",
        len(
            trajectory_index
        ),
    )

    (
        matrix,
        total_elapsed,
        completed_pairs,
        total_pairs,
    ) = build_frechet_matrix(
        representations
    )

    if (
        completed_pairs
        != total_pairs
    ):

        raise RuntimeError(
            "Frechet calculation ended before all pairs "
            "were completed."
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "VALIDATING FRECHET MATRIX"
    )

    print(
        "=" * 100
    )

    problems = validate_matrix(
        matrix
    )

    if problems:

        raise RuntimeError(
            "Frechet matrix validation failed: "
            + ", ".join(
                problems
            )
        )

    print(
        "\nStructural validation: PASS"
    )

    summary = summarize_matrix(
        matrix
    )

    print(
        "\nDiscrete Frechet distribution:"
    )

    print(
        f"Minimum: "
        f"{summary['minimum_km']:.3f} km"
    )

    print(
        f"Median:  "
        f"{summary['median_km']:.3f} km"
    )

    print(
        f"P95:     "
        f"{summary['p95_km']:.3f} km"
    )

    print(
        f"Maximum: "
        f"{summary['maximum_km']:.3f} km"
    )

    print(
        "\nTotal computation time:",
        f"{total_elapsed:.3f} s",
    )

    print(
        "Total computation time:",
        f"{total_elapsed / 60.0:.2f} min",
    )

    metadata = {
        "purpose": (
            "trajectory_metric_sensitivity"
        ),

        "metric": (
            "discrete_frechet_haversine"
        ),

        "representation_points": (
            EXPECTED_POINTS
        ),

        "trajectory_count": (
            EXPECTED_TRAJECTORIES
        ),

        "unique_pair_count": (
            total_pairs
        ),

        "total_computation_seconds": (
            float(
                total_elapsed
            )
        ),

        "distance_summary_km": (
            summary
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # The calculation is complete, so the temporary resume
    # checkpoint is no longer required.

    if STATE_PATH.exists():

        STATE_PATH.unlink()

    print(
        "\nSaved:"
    )

    print(
        FRECHET_MATRIX_PATH
    )

    print(
        METADATA_PATH
    )

    print(
        "\nFULL FRECHET MATRIX: PASS"
    )


if __name__ == "__main__":
    main()