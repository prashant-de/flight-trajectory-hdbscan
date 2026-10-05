import json

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_JANUARY_DIR,
    PROCESSED_JULY_DIR,
    QUALITY_OUTPUT_DIR,
    RAW_JANUARY_DIR,
    REDUCED_JULY_DIR,
)

from src.data_io import (
    find_flight_csvs,
)

from src.preprocessing import (
    WORKING_RESAMPLE_INTERVAL_S,
    preprocess_single_flight,
    resample_trajectory,
)

from src.july_preprocessing import (
    preprocess_july_flight,
)


# Output preparation

def prepare_output_directories():
    """
    Create generated-data and reporting directories.

    Existing generated CSV trajectory files are removed so that
    rerunning this script cannot leave stale trajectories behind.

    Raw and reduced source data are never modified.
    """

    directories = [
        PROCESSED_JANUARY_DIR,
        PROCESSED_JULY_DIR,
        QUALITY_OUTPUT_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Only clear generated trajectory CSVs.
    for directory in [
        PROCESSED_JANUARY_DIR,
        PROCESSED_JULY_DIR,
    ]:
        for file_path in directory.glob(
            "*.csv"
        ):
            file_path.unlink()


# Final trajectory format

def standardize_output_trajectory(
    trajectory,
):
    """
    Convert one resampled trajectory into the final common
    January/July processed-data schema.
    """

    output = trajectory[
        [
            "sample_time",
            "lat",
            "lon",
            "interpolated",
        ]
    ].copy()

    output.insert(
        0,
        "sample_index",
        np.arange(
            len(output),
            dtype="int64",
        ),
    )

    output.insert(
        2,
        "time_utc",
        pd.to_datetime(
            output["sample_time"],
            unit="s",
            utc=True,
        ),
    )

    return output


# Processed trajectory validation

def validate_processed_trajectory(
    trajectory,
):
    """
    Validate one final standardized trajectory.

    Returns a list of detected problems.
    An empty list means the trajectory passed validation.
    """

    problems = []

    required_columns = {
        "sample_index",
        "sample_time",
        "time_utc",
        "lat",
        "lon",
        "interpolated",
    }

    missing_columns = (
        required_columns
        - set(
            trajectory.columns
        )
    )

    if missing_columns:
        problems.append(
            "missing_columns="
            + ",".join(
                sorted(
                    missing_columns
                )
            )
        )

        return problems

    if len(trajectory) == 0:
        problems.append(
            "empty_trajectory"
        )

        return problems

    sample_time = (
        trajectory[
            "sample_time"
        ]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    latitude = (
        trajectory["lat"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    longitude = (
        trajectory["lon"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    if not np.all(
        np.isfinite(
            sample_time
        )
    ):
        problems.append(
            "nonfinite_sample_time"
        )

    if not np.all(
        np.isfinite(
            latitude
        )
    ):
        problems.append(
            "nonfinite_latitude"
        )

    if not np.all(
        np.isfinite(
            longitude
        )
    ):
        problems.append(
            "nonfinite_longitude"
        )

    if not np.all(
        (
            latitude >= -90
        )
        & (
            latitude <= 90
        )
    ):
        problems.append(
            "invalid_latitude"
        )

    if not np.all(
        (
            longitude >= -180
        )
        & (
            longitude <= 180
        )
    ):
        problems.append(
            "invalid_longitude"
        )

    if len(sample_time) > 1:
        gaps = np.diff(
            sample_time
        )

        if np.any(
            gaps <= 0
        ):
            problems.append(
                "nonincreasing_time"
            )

        # Regular trajectories use the exact 10-second grid.
        # Preserved missing-data regions may therefore have
        # gaps of 20, 30, 40 ... seconds.
        grid_ratio = (
            gaps
            / WORKING_RESAMPLE_INTERVAL_S
        )

        if not np.allclose(
            grid_ratio,
            np.round(
                grid_ratio
            ),
            atol=1e-8,
        ):
            problems.append(
                "off_10_second_grid"
            )

    expected_index = np.arange(
        len(trajectory),
        dtype="int64",
    )

    if not np.array_equal(
        trajectory[
            "sample_index"
        ].to_numpy(),
        expected_index,
    ):
        problems.append(
            "invalid_sample_index"
        )

    return problems


# Save one trajectory

def save_processed_trajectory(
    trajectory,
    output_path,
):
    """
    Standardize, validate and save one processed trajectory.
    """

    standardized = (
        standardize_output_trajectory(
            trajectory
        )
    )

    problems = (
        validate_processed_trajectory(
            standardized
        )
    )

    if problems:
        raise ValueError(
            f"{output_path.name}: "
            + "; ".join(
                problems
            )
        )

    standardized.to_csv(
        output_path,
        index=False,
    )

    return standardized


# January

def build_january_dataset():
    """
    Generate final January 10-second trajectories.
    """

    files = find_flight_csvs(
        RAW_JANUARY_DIR
    )

    records = []

    print(
        "\nBuilding January processed dataset..."
    )

    for number, file_path in enumerate(
        files,
        start=1,
    ):
        cleaned, report = (
            preprocess_single_flight(
                file_path
            )
        )

        trajectory_id = (
            file_path.stem
        )

        if cleaned is None:
            records.append(
                {
                    "month": "2026-01",
                    "trajectory_id": (
                        trajectory_id
                    ),
                    "source_file": (
                        file_path.name
                    ),
                    "processed_file": "",
                    "output_rows": 0,
                    **report,
                }
            )

        else:
            resampled = (
                resample_trajectory(
                    cleaned,
                    interval_s=(
                        WORKING_RESAMPLE_INTERVAL_S
                    ),
                )
            )

            output_path = (
                PROCESSED_JANUARY_DIR
                / file_path.name
            )

            saved = (
                save_processed_trajectory(
                    resampled,
                    output_path,
                )
            )

            records.append(
                {
                    "month": "2026-01",
                    "trajectory_id": (
                        trajectory_id
                    ),
                    "source_file": (
                        file_path.name
                    ),
                    "processed_file": (
                        output_path.name
                    ),
                    "output_rows": (
                        len(saved)
                    ),
                    **report,
                }
            )

        if (
            number % 50 == 0
            or number == len(files)
        ):
            print(
                f"January: processed "
                f"{number} / {len(files)}",
                flush=True,
            )

    return pd.DataFrame(
        records
    )


# July

def build_july_dataset():
    """
    Generate final July harmonized 10-second trajectories.
    """

    files = sorted(
        REDUCED_JULY_DIR.glob(
            "*.csv"
        )
    )

    records = []

    print(
        "\nBuilding July processed dataset..."
    )

    for number, file_path in enumerate(
        files,
        start=1,
    ):
        resampled, report = (
            preprocess_july_flight(
                file_path
            )
        )

        trajectory_id = (
            file_path.stem
        )

        if resampled is None:
            records.append(
                {
                    "month": "2026-07",
                    "trajectory_id": (
                        trajectory_id
                    ),
                    "source_file": (
                        file_path.name
                    ),
                    "processed_file": "",
                    "output_rows": 0,
                    **report,
                }
            )

        else:
            output_path = (
                PROCESSED_JULY_DIR
                / file_path.name
            )

            saved = (
                save_processed_trajectory(
                    resampled,
                    output_path,
                )
            )

            records.append(
                {
                    "month": "2026-07",
                    "trajectory_id": (
                        trajectory_id
                    ),
                    "source_file": (
                        file_path.name
                    ),
                    "processed_file": (
                        output_path.name
                    ),
                    "output_rows": (
                        len(saved)
                    ),
                    **report,
                }
            )

        if (
            number % 50 == 0
            or number == len(files)
        ):
            print(
                f"July: processed "
                f"{number} / {len(files)}",
                flush=True,
            )

    return pd.DataFrame(
        records
    )


# Reports

def save_manifests(
    january,
    july,
):
    """
    Save January, July and combined preprocessing manifests.
    """

    january_path = (
        QUALITY_OUTPUT_DIR
        / "january_preprocessing_manifest.csv"
    )

    july_path = (
        QUALITY_OUTPUT_DIR
        / "july_preprocessing_manifest.csv"
    )

    combined_path = (
        QUALITY_OUTPUT_DIR
        / "processed_dataset_manifest.csv"
    )

    january.to_csv(
        january_path,
        index=False,
    )

    july.to_csv(
        july_path,
        index=False,
    )

    combined = pd.concat(
        [
            january,
            july,
        ],
        ignore_index=True,
        sort=False,
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    return combined


def save_summary(
    january,
    july,
    combined,
):
    """
    Save a small JSON summary describing the final dataset.
    """

    january_kept = january[
        january["status"]
        == "kept"
    ]

    july_kept = july[
        july["status"]
        == "kept"
    ]

    summary = {
        "working_resample_interval_seconds": (
            WORKING_RESAMPLE_INTERVAL_S
        ),

        "january": {
            "source_trajectories": (
                int(
                    len(january)
                )
            ),

            "retained_trajectories": (
                int(
                    len(
                        january_kept
                    )
                )
            ),

            "processed_rows": (
                int(
                    january_kept[
                        "output_rows"
                    ].sum()
                )
            ),
        },

        "july": {
            "source_trajectories": (
                int(
                    len(july)
                )
            ),

            "retained_trajectories": (
                int(
                    len(
                        july_kept
                    )
                )
            ),

            "processed_rows": (
                int(
                    july_kept[
                        "output_rows"
                    ].sum()
                )
            ),
        },

        "combined": {
            "source_trajectories": (
                int(
                    len(combined)
                )
            ),

            "retained_trajectories": (
                int(
                    len(
                        combined[
                            combined[
                                "status"
                            ]
                            == "kept"
                        ]
                    )
                )
            ),

            "processed_rows": (
                int(
                    combined.loc[
                        combined[
                            "status"
                        ]
                        == "kept",
                        "output_rows",
                    ].sum()
                )
            ),
        },
    }

    summary_path = (
        QUALITY_OUTPUT_DIR
        / "processed_dataset_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    return summary


# Final verification

def verify_generated_files(
    january,
    july,
):
    """
    Verify that manifest counts match actual generated CSV counts.
    """

    january_files = list(
        PROCESSED_JANUARY_DIR.glob(
            "*.csv"
        )
    )

    july_files = list(
        PROCESSED_JULY_DIR.glob(
            "*.csv"
        )
    )

    expected_january = int(
        (
            january[
                "status"
            ] == "kept"
        ).sum()
    )

    expected_july = int(
        (
            july[
                "status"
            ] == "kept"
        ).sum()
    )

    if (
        len(january_files)
        != expected_january
    ):
        raise RuntimeError(
            "January generated-file count "
            "does not match manifest."
        )

    if (
        len(july_files)
        != expected_july
    ):
        raise RuntimeError(
            "July generated-file count "
            "does not match manifest."
        )

    return (
        len(january_files),
        len(july_files),
    )


# Main

def main():
    prepare_output_directories()

    january = (
        build_january_dataset()
    )

    july = (
        build_july_dataset()
    )

    combined = save_manifests(
        january,
        july,
    )

    summary = save_summary(
        january,
        july,
        combined,
    )

    (
        january_file_count,
        july_file_count,
    ) = verify_generated_files(
        january,
        july,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL PROCESSED DATASET"
    )

    print(
        "=" * 70
    )

    print(
        "\nJanuary processed files:"
    )
    print(
        january_file_count
    )

    print(
        "\nJuly processed files:"
    )
    print(
        july_file_count
    )

    print(
        "\nCombined processed trajectories:"
    )
    print(
        summary[
            "combined"
        ][
            "retained_trajectories"
        ]
    )

    print(
        "\nJanuary processed rows:"
    )
    print(
        summary[
            "january"
        ][
            "processed_rows"
        ]
    )

    print(
        "\nJuly processed rows:"
    )
    print(
        summary[
            "july"
        ][
            "processed_rows"
        ]
    )

    print(
        "\nCombined processed rows:"
    )
    print(
        summary[
            "combined"
        ][
            "processed_rows"
        ]
    )

    print(
        "\nAll generated trajectory files "
        "passed structural validation."
    )


if __name__ == "__main__":
    main()