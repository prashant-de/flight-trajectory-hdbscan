import numpy as np
import pandas as pd

from src.config import REDUCED_JULY_DIR
from src.preprocessing import (
    SPATIAL_MAX_SPEED_KMH,
    SPATIAL_MIN_JUMP_DISTANCE_KM,
    WORKING_RESAMPLE_INTERVAL_S,
    filter_spatially_inconsistent_updates,
    resample_trajectory,
)
from src.preprocessing import (
    MAX_SPATIAL_REMOVAL_FRACTION,
    SPATIAL_MAX_SPEED_KMH,
    SPATIAL_MIN_JUMP_DISTANCE_KM,
    WORKING_RESAMPLE_INTERVAL_S,
    filter_spatially_inconsistent_updates,
    resample_trajectory,
)


# July-specific conservative screening

# July does not contain callsign or lastposupdate.
#
# Therefore the January three-part multi-segment rule cannot be
# reproduced exactly.
#
# For July we use a conservative observable rule:
#     duration > 8 hours
# AND maximum timestamp gap > 1 hour.
#
# Normal LAX -> JFK trajectories should not satisfy both conditions.
JULY_MULTISEGMENT_MIN_DURATION_H = 8.0
JULY_MULTISEGMENT_MAX_GAP_S = 3600.0


def parse_timestamp(series):
    """
    Parse either Unix-second timestamps or datetime strings.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().mean() > 0.95:
        return pd.to_datetime(
            numeric,
            unit="s",
            utc=True,
            errors="coerce",
        )

    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    )


def datetime_to_unix_seconds(parsed_time):
    """
    Convert UTC pandas timestamps to Unix seconds.
    """

    epoch = pd.Timestamp(
        "1970-01-01",
        tz="UTC",
    )

    return (
        parsed_time - epoch
    ).dt.total_seconds()


def classify_july_file(
    flight,
    parsed_time,
    file_name,
):
    """
    Identify obvious severe multi-segment July files.

    Because July lacks callsign, classification is based only
    on observable duration and timestamp-gap evidence.
    """

    valid_times = parsed_time.dropna()

    if len(valid_times) >= 2:
        total_duration_h = (
            (
                valid_times.iloc[-1]
                - valid_times.iloc[0]
            ).total_seconds()
            / 3600
        )
    else:
        total_duration_h = np.nan

    gap_s = (
        parsed_time
        .diff()
        .dt.total_seconds()
    )

    max_gap_s = gap_s.max()

    severe_multisegment = (
        pd.notna(total_duration_h)
        and pd.notna(max_gap_s)
        and total_duration_h
        > JULY_MULTISEGMENT_MIN_DURATION_H
        and max_gap_s
        > JULY_MULTISEGMENT_MAX_GAP_S
    )

    return {
        "file_name": file_name,

        "total_duration_h": (
            total_duration_h
        ),

        "max_row_gap_s": (
            max_gap_s
        ),

        "exclude_severe_multisegment": (
            severe_multisegment
        ),
    }


def build_july_position_updates(
    flight,
    parsed_time,
):
    """
    Build usable July position updates.

    July has no lastposupdate field. Therefore consecutive identical
    coordinates are treated as carried-forward/stale observations.

    The FIRST observation of a repeated coordinate run is retained.
    """

    data = flight.copy()

    data.insert(
        0,
        "source_row_index",
        data.index,
    )

    data["parsed_time"] = (
        parsed_time
    )

    data["lat"] = pd.to_numeric(
        data["lat"],
        errors="coerce",
    )

    data["lon"] = pd.to_numeric(
        data["lon"],
        errors="coerce",
    )

    original_rows = len(data)

    valid = (
        data["parsed_time"].notna()
        & data["lat"].notna()
        & data["lon"].notna()
        & data["lat"].between(
            -90,
            90,
        )
        & data["lon"].between(
            -180,
            180,
        )
    )

    usable = (
        data.loc[
            valid
        ]
        .copy()
    )

    usable_rows = len(
        usable
    )

    # Consecutive identical coordinates are interpreted as
    # carried-forward position observations.
    coordinate_changed = (
        usable["lat"].ne(
            usable["lat"].shift(1)
        )
        | usable["lon"].ne(
            usable["lon"].shift(1)
        )
    )

    position_updates = (
        usable.loc[
            coordinate_changed
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Our shared spatial-cleaning/resampling functions use
    # lastposupdate as their position-time coordinate.
    #
    # July does not contain that field, so we use the actual
    # reduced observation timestamp as the best available proxy.
    position_updates[
        "lastposupdate"
    ] = datetime_to_unix_seconds(
        position_updates[
            "parsed_time"
        ]
    )

    report = {
        "original_rows": (
            original_rows
        ),

        "usable_geometry_rows": (
            usable_rows
        ),

        "removed_unusable_rows": (
            original_rows
            - usable_rows
        ),

        "position_update_rows": (
            len(position_updates)
        ),

        "removed_repeated_coordinate_rows": (
            usable_rows
            - len(position_updates)
        ),
    }

    return (
        position_updates,
        report,
    )


def preprocess_july_flight(
    file_path,
):
    """
    Harmonize one friend-supplied July trajectory.
    """

    flight = pd.read_csv(
        file_path
    )

    parsed_time = parse_timestamp(
        flight["time"]
    )

    classification = (
        classify_july_file(
            flight,
            parsed_time,
            file_path.name,
        )
    )

    if classification[
        "exclude_severe_multisegment"
    ]:
        return None, {
            **classification,

            "status": (
                "excluded_multisegment"
            ),

            "original_rows": len(
                flight
            ),

            "usable_geometry_rows": None,
            "removed_unusable_rows": None,
            "position_update_rows": None,
            "removed_repeated_coordinate_rows": None,
            "removed_spatial_rows": None,
            "cleaned_rows": None,
            "resampled_rows": None,
        }

    position_updates, row_report = (
        build_july_position_updates(
            flight,
            parsed_time,
        )
    )

    cleaned, spatial_report = (
        filter_spatially_inconsistent_updates(
            position_updates,
            max_speed_kmh=(
                SPATIAL_MAX_SPEED_KMH
            ),
            min_jump_distance_km=(
                SPATIAL_MIN_JUMP_DISTANCE_KM
            ),
        )
    )

    position_update_count = len(
        position_updates
    )

    if position_update_count > 0:
        spatial_removal_fraction = (
            spatial_report[
                "removed_spatial_rows"
            ]
            / position_update_count
        )
    else:
        spatial_removal_fraction = np.nan

    # If too much of the trajectory must be rejected,
    # the complete flight is considered unreliable for MVP clustering.
    if (
        pd.notna(
            spatial_removal_fraction
        )
        and spatial_removal_fraction
        > MAX_SPATIAL_REMOVAL_FRACTION
    ):
        return None, {
            **classification,
            **row_report,
            **spatial_report,

            "status": (
                "excluded_spatial_corruption"
            ),

            "spatial_removal_fraction": (
                spatial_removal_fraction
            ),

            "cleaned_rows": len(
                cleaned
            ),

            "resampled_rows": None,
        }

    resampled = (
        resample_trajectory(
            cleaned,
            interval_s=(
                WORKING_RESAMPLE_INTERVAL_S
            ),
        )
    )

    report = {
        **classification,
        **row_report,
        **spatial_report,

        "status": "kept",

        "cleaned_rows": len(
            cleaned
        ),

        "resampled_rows": len(
            resampled
        ),
        "spatial_removal_fraction": (
           spatial_removal_fraction
        ),
    }

    return (
        resampled,
        report,
    )


def main():
    files = sorted(
        REDUCED_JULY_DIR.glob(
            "*.csv"
        )
    )

    total_files = len(
        files
    )

    print(
        f"July source files: {total_files}"
    )

    print(
        "Starting final July harmonization check..."
    )

    reports = []

    for number, file_path in enumerate(
        files,
        start=1,
    ):
        _, report = (
            preprocess_july_flight(
                file_path
            )
        )

        reports.append(
            report
        )

        if (
            number % 50 == 0
            or number == total_files
        ):
            print(
                f"Processed {number} / {total_files}",
                flush=True,
            )

    table = pd.DataFrame(
        reports
    )

    kept = table[
        table["status"]
        == "kept"
    ].copy()

    multisegment = table[
        table["status"]
        == "excluded_multisegment"
    ].copy()

    spatial_corruption = table[
        table["status"]
        == "excluded_spatial_corruption"
    ].copy()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL JULY HARMONIZATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nSource files:"
    )
    print(
        len(table)
    )

    print(
        "\nFlights kept:"
    )
    print(
        len(kept)
    )

    print(
        "\nExcluded severe multi-segment flights:"
    )
    print(
        len(multisegment)
    )

    print(
        "\nExcluded severe spatial-corruption flights:"
    )
    print(
        len(spatial_corruption)
    )

    print(
        "\nFinal exact 10-second rows:"
    )
    print(
        int(
            kept[
                "resampled_rows"
            ].sum()
        )
    )

    print(
        "\n10-second rows per retained trajectory:"
    )

    print(
        "minimum:",
        int(
            kept[
                "resampled_rows"
            ].min()
        ),
    )

    print(
        "median:",
        f"{kept['resampled_rows'].median():.1f}",
    )

    print(
        "mean:",
        f"{kept['resampled_rows'].mean():.1f}",
    )

    print(
        "maximum:",
        int(
            kept[
                "resampled_rows"
            ].max()
        ),
    )

    print(
        "\nFlights with some spatial cleaning but still retained:"
    )

    print(
        int(
            (
                kept[
                    "removed_spatial_rows"
                ] > 0
            ).sum()
        )
    )

    print(
        "\nLargest retained spatial-removal percentage:"
    )

    print(
        f"{kept['spatial_removal_fraction'].max() * 100:.3f}%"
    )

    print(
        "\nExcluded severe spatial-corruption flights:"
    )

    if len(
        spatial_corruption
    ) == 0:
        print(
            "None"
        )
    else:
        columns = [
            "file_name",
            "original_rows",
            "position_update_rows",
            "removed_spatial_rows",
            "spatial_removal_fraction",
            "cleaned_rows",
        ]

        display = (
            spatial_corruption[
                columns
            ]
            .copy()
        )

        display[
            "spatial_removal_percent"
        ] = (
            display[
                "spatial_removal_fraction"
            ]
            * 100
        )

        print(
            display
            .sort_values(
                "spatial_removal_fraction",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )

    print(
        "\nExcluded multi-segment flights:"
    )

    if len(
        multisegment
    ) == 0:
        print(
            "None"
        )
    else:
        print(
            multisegment[
                [
                    "file_name",
                    "total_duration_h",
                    "max_row_gap_s",
                ]
            ]
            .sort_values(
                "max_row_gap_s",
                ascending=False,
            )
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()