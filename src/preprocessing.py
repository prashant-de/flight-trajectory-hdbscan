import math

import numpy as np
import pandas as pd

from src.config import RAW_JANUARY_DIR
from src.data_io import find_flight_csvs


# ---------------------------------------------------------------------
# MVP preprocessing thresholds
# ---------------------------------------------------------------------

# Severe multi-segment trajectory classification.
MULTISEGMENT_MAX_GAP_S = 3600.0
MULTISEGMENT_MIN_DURATION_H = 8.0

# Spatial discontinuity cleaning.
#
# A point is rejected only when BOTH conditions are true:
#   distance > 10 km
#   implied speed > 3000 km/h
SPATIAL_MIN_JUMP_DISTANCE_KM = 10.0
SPATIAL_MAX_SPEED_KMH = 3000.0

EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------------
# File-level quality classification
# ---------------------------------------------------------------------

def classify_loaded_trajectory(
    flight,
    parsed_time,
    file_name,
):
    """
    Classify one already-loaded source trajectory.

    A severe multi-segment candidate must satisfy all three:
    - maximum internal row gap > 1 hour,
    - total duration > 8 hours,
    - more than one observed callsign.

    The raw file is never modified.
    """

    row_gap_s = (
        parsed_time
        .diff()
        .dt.total_seconds()
    )

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

    max_row_gap_s = row_gap_s.max()

    # Keep callsign as text and remove empty values.
    callsign = (
        flight["callsign"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    unique_callsigns = (
        callsign
        .dropna()
        .unique()
        .tolist()
    )

    callsign_count = len(
        unique_callsigns
    )

    severe_multisegment = (
        pd.notna(max_row_gap_s)
        and pd.notna(total_duration_h)
        and max_row_gap_s
        > MULTISEGMENT_MAX_GAP_S
        and total_duration_h
        > MULTISEGMENT_MIN_DURATION_H
        and callsign_count > 1
    )

    return {
        "file_name": file_name,

        "exclude_severe_multisegment": (
            severe_multisegment
        ),

        "total_duration_h": (
            total_duration_h
        ),

        "max_row_gap_s": (
            max_row_gap_s
        ),

        "callsign_count": (
            callsign_count
        ),

        "callsigns": ", ".join(
            unique_callsigns
        ),
    }


# ---------------------------------------------------------------------
# Base position-update preparation
# ---------------------------------------------------------------------

def build_clean_position_updates_from_frame(
    flight,
    parsed_time,
):
    """
    Convert one loaded source file into usable position updates.

    This stage:
    - preserves the source row index,
    - removes unusable timestamps,
    - removes missing coordinates,
    - removes invalid coordinate ranges,
    - removes rows without lastposupdate,
    - removes carried-forward copies of the same position update.

    Spatial discontinuities are handled separately.
    """

    flight = flight.copy()

    # Preserve raw CSV row number for traceability.
    flight.insert(
        0,
        "source_row_index",
        flight.index,
    )

    flight["parsed_time"] = (
        parsed_time
    )

    original_rows = len(flight)

    valid_geometry = (
        parsed_time.notna()
        & flight["lat"].notna()
        & flight["lon"].notna()
        & flight["lat"].between(
            -90,
            90,
        )
        & flight["lon"].between(
            -180,
            180,
        )
        & flight["lastposupdate"].notna()
    )

    usable = (
        flight.loc[
            valid_geometry
        ]
        .copy()
    )

    usable_rows = len(usable)

    # Keep only genuine new position updates.
    new_position_update = (
        usable["lastposupdate"]
        .ne(
            usable[
                "lastposupdate"
            ].shift(1)
        )
    )

    position_updates = (
        usable.loc[
            new_position_update
        ]
        .copy()
        .reset_index(drop=True)
    )

    position_update_rows = len(
        position_updates
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
            position_update_rows
        ),

        "removed_carried_forward_rows": (
            usable_rows
            - position_update_rows
        ),
    }

    return position_updates, report


# ---------------------------------------------------------------------
# Vectorized consecutive-distance helper
# ---------------------------------------------------------------------

def consecutive_haversine_km(
    latitude,
    longitude,
):
    """
    Vectorized great-circle distances between consecutive points.

    The first distance is NaN because there is no previous point.
    """

    n = len(latitude)

    result = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    if n <= 1:
        return result

    lat_rad = np.radians(
        latitude
    )

    lon_rad = np.radians(
        longitude
    )

    lat1 = lat_rad[:-1]
    lat2 = lat_rad[1:]

    lon1 = lon_rad[:-1]
    lon2 = lon_rad[1:]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(
        a,
        0,
        1,
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    result[1:] = (
        EARTH_RADIUS_KM * c
    )

    return result


# ---------------------------------------------------------------------
# Optimized spatial discontinuity filter
# ---------------------------------------------------------------------

def filter_spatially_inconsistent_updates(
    position_updates,
    max_speed_kmh=SPATIAL_MAX_SPEED_KMH,
    min_jump_distance_km=SPATIAL_MIN_JUMP_DISTANCE_KM,
):
    """
    Remove clearly impossible spatial position updates.

    A candidate is rejected only when BOTH:
    - distance from the last accepted point > 10 km,
    - implied speed > 3000 km/h.

    The function first performs a fast vectorized test.

    If no suspicious consecutive jumps exist, the trajectory is
    immediately returned without running the slower sequential filter.
    """

    n = len(position_updates)

    empty_report = {
        "spatial_input_rows": n,
        "spatial_output_rows": n,
        "removed_spatial_rows": 0,
        "max_rejected_distance_km": np.nan,
        "max_rejected_speed_kmh": np.nan,
        "nonpositive_update_time_rows": 0,
    }

    if n <= 1:
        return (
            position_updates.copy(),
            empty_report,
        )

    latitude = (
        position_updates["lat"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    longitude = (
        position_updates["lon"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    update_time = (
        position_updates[
            "lastposupdate"
        ]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    # -------------------------------------------------------------
    # Fast vectorized screening
    # -------------------------------------------------------------

    consecutive_distance = (
        consecutive_haversine_km(
            latitude,
            longitude,
        )
    )

    update_gap_s = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    update_gap_s[1:] = np.diff(
        update_time
    )

    consecutive_speed = np.full(
        n,
        np.nan,
        dtype="float64",
    )

    positive_gap = (
        update_gap_s > 0
    )

    consecutive_speed[
        positive_gap
    ] = (
        consecutive_distance[
            positive_gap
        ]
        / update_gap_s[
            positive_gap
        ]
        * 3600
    )

    initial_bad = (
        (
            consecutive_distance
            > min_jump_distance_km
        )
        & (
            consecutive_speed
            > max_speed_kmh
        )
    )

    nonpositive_count = int(
        np.sum(
            (
                ~np.isnan(update_gap_s)
            )
            & (
                update_gap_s <= 0
            )
        )
    )

    # Most flights should take this fast path.
    if (
        not np.any(initial_bad)
        and nonpositive_count == 0
    ):
        return (
            position_updates.copy(),
            empty_report,
        )

    # -------------------------------------------------------------
    # Sequential accepted-point filter
    # -------------------------------------------------------------
    #
    # This is necessary only for trajectories containing suspicious
    # jumps. A rejected candidate is skipped and the next candidate
    # is compared with the last accepted good position.
    # -------------------------------------------------------------

    lat_rad = np.radians(
        latitude
    )

    lon_rad = np.radians(
        longitude
    )

    cos_lat = np.cos(
        lat_rad
    )

    accepted = np.empty(
        n,
        dtype=np.int64,
    )

    accepted_count = 1
    accepted[0] = 0

    rejected_count = 0
    max_rejected_distance = np.nan
    max_rejected_speed = np.nan
    nonpositive_update_time_rows = 0

    for candidate_index in range(
        1,
        n,
    ):
        previous_index = accepted[
            accepted_count - 1
        ]

        time_difference_s = (
            update_time[
                candidate_index
            ]
            - update_time[
                previous_index
            ]
        )

        if time_difference_s <= 0:
            nonpositive_update_time_rows += 1
            rejected_count += 1
            continue

        dlat = (
            lat_rad[
                candidate_index
            ]
            - lat_rad[
                previous_index
            ]
        )

        dlon = (
            lon_rad[
                candidate_index
            ]
            - lon_rad[
                previous_index
            ]
        )

        a = (
            math.sin(
                dlat / 2
            ) ** 2
            + cos_lat[
                previous_index
            ]
            * cos_lat[
                candidate_index
            ]
            * math.sin(
                dlon / 2
            ) ** 2
        )

        # Numerical protection.
        a = min(
            1.0,
            max(
                0.0,
                a,
            ),
        )

        c = 2 * math.atan2(
            math.sqrt(a),
            math.sqrt(
                1 - a
            ),
        )

        distance_km = (
            EARTH_RADIUS_KM * c
        )

        implied_speed_kmh = (
            distance_km
            / time_difference_s
            * 3600
        )

        impossible_jump = (
            distance_km
            > min_jump_distance_km
            and implied_speed_kmh
            > max_speed_kmh
        )

        if impossible_jump:
            rejected_count += 1

            if (
                np.isnan(
                    max_rejected_distance
                )
                or distance_km
                > max_rejected_distance
            ):
                max_rejected_distance = (
                    distance_km
                )

            if (
                np.isnan(
                    max_rejected_speed
                )
                or implied_speed_kmh
                > max_rejected_speed
            ):
                max_rejected_speed = (
                    implied_speed_kmh
                )

            continue

        accepted[
            accepted_count
        ] = candidate_index

        accepted_count += 1

    accepted_indices = (
        accepted[
            :accepted_count
        ]
    )

    cleaned = (
        position_updates.iloc[
            accepted_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    report = {
        "spatial_input_rows": n,

        "spatial_output_rows": (
            len(cleaned)
        ),

        "removed_spatial_rows": (
            rejected_count
        ),

        "max_rejected_distance_km": (
            max_rejected_distance
        ),

        "max_rejected_speed_kmh": (
            max_rejected_speed
        ),

        "nonpositive_update_time_rows": (
            nonpositive_update_time_rows
        ),
    }

    return cleaned, report


# ---------------------------------------------------------------------
# Complete preprocessing for one flight
# ---------------------------------------------------------------------

def preprocess_single_flight(
    file_path,
):
    """
    Apply the current MVP preprocessing rules to one source CSV.

    Important:
    The source file is read exactly once.
    """

    # Read once.
    flight = pd.read_csv(
        file_path
    )

    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    # -------------------------------------------------------------
    # File-level classification
    # -------------------------------------------------------------

    classification = (
        classify_loaded_trajectory(
            flight,
            parsed_time,
            file_path.name,
        )
    )

    if classification[
        "exclude_severe_multisegment"
    ]:
        report = {
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
            "removed_carried_forward_rows": None,

            "spatial_input_rows": None,
            "spatial_output_rows": None,
            "removed_spatial_rows": None,
            "max_rejected_distance_km": None,
            "max_rejected_speed_kmh": None,
            "nonpositive_update_time_rows": None,
        }

        return None, report

    # -------------------------------------------------------------
    # Base position-update preparation
    # -------------------------------------------------------------

    position_updates, row_report = (
        build_clean_position_updates_from_frame(
            flight,
            parsed_time,
        )
    )

    # -------------------------------------------------------------
    # Spatial discontinuity cleaning
    # -------------------------------------------------------------

    cleaned, spatial_report = (
        filter_spatially_inconsistent_updates(
            position_updates,
        )
    )

    report = {
        **classification,
        **row_report,
        **spatial_report,

        "status": "kept",
    }

    return cleaned, report


# ---------------------------------------------------------------------
# January-wide preprocessing verification
# ---------------------------------------------------------------------

def main():
    """
    Run the current MVP preprocessing policy across all
    original January trajectory files.

    Cleaned trajectories are NOT saved yet.
    This run verifies the dataset-wide effect of the rules.
    """

    flight_files = find_flight_csvs(
        RAW_JANUARY_DIR
    )

    total_files = len(
        flight_files
    )

    reports = []

    print(
        f"January source files: {total_files}"
    )

    print(
        "Starting preprocessing..."
    )

    for number, file_path in enumerate(
        flight_files,
        start=1,
    ):
        _, report = (
            preprocess_single_flight(
                file_path
            )
        )

        reports.append(
            report
        )

        # Visible progress so the process never appears frozen.
        if (
            number % 25 == 0
            or number == total_files
        ):
            print(
                f"Processed {number} / {total_files}",
                flush=True,
            )

    report_table = pd.DataFrame(
        reports
    )

    kept = report_table[
        report_table["status"]
        == "kept"
    ].copy()

    excluded = report_table[
        report_table["status"]
        == "excluded_multisegment"
    ].copy()

    print("\n" + "=" * 70)
    print("JANUARY PREPROCESSING SUMMARY")
    print("=" * 70)

    print(
        "\nJanuary source files:"
    )
    print(
        len(report_table)
    )

    print(
        "\nFlights kept:"
    )
    print(
        len(kept)
    )

    print(
        "\nFlights excluded as severe multi-segment candidates:"
    )
    print(
        len(excluded)
    )

    print(
        "\nTotal original rows in kept flights:"
    )
    print(
        int(
            kept[
                "original_rows"
            ].sum()
        )
    )

    print(
        "\nTotal unusable rows removed:"
    )
    print(
        int(
            kept[
                "removed_unusable_rows"
            ].sum()
        )
    )

    print(
        "\nTotal carried-forward rows removed:"
    )
    print(
        int(
            kept[
                "removed_carried_forward_rows"
            ].sum()
        )
    )

    print(
        "\nTotal spatially inconsistent updates removed:"
    )
    print(
        int(
            kept[
                "removed_spatial_rows"
            ].sum()
        )
    )

    print(
        "\nFinal cleaned position-update rows:"
    )
    print(
        int(
            kept[
                "spatial_output_rows"
            ].sum()
        )
    )

    print(
        "\nFlights with at least one spatial update removed:"
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
        "\nFlights with non-positive position-update timing:"
    )
    print(
        int(
            (
                kept[
                    "nonpositive_update_time_rows"
                ] > 0
            ).sum()
        )
    )

    print(
        "\nLargest number of spatial removals in one flight:"
    )
    print(
        int(
            kept[
                "removed_spatial_rows"
            ].max()
        )
    )

    print(
        "\nTop 10 flights by spatial removals:"
    )

    spatial_columns = [
        "file_name",
        "original_rows",
        "position_update_rows",
        "removed_spatial_rows",
        "spatial_output_rows",
        "max_rejected_distance_km",
        "max_rejected_speed_kmh",
    ]

    print(
        kept
        .sort_values(
            "removed_spatial_rows",
            ascending=False,
        )
        [spatial_columns]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nExcluded multi-segment files:"
    )

    excluded_columns = [
        "file_name",
        "callsigns",
        "total_duration_h",
        "max_row_gap_s",
    ]

    print(
        excluded[
            excluded_columns
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