import math

import numpy as np
import pandas as pd

from src.config import RAW_JANUARY_DIR
from src.data_io import find_flight_csvs


# MVP preprocessing thresholds

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

# Exclude an entire trajectory if spatial corruption is so extensive
# that more than 20% of its unique position updates must be rejected.
MAX_SPATIAL_REMOVAL_FRACTION = 0.20

EARTH_RADIUS_KM = 6371.0088

# Resampling settings

RESAMPLE_INTERVALS_S = [
    5,
    10,
    30,
]
WORKING_RESAMPLE_INTERVAL_S = 10

# Never interpolate across a position-update gap larger than this.
MAX_INTERPOLATION_GAP_S = 60.0

# File-level quality classification

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


# Base position-update preparation

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


# Vectorized consecutive-distance helper

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

def pointwise_haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Vectorized Haversine distance between corresponding points.

    All inputs must have matching lengths.
    """

    lat1 = np.radians(
        np.asarray(
            lat1,
            dtype="float64",
        )
    )

    lon1 = np.radians(
        np.asarray(
            lon1,
            dtype="float64",
        )
    )

    lat2 = np.radians(
        np.asarray(
            lat2,
            dtype="float64",
        )
    )

    lon2 = np.radians(
        np.asarray(
            lon2,
            dtype="float64",
        )
    )

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

    return (
        EARTH_RADIUS_KM * c
    )

# Optimized spatial discontinuity filter

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

    # Fast vectorized screening

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

    # Sequential accepted-point filter
    #
    # This is necessary only for trajectories containing suspicious
    # jumps. A rejected candidate is skipped and the next candidate
    # is compared with the last accepted good position.

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

def resample_trajectory(
    cleaned_trajectory,
    interval_s,
    max_interpolation_gap_s=MAX_INTERPOLATION_GAP_S,
):
    """
    Resample a cleaned trajectory onto a regular time grid.

    Linear interpolation is allowed only when the two surrounding
    cleaned position updates are separated by no more than
    max_interpolation_gap_s.

    Large observation gaps are preserved rather than bridged.
    """

    if len(cleaned_trajectory) < 2:
        return cleaned_trajectory.copy()

    trajectory = (
        cleaned_trajectory
        .sort_values("lastposupdate")
        .reset_index(drop=True)
        .copy()
    )

    update_time = (
        trajectory["lastposupdate"]
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

    # Regular Unix-second sampling grid.
    start_time = math.ceil(
        update_time[0] / interval_s
    ) * interval_s

    end_time = math.floor(
        update_time[-1] / interval_s
    ) * interval_s

    if end_time < start_time:
        return trajectory.iloc[
            [0]
        ].copy()

    target_times = np.arange(
        start_time,
        end_time + interval_s,
        interval_s,
        dtype="float64",
    )

    output_rows = []

    # Locate the source interval surrounding each target timestamp.
    right_indices = np.searchsorted(
        update_time,
        target_times,
        side="left",
    )

    for target_time, right_index in zip(
        target_times,
        right_indices,
    ):
        # Exact match with an observed position update.
        if (
            right_index < len(update_time)
            and update_time[right_index]
            == target_time
        ):
            output_rows.append(
                {
                    "sample_time": target_time,
                    "lat": latitude[right_index],
                    "lon": longitude[right_index],
                    "interpolated": False,
                }
            )

            continue

        # Cannot interpolate outside the observed range.
        if (
            right_index == 0
            or right_index >= len(update_time)
        ):
            continue

        left_index = right_index - 1

        left_time = update_time[
            left_index
        ]

        right_time = update_time[
            right_index
        ]

        source_gap_s = (
            right_time - left_time
        )

        # Preserve substantial observation gaps.
        if (
            source_gap_s <= 0
            or source_gap_s
            > max_interpolation_gap_s
        ):
            continue

        fraction = (
            target_time - left_time
        ) / source_gap_s

        interpolated_lat = (
            latitude[left_index]
            + fraction
            * (
                latitude[right_index]
                - latitude[left_index]
            )
        )

        interpolated_lon = (
            longitude[left_index]
            + fraction
            * (
                longitude[right_index]
                - longitude[left_index]
            )
        )

        output_rows.append(
            {
                "sample_time": target_time,
                "lat": interpolated_lat,
                "lon": interpolated_lon,
                "interpolated": True,
            }
        )

    return pd.DataFrame(
        output_rows
    )

def evaluate_resampled_trajectory(
    cleaned_trajectory,
    resampled_trajectory,
    interval_s,
):
    """
    Measure geometric error introduced by time resampling.

    The resampled trajectory is interpolated back onto cleaned
    position-update timestamps, but only inside continuous
    resampled segments.

    Missing-data gaps are therefore not artificially bridged.
    """

    if (
        len(cleaned_trajectory) == 0
        or len(resampled_trajectory) < 2
    ):
        return {
            "evaluated_rows": 0,
            "coverage_percent": 0.0,
            "median_error_km": np.nan,
            "p95_error_km": np.nan,
            "max_error_km": np.nan,
        }

    cleaned = (
        cleaned_trajectory
        .sort_values("lastposupdate")
        .reset_index(drop=True)
    )

    resampled = (
        resampled_trajectory
        .sort_values("sample_time")
        .reset_index(drop=True)
    )

    cleaned_time = (
        cleaned["lastposupdate"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    cleaned_lat = (
        cleaned["lat"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    cleaned_lon = (
        cleaned["lon"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    sample_time = (
        resampled["sample_time"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    sample_lat = (
        resampled["lat"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    sample_lon = (
        resampled["lon"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    # A gap larger than the expected sampling interval indicates
    # that Step 2.4A intentionally preserved a missing-data region.
    sample_gap = np.diff(
        sample_time
    )

    new_segment = np.concatenate(
        (
            [True],
            sample_gap
            > interval_s * 1.5,
        )
    )

    segment_id = np.cumsum(
        new_segment
    )

    all_errors = []

    for current_segment in np.unique(
        segment_id
    ):
        mask = (
            segment_id
            == current_segment
        )

        segment_time = sample_time[
            mask
        ]

        segment_lat = sample_lat[
            mask
        ]

        segment_lon = sample_lon[
            mask
        ]

        if len(segment_time) < 2:
            continue

        # Evaluate only cleaned observations covered by this
        # continuous resampled segment.
        cleaned_mask = (
            (cleaned_time >= segment_time[0])
            & (cleaned_time <= segment_time[-1])
        )

        if not np.any(
            cleaned_mask
        ):
            continue

        target_time = cleaned_time[
            cleaned_mask
        ]

        reconstructed_lat = np.interp(
            target_time,
            segment_time,
            segment_lat,
        )

        reconstructed_lon = np.interp(
            target_time,
            segment_time,
            segment_lon,
        )

        error_km = pointwise_haversine_km(
            cleaned_lat[
                cleaned_mask
            ],
            cleaned_lon[
                cleaned_mask
            ],
            reconstructed_lat,
            reconstructed_lon,
        )

        all_errors.append(
            error_km
        )

    if not all_errors:
        return {
            "evaluated_rows": 0,
            "coverage_percent": 0.0,
            "median_error_km": np.nan,
            "p95_error_km": np.nan,
            "max_error_km": np.nan,
        }

    errors = np.concatenate(
        all_errors
    )

    evaluated_rows = len(
        errors
    )

    coverage_percent = (
        evaluated_rows
        / len(cleaned)
        * 100
    )

    return {
        "evaluated_rows": (
            evaluated_rows
        ),

        "coverage_percent": (
            coverage_percent
        ),

        "median_error_km": (
            np.median(
                errors
            )
        ),

        "p95_error_km": (
            np.percentile(
                errors,
                95,
            )
        ),

        "max_error_km": (
            np.max(
                errors
            )
        ),
    }

# Complete preprocessing for one flight

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

    # File-level classification

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
            "spatial_removal_fraction": None,
        }

        return None, report

    # Base position-update preparation

    position_updates, row_report = (
        build_clean_position_updates_from_frame(
            flight,
            parsed_time,
        )
    )

    # Spatial discontinuity cleaning

    cleaned, spatial_report = (
        filter_spatially_inconsistent_updates(
            position_updates,
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

    # Apply the same final trajectory-level corruption rule
    # used for July.
    if (
        pd.notna(
            spatial_removal_fraction
        )
        and spatial_removal_fraction
        > MAX_SPATIAL_REMOVAL_FRACTION
    ):
        report = {
            **classification,
            **row_report,
            **spatial_report,

            "status": (
                "excluded_spatial_corruption"
            ),

            "spatial_removal_fraction": (
                spatial_removal_fraction
            ),
        }

        return None, report

    report = {
    **classification,
    **row_report,
    **spatial_report,

    "status": "kept",

    "spatial_removal_fraction": (
        spatial_removal_fraction
    ),
}

    return cleaned, report


# January-wide preprocessing verification
def main():
    """
    Verify final January trajectory eligibility using the
    same 20% spatial-corruption rule applied to July.
    """

    flight_files = find_flight_csvs(
        RAW_JANUARY_DIR
    )

    reports = []

    total_files = len(
        flight_files
    )

    print(
        f"January source files: {total_files}"
    )

    print(
        "Checking final January trajectory eligibility..."
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
        table["status"] == "kept"
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
        "FINAL JANUARY ELIGIBILITY SUMMARY"
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
        "\nLargest retained spatial-removal percentage:"
    )

    print(
        f"{kept['spatial_removal_fraction'].max() * 100:.3f}%"
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
            "position_update_rows",
            "removed_spatial_rows",
            "spatial_removal_fraction",
            "spatial_output_rows",
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


if __name__ == "__main__":
    main()