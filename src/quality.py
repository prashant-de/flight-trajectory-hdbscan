import numpy as np
import pandas as pd

from src.config import RAW_JANUARY_DIR
from src.data_io import find_flight_csvs


# Development / inspection files

# Four flights used during early quality validation.
DEVELOPMENT_FILES = [
    RAW_JANUARY_DIR / "20260106T062826Z_a8584a_FFT2504.csv",
    RAW_JANUARY_DIR / "20260101T000338Z_ada5ee_JBU524.csv",
    RAW_JANUARY_DIR / "20260104T091537Z_ada227_JBU1724.csv",
    RAW_JANUARY_DIR / "20260105T150813Z_a033ae_AAL2.csv",
]

# January flight containing the largest row-time gap found so far.
CURRENT_SUSPICIOUS_FILE = (
    RAW_JANUARY_DIR
    / "20260121T043329Z_e8021d_LPE2690.csv"
)


# Shared helpers

def datetime_to_unix_seconds(parsed_time):
    """
    Convert UTC pandas datetime values to Unix seconds.

    Using subtraction from the Unix epoch is safer than relying on
    pandas' internal datetime resolution. Missing timestamps remain NaN.
    """

    epoch = pd.Timestamp(
        "1970-01-01",
        tz="UTC",
    )

    return (
        parsed_time - epoch
    ).dt.total_seconds()


# ---------------------------------------------------------------------
# Step 2.1 — Timestamp and freshness quality
# ---------------------------------------------------------------------

def summarize_timestamp_freshness(file_path):
    """
    Return timestamp/freshness quality metrics for one flight.

    Supplied position/contact-age fields are preserved only for
    verification. Freshness decisions should use the recalculated
    values based on time, lastposupdate, and lastcontact.
    """

    columns = [
        "time",
        "lastposupdate",
        "lastcontact",
        "position_age_seconds",
        "contact_age_seconds",
    ]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
    )

    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    time_unix_seconds = datetime_to_unix_seconds(
        parsed_time
    )

    # Recalculate freshness using compatible Unix-second units.
    position_age = (
        time_unix_seconds
        - flight["lastposupdate"]
    )

    contact_age = (
        time_unix_seconds
        - flight["lastcontact"]
    )

    return {
        "file_name": file_path.name,
        "rows": len(flight),

        # Timestamp parsing.
        "unparsed_times": (
            parsed_time.isna().sum()
        ),

        # Position/contact update availability.
        "missing_lastposupdate": (
            flight["lastposupdate"].isna().sum()
        ),

        "missing_lastposupdate_fraction": (
            flight["lastposupdate"].isna().mean()
        ),

        "missing_lastcontact": (
            flight["lastcontact"].isna().sum()
        ),

        "valid_position_age_rows": (
            position_age.notna().sum()
        ),

        # Recalculated freshness sanity checks.
        "negative_position_age": (
            position_age < 0
        ).sum(),

        "negative_contact_age": (
            contact_age < 0
        ).sum(),

        # Freshness summary.
        "median_position_age_s": (
            position_age.median()
        ),

        "max_position_age_s": (
            position_age.max()
        ),

        "median_contact_age_s": (
            contact_age.median()
        ),

        "max_contact_age_s": (
            contact_age.max()
        ),

        # Diagnostic thresholds only.
        # These are NOT cleaning thresholds.
        "position_age_gt_10s": (
            position_age > 10
        ).sum(),

        "position_age_gt_30s": (
            position_age > 30
        ).sum(),

        "position_age_gt_60s": (
            position_age > 60
        ).sum(),

        # Check the supplied age columns.
        "negative_supplied_position_age": (
            flight["position_age_seconds"] < 0
        ).sum(),

        "negative_supplied_contact_age": (
            flight["contact_age_seconds"] < 0
        ).sum(),

        "missing_supplied_position_age": (
            flight[
                "position_age_seconds"
            ].isna().sum()
        ),

        "missing_supplied_contact_age": (
            flight[
                "contact_age_seconds"
            ].isna().sum()
        ),
    }


def build_january_timestamp_summary():
    """
    Build one timestamp/freshness summary record
    for every original January flight.
    """

    flight_files = find_flight_csvs(
        RAW_JANUARY_DIR
    )

    summaries = [
        summarize_timestamp_freshness(
            file_path
        )
        for file_path in flight_files
    ]

    return pd.DataFrame(summaries)


# Step 2.2 — Row gaps and coordinate quality

def summarize_gap_coordinate_quality(file_path):
    """
    Summarize:
    - row-time gaps,
    - coordinate availability,
    - coordinate range validity,
    - exact consecutive repeated positions.
    """

    columns = [
        "time",
        "lat",
        "lon",
    ]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
    )

    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    # Row-time gaps

    row_gap_seconds = (
        parsed_time
        .diff()
        .dt.total_seconds()
    )
    # Coordinate availability
    missing_lat = flight["lat"].isna()
    missing_lon = flight["lon"].isna()

    # Coordinate range validation

    invalid_lat = (
        flight["lat"].notna()
        & ~flight["lat"].between(
            -90,
            90,
        )
    )

    invalid_lon = (
        flight["lon"].notna()
        & ~flight["lon"].between(
            -180,
            180,
        )
    )

    # Consecutive repeated coordinates

    same_as_previous = (
        flight["lat"].eq(
            flight["lat"].shift(1)
        )
        & flight["lon"].eq(
            flight["lon"].shift(1)
        )
        & flight["lat"].notna()
        & flight["lon"].notna()
    )

    # Start a new group whenever the coordinates change.
    coordinate_group = (
        ~same_as_previous
    ).cumsum()

    longest_run_rows = 0
    longest_run_duration_s = 0.0

    for _, group_indices in flight.groupby(
        coordinate_group
    ).groups.items():

        if len(group_indices) <= 1:
            continue

        first_index = group_indices[0]
        last_index = group_indices[-1]

        duration_s = (
            parsed_time.loc[last_index]
            - parsed_time.loc[first_index]
        ).total_seconds()

        if (
            len(group_indices)
            > longest_run_rows
        ):
            longest_run_rows = len(
                group_indices
            )

        if (
            duration_s
            > longest_run_duration_s
        ):
            longest_run_duration_s = (
                duration_s
            )
        if len(parsed_time) > 0:
                total_duration_s = (
                    parsed_time.iloc[-1]
                    - parsed_time.iloc[0]
                ).total_seconds()
        else:
                total_duration_s = float("nan")

    return {
        "file_name": file_path.name,
        "rows": len(flight),

        # Timestamp parsing.
        "unparsed_times": (
            parsed_time.isna().sum()
        ),

        # Row ordering / timing.
        "duplicate_timestamps": (
            row_gap_seconds == 0
        ).sum(),

        "negative_time_gaps": (
            row_gap_seconds < 0
        ).sum(),

        "max_row_gap_s": (
            row_gap_seconds.max()
        ),

        # Diagnostic thresholds only.
        "row_gaps_gt_1s": (
            row_gap_seconds > 1
        ).sum(),

        "row_gaps_gt_5s": (
            row_gap_seconds > 5
        ).sum(),

        "row_gaps_gt_30s": (
            row_gap_seconds > 30
        ).sum(),

        "row_gaps_gt_60s": (
            row_gap_seconds > 60
        ).sum(),

        # Coordinate availability.
        "missing_lat": (
            missing_lat.sum()
        ),

        "missing_lon": (
            missing_lon.sum()
        ),

        "missing_either_coordinate": (
            missing_lat
            | missing_lon
        ).sum(),

        # Coordinate validity.
        "invalid_lat": (
            invalid_lat.sum()
        ),

        "invalid_lon": (
            invalid_lon.sum()
        ),

        # Repeated coordinates.
        "repeated_coordinate_rows": (
            same_as_previous.sum()
        ),

        "longest_identical_coordinate_run_rows": (
            longest_run_rows
        ),

        "longest_identical_coordinate_run_s": (
            longest_run_duration_s
        ),
        "start_time": parsed_time.iloc[0],
        "end_time": parsed_time.iloc[-1],
        "total_duration_h": total_duration_s / 3600,
    }


def build_january_gap_coordinate_summary():
    """
    Build one gap/coordinate quality summary
    for every original January flight.
    """

    flight_files = find_flight_csvs(
        RAW_JANUARY_DIR
    )

    summaries = [
        summarize_gap_coordinate_quality(
            file_path
        )
        for file_path in flight_files
    ]

    return pd.DataFrame(summaries)


# Step 2.2 — Detailed repeated-coordinate inspection

def inspect_longest_repeated_coordinate_run(
    file_path,
):
    """
    Return details for the longest consecutive
    identical-coordinate run in one flight.
    """

    columns = [
        "time",
        "lat",
        "lon",
        "lastposupdate",
    ]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
    )

    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    time_unix_seconds = (
        datetime_to_unix_seconds(
            parsed_time
        )
    )

    position_age = (
        time_unix_seconds
        - flight["lastposupdate"]
    )

    same_as_previous = (
        flight["lat"].eq(
            flight["lat"].shift(1)
        )
        & flight["lon"].eq(
            flight["lon"].shift(1)
        )
        & flight["lat"].notna()
        & flight["lon"].notna()
    )

    coordinate_group = (
        ~same_as_previous
    ).cumsum()

    longest_indices = None
    longest_duration_s = -1.0

    for _, group_indices in flight.groupby(
        coordinate_group
    ).groups.items():

        if len(group_indices) <= 1:
            continue

        first_index = group_indices[0]
        last_index = group_indices[-1]

        duration_s = (
            parsed_time.loc[last_index]
            - parsed_time.loc[first_index]
        ).total_seconds()

        if duration_s > longest_duration_s:
            longest_duration_s = (
                duration_s
            )

            longest_indices = (
                group_indices
            )

    if longest_indices is None:
        return {
            "file_name": file_path.name,
            "repeated_run_found": False,
        }

    first_index = longest_indices[0]
    last_index = longest_indices[-1]

    return {
        "file_name": file_path.name,
        "repeated_run_found": True,

        "start_time": (
            parsed_time.loc[first_index]
        ),

        "end_time": (
            parsed_time.loc[last_index]
        ),

        "run_rows": len(
            longest_indices
        ),

        "run_duration_s": (
            longest_duration_s
        ),

        "latitude": (
            flight.loc[
                first_index,
                "lat",
            ]
        ),

        "longitude": (
            flight.loc[
                first_index,
                "lon",
            ]
        ),

        "start_position_age_s": (
            position_age.loc[
                first_index
            ]
        ),

        "end_position_age_s": (
            position_age.loc[
                last_index
            ]
        ),

        "max_position_age_in_run_s": (
            position_age.loc[
                longest_indices
            ].max()
        ),
    }


# Step 2.2 — Extreme row-gap inspection

def inspect_largest_gap_boundary(
    file_path,
    context_rows=4,
):
    """
    Inspect several observations immediately
    before and after the largest row-time gap.

    This helps determine whether a large gap is:
    - missing observation coverage,
    - a trajectory extraction problem,
    - or multiple operational segments joined together.
    """

    columns = [
        "time",
        "icao24",
        "callsign",
        "flight_key",
        "lat",
        "lon",
        "geoaltitude",
        "velocity",
        "heading",
        "onground",
        "lastposupdate",
    ]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
        dtype={
            "icao24": "string",
            "callsign": "string",
            "flight_key": "string",
        },
    )

    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    row_gap_seconds = (
        parsed_time
        .diff()
        .dt.total_seconds()
    )

    # idxmax() identifies the first row AFTER
    # the largest time gap.
    gap_end_index = (
        row_gap_seconds.idxmax()
    )

    gap_seconds = (
        row_gap_seconds.loc[
            gap_end_index
        ]
    )

    gap_position = (
        flight.index.get_loc(
            gap_end_index
        )
    )

    start_position = max(
        0,
        gap_position - context_rows,
    )

    end_position = min(
        len(flight),
        gap_position
        + context_rows
        + 1,
    )

    context = flight.iloc[
        start_position:end_position
    ].copy()

    # Add parsed timestamps to make the boundary
    # easier to interpret.
    context["parsed_time"] = (
        parsed_time.iloc[
            start_position:end_position
        ].to_numpy()
    )

    # Show the time difference from the previous row.
    context["gap_from_previous_s"] = (
        row_gap_seconds.iloc[
            start_position:end_position
        ].to_numpy()
    )

    columns_to_show = [
        "parsed_time",
        "gap_from_previous_s",
        "icao24",
        "callsign",
        "flight_key",
        "lat",
        "lon",
        "geoaltitude",
        "velocity",
        "heading",
        "onground",
        "lastposupdate",
    ]

    return {
        "file_name": file_path.name,

        "largest_gap_s": (
            gap_seconds
        ),

        "largest_gap_hours": (
            gap_seconds / 3600
        ),

        "context": context[
            columns_to_show
        ],
    }
def inspect_callsign_segments(file_path):
    """Summarize callsign blocks within one trajectory file."""

    columns = [
        "time",
        "icao24",
        "callsign",
        "lat",
        "lon",
        "onground",
    ]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
        dtype={
            "icao24": "string",
            "callsign": "string",
        },
    )

    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    # Normalize callsign strings.
    callsign = (
        flight["callsign"]
        .str.strip()
        .replace("", pd.NA)
    )

    # Start a new segment whenever the callsign changes.
    segment_id = (
        callsign.ne(callsign.shift(1))
        | callsign.isna().ne(
            callsign.shift(1).isna()
        )
    ).cumsum()

    records = []

    for _, indices in flight.groupby(
        segment_id
    ).groups.items():

        first_index = indices[0]
        last_index = indices[-1]

        records.append(
            {
                "callsign": callsign.loc[first_index],
                "start_time": parsed_time.loc[first_index],
                "end_time": parsed_time.loc[last_index],
                "rows": len(indices),
                "start_lat": flight.loc[first_index, "lat"],
                "start_lon": flight.loc[first_index, "lon"],
                "end_lat": flight.loc[last_index, "lat"],
                "end_lon": flight.loc[last_index, "lon"],
            }
        )

    return pd.DataFrame(records)
def summarize_callsign_consistency(file_path):
    """
    Summarize callsign consistency and basic temporal continuity
    for one trajectory file.
    """

    columns = [
        "time",
        "callsign",
    ]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
        dtype={
            "callsign": "string",
        },
    )

    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    callsign = (
        flight["callsign"]
        .str.strip()
        .replace("", pd.NA)
    )

    unique_callsigns = (
        callsign
        .dropna()
        .unique()
        .tolist()
    )

    row_gaps = (
        parsed_time
        .diff()
        .dt.total_seconds()
    )

    if len(flight) > 0:
        total_duration_s = (
            parsed_time.iloc[-1]
            - parsed_time.iloc[0]
        ).total_seconds()
    else:
        total_duration_s = float("nan")

    return {
        "file_name": file_path.name,
        "rows": len(flight),

        "callsign_count": len(
            unique_callsigns
        ),

        "callsigns": ", ".join(
            unique_callsigns
        ),

        "start_time": (
            parsed_time.iloc[0]
        ),

        "end_time": (
            parsed_time.iloc[-1]
        ),

        "total_duration_h": (
            total_duration_s / 3600
        ),

        "max_row_gap_s": (
            row_gaps.max()
        ),

        "row_gaps_gt_60s": (
            row_gaps > 60
        ).sum(),
    }

def summarize_spatial_jumps(file_path):
    """
    Summarize distances and implied speeds between consecutive
    valid trajectory coordinates.

    These are diagnostic metrics only. No cleaning threshold is
    selected here.
    """


    columns = [
        "time",
        "lat",
        "lon",
    ]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
    )

    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    # Time difference between consecutive observations.
    time_gap_s = (
        parsed_time
        .diff()
        .dt.total_seconds()
    )

    # Previous/current coordinates.
    lat1 = np.radians(
        flight["lat"].shift(1)
    )

    lon1 = np.radians(
        flight["lon"].shift(1)
    )

    lat2 = np.radians(
        flight["lat"]
    )

    lon2 = np.radians(
        flight["lon"]
    )

    # Haversine distance between consecutive coordinates.
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    earth_radius_km = 6371.0088

    distance_km = (
        earth_radius_km * c
    )

    # Implied speed is only meaningful when elapsed time is positive.
    valid_speed = (
        distance_km.notna()
        & time_gap_s.notna()
        & (time_gap_s > 0)
    )

    implied_speed_kmh = pd.Series(
        np.nan,
        index=flight.index,
        dtype="float64",
    )

    implied_speed_kmh.loc[valid_speed] = (
        distance_km.loc[valid_speed]
        / time_gap_s.loc[valid_speed]
        * 3600
    )

    return {
        "file_name": file_path.name,
        "rows": len(flight),

        "max_consecutive_distance_km": (
            distance_km.max()
        ),

        "max_implied_speed_kmh": (
            implied_speed_kmh.max()
        ),

        # Reporting bins only — not cleaning rules.
        "jumps_gt_10km": (
            distance_km > 10
        ).sum(),

        "jumps_gt_50km": (
            distance_km > 50
        ).sum(),

        "jumps_gt_100km": (
            distance_km > 100
        ).sum(),

        "implied_speed_gt_1500_kmh": (
            implied_speed_kmh > 1500
        ).sum(),

        "implied_speed_gt_5000_kmh": (
            implied_speed_kmh > 5000
        ).sum(),
    }

def summarize_position_update_jumps(file_path):
    """
    Measure spatial movement between consecutive position updates
    while preserving the original trajectory row order.

    lastposupdate is used to determine when a new position was
    reported, but rows are NOT reordered by lastposupdate.
    """

    columns = [
        "lat",
        "lon",
        "lastposupdate",
    ]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
    )

    # Keep rows where position and update time are available.
    valid = flight[
        flight["lat"].notna()
        & flight["lon"].notna()
        & flight["lastposupdate"].notna()
    ].copy()

    # Keep only rows where the position-update timestamp changes
    # relative to the previous valid observation.
    #
    # Important:
    # We preserve original CSV/trajectory order.
    new_update = valid[
        "lastposupdate"
    ].ne(
        valid["lastposupdate"].shift(1)
    )

    updates = (
        valid.loc[new_update]
        .copy()
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Time between consecutive position updates
    # -------------------------------------------------------------

    update_gap_s = (
        updates["lastposupdate"]
        .diff()
    )

    # -------------------------------------------------------------
    # Haversine distance between consecutive updated positions
    # -------------------------------------------------------------

    lat1 = np.radians(
        updates["lat"].shift(1)
    )

    lon1 = np.radians(
        updates["lon"].shift(1)
    )

    lat2 = np.radians(
        updates["lat"]
    )

    lon2 = np.radians(
        updates["lon"]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    # Numerical protection for the Haversine formula.
    a = np.clip(a, 0, 1)

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    earth_radius_km = 6371.0088

    distance_km = (
        earth_radius_km * c
    )

    # -------------------------------------------------------------
    # Implied speed
    # -------------------------------------------------------------

    valid_speed = (
        distance_km.notna()
        & update_gap_s.notna()
        & (update_gap_s > 0)
    )

    implied_speed_kmh = pd.Series(
        np.nan,
        index=updates.index,
        dtype="float64",
    )

    implied_speed_kmh.loc[valid_speed] = (
        distance_km.loc[valid_speed]
        / update_gap_s.loc[valid_speed]
        * 3600
    )

    return {
        "file_name": file_path.name,
        "original_rows": len(flight),
        "valid_position_rows": len(valid),
        "position_update_events": len(updates),

        # Check whether lastposupdate behaves monotonically
        # while following the original trajectory order.
        "negative_update_gaps": (
            update_gap_s < 0
        ).sum(),

        "zero_update_gaps": (
            update_gap_s == 0
        ).sum(),

        "max_update_gap_s": (
            update_gap_s.max()
        ),

        "max_update_distance_km": (
            distance_km.max()
        ),

        "max_update_implied_speed_kmh": (
            implied_speed_kmh.max()
        ),

        # Diagnostic counts only.
        "update_jumps_gt_10km": (
            distance_km > 10
        ).sum(),

        "update_jumps_gt_50km": (
            distance_km > 50
        ).sum(),

        "update_jumps_gt_100km": (
            distance_km > 100
        ).sum(),

        "update_speed_gt_1500_kmh": (
            implied_speed_kmh > 1500
        ).sum(),

        "update_speed_gt_5000_kmh": (
            implied_speed_kmh > 5000
        ).sum(),
    }
def inspect_largest_position_update_jump(file_path):
    """
    Inspect the consecutive position-update pair with the
    largest geographic distance.
    """

    columns = [
        "time",
        "lat",
        "lon",
        "lastposupdate",
        "callsign",
        "icao24",
        "geoaltitude",
        "velocity",
        "heading",
    ]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
        dtype={
            "callsign": "string",
            "icao24": "string",
        },
    )

    valid = flight[
        flight["lat"].notna()
        & flight["lon"].notna()
        & flight["lastposupdate"].notna()
    ].copy()

    # Preserve original trajectory order.
    new_update = valid[
        "lastposupdate"
    ].ne(
        valid["lastposupdate"].shift(1)
    )

    updates = (
        valid.loc[new_update]
        .copy()
        .reset_index()
        .rename(
            columns={
                "index": "original_row_index"
            }
        )
    )

    lat1 = np.radians(
        updates["lat"].shift(1)
    )

    lon1 = np.radians(
        updates["lon"].shift(1)
    )

    lat2 = np.radians(
        updates["lat"]
    )

    lon2 = np.radians(
        updates["lon"]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    earth_radius_km = 6371.0088

    distance_km = (
        earth_radius_km * c
    )

    update_gap_s = (
        updates["lastposupdate"]
        .diff()
    )

    largest_jump_index = (
        distance_km.idxmax()
    )

    previous_index = (
        largest_jump_index - 1
    )

    jump_distance_km = (
        distance_km.loc[
            largest_jump_index
        ]
    )

    jump_time_s = (
        update_gap_s.loc[
            largest_jump_index
        ]
    )

    implied_speed_kmh = (
        jump_distance_km
        / jump_time_s
        * 3600
    )

    columns_to_show = [
        "original_row_index",
        "time",
        "lastposupdate",
        "callsign",
        "icao24",
        "lat",
        "lon",
        "geoaltitude",
        "velocity",
        "heading",
    ]

    pair = updates.loc[
        [
            previous_index,
            largest_jump_index,
        ],
        columns_to_show,
    ]

    print("File:", file_path.name)

    print(
        "\nLargest update-to-update distance (km):"
    )
    print(jump_distance_km)

    print(
        "\nTime between position updates (seconds):"
    )
    print(jump_time_s)

    print(
        "\nImplied speed (km/h):"
    )
    print(implied_speed_kmh)

    print(
        "\nObservations around largest jump:"
    )
    print(
        pair.to_string(
            index=False
        )
    )

def inspect_position_updates_around_largest_jump(
    file_path,
    context_updates=5,
):
    """
    Show several position updates before and after the
    largest spatial jump in one trajectory.
    """

    columns = [
    "time",
    "lat",
    "lon",
    "lastposupdate",
    "lastcontact",
    "callsign",
    "icao24",
    "geoaltitude",
    "baroaltitude",
    "velocity",
    "heading",
    "vertrate",
    "serials",
]

    flight = pd.read_csv(
        file_path,
        usecols=columns,
        dtype={
            "callsign": "string",
            "icao24": "string",
        },
    )

    valid = flight[
        flight["lat"].notna()
        & flight["lon"].notna()
        & flight["lastposupdate"].notna()
    ].copy()

    # Keep genuinely new position updates while preserving
    # the original trajectory order.
    new_update = valid[
        "lastposupdate"
    ].ne(
        valid["lastposupdate"].shift(1)
    )

    updates = (
        valid.loc[new_update]
        .copy()
        .reset_index()
        .rename(
            columns={
                "index": "original_row_index"
            }
        )
    )

    lat1 = np.radians(
        updates["lat"].shift(1)
    )

    lon1 = np.radians(
        updates["lon"].shift(1)
    )

    lat2 = np.radians(
        updates["lat"]
    )

    lon2 = np.radians(
        updates["lon"]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    distance_km = (
        6371.0088 * c
    )

    largest_jump_index = (
        distance_km.idxmax()
    )

    start = max(
        0,
        largest_jump_index
        - context_updates,
    )

    end = min(
        len(updates),
        largest_jump_index
        + context_updates
        + 1,
    )

    context = updates.iloc[
        start:end
    ].copy()

    # Include distance from the previous position update.
    context["distance_from_previous_km"] = (
        distance_km.iloc[
            start:end
        ].to_numpy()
    )

    columns_to_show = [
    "original_row_index",
    "time",
    "lastposupdate",
    "lastcontact",
    "lat",
    "lon",
    "distance_from_previous_km",
    "geoaltitude",
    "baroaltitude",
    "velocity",
    "heading",
    "vertrate",
    "serials",
    "callsign",
]

    print(
        context[
            columns_to_show
        ].to_string(
            index=False
        )
    )
def main():
    suspicious_file = (
        RAW_JANUARY_DIR
        / "20260105T150813Z_a033ae_AAL2.csv"
    )

    inspect_position_updates_around_largest_jump(
        suspicious_file,
        context_updates=5,
    )


if __name__ == "__main__":
    main()