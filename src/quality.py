import pandas as pd

from src.config import RAW_JANUARY_DIR
from src.data_io import find_flight_csvs


# Four development flights used for early quality validation.
DEVELOPMENT_FILES = [
    RAW_JANUARY_DIR / "20260106T062826Z_a8584a_FFT2504.csv",
    RAW_JANUARY_DIR / "20260101T000338Z_ada5ee_JBU524.csv",
    RAW_JANUARY_DIR / "20260104T091537Z_ada227_JBU1724.csv",
    RAW_JANUARY_DIR / "20260105T150813Z_a033ae_AAL2.csv",
]


def inspect_timestamp_fields(file_path):
    """Inspect timestamp and freshness-related fields for one trajectory."""

    # Read only the columns needed for this check.
    columns = [
        "time",
        "lastposupdate",
        "lastcontact",
        "position_age_seconds",
        "contact_age_seconds",
    ]

    flight = pd.read_csv(file_path, usecols=columns)

    # Convert the ISO timestamp strings into UTC datetimes.
    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    print("File:", file_path.name)
    print("Rows:", len(flight))

    print("\nUnparsed time values:")
    print(parsed_time.isna().sum())

    print("\nFirst timestamp:")
    print(parsed_time.iloc[0])

    print("\nLast timestamp:")
    print(parsed_time.iloc[-1])

    print("\nFirst 3 lastposupdate values:")
    print(flight["lastposupdate"].head(3).tolist())

    print("\nFirst 3 lastcontact values:")
    print(flight["lastcontact"].head(3).tolist())

     # Show the datetime resolution used by pandas.
    print("\nParsed time resolution:")
    print(parsed_time.dt.unit)

     # Explicitly convert the datetime values to second resolution
     # before obtaining their Unix integer representation.
    time_unix_seconds = parsed_time.dt.as_unit("s").astype("int64")

    # Recalculate the two age fields using compatible units.
    recalculated_position_age = (
        time_unix_seconds - flight["lastposupdate"]
    )

    recalculated_contact_age = (
        time_unix_seconds - flight["lastcontact"]
    )

    comparison = pd.DataFrame(
        {
            "time": flight["time"],
            "supplied_position_age": flight["position_age_seconds"],
            "recalculated_position_age": recalculated_position_age,
            "supplied_contact_age": flight["contact_age_seconds"],
            "recalculated_contact_age": recalculated_contact_age,
        }
    )


    print("\nFirst 5 age comparisons:")
    print(comparison.head().to_string(index=False))

        # Check whether any recalculated ages are negative.
    # Negative ages could indicate timestamp ordering or unit problems.
    print("\nNegative recalculated position ages:")
    print((recalculated_position_age < 0).sum())

    print("\nNegative recalculated contact ages:")
    print((recalculated_contact_age < 0).sum())

    # Basic freshness summary.
    print("\nMaximum recalculated position age (seconds):")
    print(recalculated_position_age.max())

    print("\nMaximum recalculated contact age (seconds):")
    print(recalculated_contact_age.max())

    print("\nMedian recalculated position age (seconds):")
    print(recalculated_position_age.median())

    print("\nMedian recalculated contact age (seconds):")
    print(recalculated_contact_age.median())

    # These are reporting thresholds only.
    # We are NOT using them to remove observations.
    for threshold in [10, 30, 60]:
        stale_position_rows = (recalculated_position_age > threshold).sum()

        print(
            f"\nRows with recalculated position age > {threshold} seconds:"
        )
        print(stale_position_rows)

def summarize_timestamp_freshness(file_path):
    """Return timestamp/freshness quality metrics for one flight."""

    columns = [
    "time",
    "lastposupdate",
    "lastcontact",
    "position_age_seconds",
    "contact_age_seconds",
]

    flight = pd.read_csv(file_path, usecols=columns)

    # Parse trajectory row timestamps as UTC.
    parsed_time = pd.to_datetime(
        flight["time"],
        utc=True,
        errors="coerce",
    )

    # Convert explicitly to Unix seconds.
    time_unix_seconds = parsed_time.dt.as_unit("s").astype("int64")

    # Age tells us how old the latest position/contact information
    # was at each trajectory row timestamp.
    position_age = time_unix_seconds - flight["lastposupdate"]
    contact_age = time_unix_seconds - flight["lastcontact"]

    return {
        "file_name": file_path.name,
        "rows": len(flight),
        "unparsed_times": parsed_time.isna().sum(),

        "missing_lastposupdate": flight["lastposupdate"].isna().sum(),
        "missing_lastposupdate_fraction": (
            flight["lastposupdate"].isna().mean()
        ),
        "missing_lastcontact": flight["lastcontact"].isna().sum(),

        "valid_position_age_rows": position_age.notna().sum(),

        "negative_position_age": (position_age < 0).sum(),
        "negative_contact_age": (contact_age < 0).sum(),

        "median_position_age_s": position_age.median(),
        "max_position_age_s": position_age.max(),

        "median_contact_age_s": contact_age.median(),
        "max_contact_age_s": contact_age.max(),

        "position_age_gt_10s": (position_age > 10).sum(),
        "position_age_gt_30s": (position_age > 30).sum(),
        "position_age_gt_60s": (position_age > 60).sum(),

                # Check the supplied age columns separately.
        # We keep them for traceability, but do not trust them automatically.
        "negative_supplied_position_age": (
            flight["position_age_seconds"] < 0
        ).sum(),

        "negative_supplied_contact_age": (
            flight["contact_age_seconds"] < 0
        ).sum(),

        "missing_supplied_position_age": (
            flight["position_age_seconds"].isna().sum()
        ),

        "missing_supplied_contact_age": (
            flight["contact_age_seconds"].isna().sum()
        ),
    }

if __name__ == "__main__":
    flight_files = find_flight_csvs(RAW_JANUARY_DIR)

    summaries = []

    for file_path in flight_files:
        summaries.append(
            summarize_timestamp_freshness(file_path)
        )

    summary_table = pd.DataFrame(summaries)

    total_rows = summary_table["rows"].sum()
    total_missing_position_updates = (
        summary_table["missing_lastposupdate"].sum()
    )

    print("January flights checked:", len(summary_table))
    print("Total January rows:", total_rows)

    print("\nTotal rows with missing lastposupdate:")
    print(total_missing_position_updates)

    print("\nOverall missing-lastposupdate fraction:")
    print(total_missing_position_updates / total_rows)

    print("\nFlights containing negative supplied position ages:")
    print(
        (
            summary_table["negative_supplied_position_age"] > 0
        ).sum()
    )

    print("\nFlights containing negative supplied contact ages:")
    print(
        (
            summary_table["negative_supplied_contact_age"] > 0
        ).sum()
    )

    print("\nTotal negative supplied position-age rows:")
    print(
        summary_table["negative_supplied_position_age"].sum()
    )

    print("\nTotal negative supplied contact-age rows:")
    print(
        summary_table["negative_supplied_contact_age"].sum()
    )

    print("\nFlights with missing supplied position ages:")
    print(
        (
            summary_table["missing_supplied_position_age"] > 0
        ).sum()
    )

    print("\nFlights with missing supplied contact ages:")
    print(
        (
            summary_table["missing_supplied_contact_age"] > 0
        ).sum()
    )