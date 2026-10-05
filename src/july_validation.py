import numpy as np
import pandas as pd

from src.config import REDUCED_JULY_DIR


def parse_timestamp(series):
    """
    Parse either Unix-second timestamps or normal datetime strings.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_fraction = (
        numeric.notna().mean()
    )

    if numeric_fraction > 0.95:
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


def inspect_july_file(file_path):
    """
    Inspect one friend-supplied July reduced trajectory.
    """

    flight = pd.read_csv(
        file_path
    )

    required_columns = {
        "time",
        "lat",
        "lon",
    }

    missing_required = (
        required_columns
        - set(flight.columns)
    )

    if missing_required:
        return {
            "file_name": file_path.name,
            "rows": len(flight),
            "valid_structure": False,
            "missing_columns": ",".join(
                sorted(missing_required)
            ),
        }

    parsed_time = parse_timestamp(
        flight["time"]
    )

    time_gap_s = (
        parsed_time
        .diff()
        .dt.total_seconds()
    )

    latitude = pd.to_numeric(
        flight["lat"],
        errors="coerce",
    )

    longitude = pd.to_numeric(
        flight["lon"],
        errors="coerce",
    )

    valid_coordinates = (
        latitude.notna()
        & longitude.notna()
        & latitude.between(
            -90,
            90,
        )
        & longitude.between(
            -180,
            180,
        )
    )

    repeated_coordinate = (
        latitude.eq(
            latitude.shift(1)
        )
        & longitude.eq(
            longitude.shift(1)
        )
    )

    positive_gaps = time_gap_s[
        time_gap_s > 0
    ]

    valid_times = (
        parsed_time.dropna()
    )

    if len(valid_times) >= 2:
        duration_h = (
            (
                valid_times.iloc[-1]
                - valid_times.iloc[0]
            ).total_seconds()
            / 3600
        )
    else:
        duration_h = np.nan

    if len(positive_gaps) > 0:
        median_gap_s = (
            positive_gaps.median()
        )

        max_gap_s = (
            positive_gaps.max()
        )

        near_10s_percent = (
            positive_gaps.between(
                8,
                12,
            ).mean()
            * 100
        )
    else:
        median_gap_s = np.nan
        max_gap_s = np.nan
        near_10s_percent = np.nan

    return {
        "file_name": (
            file_path.name
        ),

        "rows": (
            len(flight)
        ),

        "valid_structure": True,

        "missing_columns": "",

        "duration_h": (
            duration_h
        ),

        "missing_time_rows": (
            int(
                parsed_time.isna().sum()
            )
        ),

        "duplicate_time_rows": (
            int(
                parsed_time.duplicated().sum()
            )
        ),

        "nonpositive_gap_rows": (
            int(
                (
                    time_gap_s <= 0
                ).sum()
            )
        ),

        "median_gap_s": (
            median_gap_s
        ),

        "max_gap_s": (
            max_gap_s
        ),

        "near_10s_percent": (
            near_10s_percent
        ),

        "invalid_coordinate_rows": (
            int(
                (~valid_coordinates).sum()
            )
        ),

        "repeated_coordinate_rows": (
            int(
                repeated_coordinate.sum()
            )
        ),
    }


def main():
    files = sorted(
        REDUCED_JULY_DIR.glob(
            "*.csv"
        )
    )

    print(
        f"July CSV files: {len(files)}"
    )

    if not files:
        print(
            "No July CSV files found."
        )
        return

    # Inspect actual schema before dataset-wide interpretation.
    sample = pd.read_csv(
        files[0]
    )

    print(
        "\nSample file:"
    )
    print(
        files[0].name
    )

    print(
        "\nColumns:"
    )
    print(
        list(sample.columns)
    )

    print(
        "\nSample shape:"
    )
    print(
        sample.shape
    )

    records = []

    for number, file_path in enumerate(
        files,
        start=1,
    ):
        records.append(
            inspect_july_file(
                file_path
            )
        )

        if (
            number % 100 == 0
            or number == len(files)
        ):
            print(
                f"Processed {number} / {len(files)}",
                flush=True,
            )

    table = pd.DataFrame(
        records
    )

    valid = table[
        table["valid_structure"]
        == True
    ].copy()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "JULY REDUCED-DATA COMPATIBILITY SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nTotal files:"
    )
    print(
        len(table)
    )

    print(
        "\nFiles with required time/lat/lon structure:"
    )
    print(
        len(valid)
    )

    if len(valid) == 0:
        return

    print(
        "\nRows per flight:"
    )

    print(
        "minimum:",
        int(
            valid["rows"].min()
        ),
    )

    print(
        "median:",
        f"{valid['rows'].median():.1f}",
    )

    print(
        "mean:",
        f"{valid['rows'].mean():.1f}",
    )

    print(
        "maximum:",
        int(
            valid["rows"].max()
        ),
    )

    print(
        "\nMedian timestamp gap across flights:"
    )

    print(
        f"{valid['median_gap_s'].median():.3f} s"
    )

    print(
        "\nMedian percentage of gaps between 8 and 12 seconds:"
    )

    print(
        f"{valid['near_10s_percent'].median():.3f}%"
    )

    print(
        "\nFiles with duplicate timestamps:"
    )

    print(
        int(
            (
                valid[
                    "duplicate_time_rows"
                ] > 0
            ).sum()
        )
    )

    print(
        "\nFiles with non-positive timestamp gaps:"
    )

    print(
        int(
            (
                valid[
                    "nonpositive_gap_rows"
                ] > 0
            ).sum()
        )
    )

    print(
        "\nFiles with invalid/missing coordinates:"
    )

    print(
        int(
            (
                valid[
                    "invalid_coordinate_rows"
                ] > 0
            ).sum()
        )
    )

    print(
        "\nFiles containing repeated consecutive coordinates:"
    )

    print(
        int(
            (
                valid[
                    "repeated_coordinate_rows"
                ] > 0
            ).sum()
        )
    )

    print(
        "\nFiles with timestamp gap > 60 seconds:"
    )

    print(
        int(
            (
                valid[
                    "max_gap_s"
                ] > 60
            ).sum()
        )
    )

    print(
        "\n10 files with largest timestamp gaps:"
    )

    columns = [
        "file_name",
        "rows",
        "duration_h",
        "median_gap_s",
        "max_gap_s",
        "near_10s_percent",
        "repeated_coordinate_rows",
        "invalid_coordinate_rows",
    ]

    print(
        valid
        .sort_values(
            "max_gap_s",
            ascending=False,
        )
        [columns]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()