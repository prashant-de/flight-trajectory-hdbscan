"""
Analyze temporal concentration of HDBSCAN anomaly candidates.

Purpose
-------
The top-ranked anomaly candidates showed a strong concentration on
2026-07-28.

This script tests whether that concentration remains visible when
all trajectories are considered, rather than only the highest-ranked
candidates.

For each date it reports:

- total trajectory count;
- primary HDBSCAN noise count;
- primary noise percentage;
- persistent-noise count across k=10,15,20;
- persistent-noise percentage;
- median nearest-medoid distance among anomaly candidates;
- maximum nearest-medoid distance;
- number of extreme candidates.

An "extreme" candidate is defined here using the 95th percentile
of nearest-medoid distance across all 295 primary anomaly candidates.
This is descriptive, not a new anomaly definition.
"""

import numpy as np
import pandas as pd

from src.config import (
    HDBSCAN_OUTPUT_DIR,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

FINAL_LABELS_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "final"
    / "labels.csv"
)

ANOMALY_TABLE_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "anomaly_analysis"
    / "anomaly_candidates_k15_mcs30.csv"
)

OUTPUT_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "anomaly_analysis"
    / "anomaly_by_date.csv"
)


# ---------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------

def extract_date_from_trajectory_id(
    series,
):
    """
    Extract YYYYMMDD from the beginning of trajectory_id
    and convert it to a date string.
    """

    raw_date = (
        series
        .astype(str)
        .str.slice(
            0,
            8,
        )
    )

    parsed = pd.to_datetime(
        raw_date,
        format="%Y%m%d",
        errors="coerce",
    )

    if parsed.isna().any():
        bad = (
            series[
                parsed.isna()
            ]
            .head(
                10
            )
            .tolist()
        )

        raise RuntimeError(
            "Could not parse date from some trajectory IDs: "
            f"{bad}"
        )

    return (
        parsed
        .dt.strftime(
            "%Y-%m-%d"
        )
    )


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def build_daily_table(
    labels,
    anomalies,
):
    """
    Build one row per flight date.
    """

    labels = (
        labels
        .copy()
    )

    anomalies = (
        anomalies
        .copy()
    )

    labels[
        "flight_date"
    ] = extract_date_from_trajectory_id(
        labels[
            "trajectory_id"
        ]
    )

    anomalies[
        "flight_date"
    ] = extract_date_from_trajectory_id(
        anomalies[
            "trajectory_id"
        ]
    )

    # -------------------------------------------------------------
    # Total flights per day
    # -------------------------------------------------------------

    total_by_date = (
        labels
        .groupby(
            "flight_date"
        )
        .size()
        .rename(
            "total_trajectories"
        )
    )

    # -------------------------------------------------------------
    # Primary HDBSCAN noise per day
    # -------------------------------------------------------------

    noise_by_date = (
        anomalies
        .groupby(
            "flight_date"
        )
        .size()
        .rename(
            "primary_noise_count"
        )
    )

    # -------------------------------------------------------------
    # Persistent noise across all three nearby k settings
    # -------------------------------------------------------------

    persistent = (
        anomalies[
            anomalies[
                "persistent_noise_count"
            ]
            == 3
        ]
    )

    persistent_by_date = (
        persistent
        .groupby(
            "flight_date"
        )
        .size()
        .rename(
            "persistent_noise_count"
        )
    )

    # -------------------------------------------------------------
    # 2-of-3 noise
    # -------------------------------------------------------------

    noise_two = (
        anomalies[
            anomalies[
                "persistent_noise_count"
            ]
            == 2
        ]
    )

    noise_two_by_date = (
        noise_two
        .groupby(
            "flight_date"
        )
        .size()
        .rename(
            "noise_2_of_3_count"
        )
    )

    # -------------------------------------------------------------
    # Distance statistics
    # -------------------------------------------------------------

    anomaly_distance_stats = (
        anomalies
        .groupby(
            "flight_date"
        )[
            "nearest_medoid_distance_km"
        ]
        .agg(
            median_nearest_medoid_distance_km="median",
            maximum_nearest_medoid_distance_km="max",
        )
    )

    # -------------------------------------------------------------
    # Core-distance statistics
    # -------------------------------------------------------------

    core_stats = (
        anomalies
        .groupby(
            "flight_date"
        )[
            "core_distance_k15_km"
        ]
        .agg(
            median_core_distance_k15_km="median",
            maximum_core_distance_k15_km="max",
        )
    )

    # -------------------------------------------------------------
    # Extreme-distance threshold
    # -------------------------------------------------------------

    extreme_threshold = float(
        np.percentile(
            anomalies[
                "nearest_medoid_distance_km"
            ],
            95,
        )
    )

    anomalies[
        "extreme_nearest_medoid_distance"
    ] = (
        anomalies[
            "nearest_medoid_distance_km"
        ]
        >= extreme_threshold
    )

    extreme_by_date = (
        anomalies[
            anomalies[
                "extreme_nearest_medoid_distance"
            ]
        ]
        .groupby(
            "flight_date"
        )
        .size()
        .rename(
            "extreme_candidate_count"
        )
    )

    # -------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------

    daily = (
        total_by_date
        .to_frame()
        .join(
            noise_by_date,
            how="left",
        )
        .join(
            persistent_by_date,
            how="left",
        )
        .join(
            noise_two_by_date,
            how="left",
        )
        .join(
            anomaly_distance_stats,
            how="left",
        )
        .join(
            core_stats,
            how="left",
        )
        .join(
            extreme_by_date,
            how="left",
        )
        .reset_index()
    )

    count_columns = [
        "primary_noise_count",
        "persistent_noise_count",
        "noise_2_of_3_count",
        "extreme_candidate_count",
    ]

    for column in (
        count_columns
    ):
        daily[
            column
        ] = (
            daily[
                column
            ]
            .fillna(
                0
            )
            .astype(
                "int64"
            )
        )

    daily[
        "primary_noise_percent"
    ] = (
        daily[
            "primary_noise_count"
        ]
        / daily[
            "total_trajectories"
        ]
        * 100.0
    )

    daily[
        "persistent_noise_percent"
    ] = (
        daily[
            "persistent_noise_count"
        ]
        / daily[
            "total_trajectories"
        ]
        * 100.0
    )

    daily[
        "extreme_candidate_percent"
    ] = (
        daily[
            "extreme_candidate_count"
        ]
        / daily[
            "total_trajectories"
        ]
        * 100.0
    )

    daily[
        "month"
    ] = (
        daily[
            "flight_date"
        ]
        .str.slice(
            0,
            7,
        )
    )

    # Put date/month first.
    columns = [
        "flight_date",
        "month",
        "total_trajectories",
        "primary_noise_count",
        "primary_noise_percent",
        "persistent_noise_count",
        "persistent_noise_percent",
        "noise_2_of_3_count",
        "extreme_candidate_count",
        "extreme_candidate_percent",
        "median_nearest_medoid_distance_km",
        "maximum_nearest_medoid_distance_km",
        "median_core_distance_k15_km",
        "maximum_core_distance_k15_km",
    ]

    daily = (
        daily[
            columns
        ]
        .sort_values(
            "flight_date"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        daily,
        extreme_threshold,
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_top_days(
    daily,
):
    """
    Print several useful rankings.
    """

    print(
        "\n"
        + "=" * 110
    )

    print(
        "TOP DAYS BY PRIMARY HDBSCAN NOISE COUNT"
    )

    print(
        "=" * 110
    )

    columns = [
        "flight_date",
        "total_trajectories",
        "primary_noise_count",
        "primary_noise_percent",
        "persistent_noise_count",
        "persistent_noise_percent",
        "extreme_candidate_count",
        "maximum_nearest_medoid_distance_km",
    ]

    top_count = (
        daily
        .sort_values(
            [
                "primary_noise_count",
                "persistent_noise_count",
                "maximum_nearest_medoid_distance_km",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(
            15
        )
    )

    print(
        "\n",
        top_count[
            columns
        ]
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        ),
        sep="",
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "TOP DAYS BY PRIMARY NOISE RATE"
    )

    print(
        "=" * 110
    )

    top_rate = (
        daily
        .sort_values(
            [
                "primary_noise_percent",
                "primary_noise_count",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            15
        )
    )

    print(
        "\n",
        top_rate[
            columns
        ]
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        ),
        sep="",
    )


def print_event_window(
    daily,
):
    """
    Print July 27-29 specifically.
    """

    event_dates = [
        "2026-07-27",
        "2026-07-28",
        "2026-07-29",
    ]

    event = (
        daily[
            daily[
                "flight_date"
            ]
            .isin(
                event_dates
            )
        ]
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "JULY 27-29 EVENT WINDOW"
    )

    print(
        "=" * 110
    )

    print(
        "\n",
        event.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        ),
        sep="",
    )


def print_month_summary(
    daily,
):
    """
    Aggregate daily results by month.
    """

    month_summary = (
        daily
        .groupby(
            "month",
            as_index=False,
        )
        .agg(
            total_trajectories=(
                "total_trajectories",
                "sum",
            ),

            primary_noise_count=(
                "primary_noise_count",
                "sum",
            ),

            persistent_noise_count=(
                "persistent_noise_count",
                "sum",
            ),

            extreme_candidate_count=(
                "extreme_candidate_count",
                "sum",
            ),
        )
    )

    month_summary[
        "primary_noise_percent"
    ] = (
        month_summary[
            "primary_noise_count"
        ]
        / month_summary[
            "total_trajectories"
        ]
        * 100.0
    )

    month_summary[
        "persistent_noise_percent"
    ] = (
        month_summary[
            "persistent_noise_count"
        ]
        / month_summary[
            "total_trajectories"
        ]
        * 100.0
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "MONTH SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        "\n",
        month_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        ),
        sep="",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print(
        "=" * 110
    )

    print(
        "DAILY HDBSCAN ANOMALY CONCENTRATION"
    )

    print(
        "=" * 110
    )

    if not FINAL_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Missing final labels: {FINAL_LABELS_PATH}"
        )

    if not ANOMALY_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing anomaly table: {ANOMALY_TABLE_PATH}"
        )

    labels = pd.read_csv(
        FINAL_LABELS_PATH
    )

    anomalies = pd.read_csv(
        ANOMALY_TABLE_PATH
    )

    (
        daily,
        extreme_threshold,
    ) = build_daily_table(
        labels,
        anomalies,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nTotal trajectories:",
        len(
            labels
        ),
    )

    print(
        "Primary anomaly candidates:",
        len(
            anomalies
        ),
    )

    print(
        "\nExtreme candidate threshold "
        "(95th percentile nearest-medoid distance):"
    )

    print(
        f"{extreme_threshold:.3f} km"
    )

    print_top_days(
        daily
    )

    print_event_window(
        daily
    )

    print_month_summary(
        daily
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()