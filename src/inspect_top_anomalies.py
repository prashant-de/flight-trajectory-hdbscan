"""
Inspect the highest-ranked HDBSCAN anomaly candidates individually.

Purpose
-------
The top anomaly candidates show large geometric deviations from the
recurring route medoids.

Before interpreting them as operational anomalies, this script checks
whether their processed trajectories also show signs of possible data
quality problems.

For each candidate it reports:

- observation count;
- trajectory duration;
- maximum timestamp gap;
- number of gaps > 20 seconds;
- interpolated fraction;
- total geographic path length;
- maximum consecutive geographic step;
- maximum implied step speed.

It also creates one diagnostic figure per candidate so that individual
trajectory geometry can be inspected without overlap from the other
anomaly candidates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    HDBSCAN_OUTPUT_DIR,
    PROCESSED_JANUARY_DIR,
    PROCESSED_JULY_DIR,
)


TOP_N = 12

ANOMALY_TABLE_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "anomaly_analysis"
    / "anomaly_candidates_k15_mcs30.csv"
)

MEDOID_TABLE_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "representative_routes_k15_mcs30.csv"
)

OUTPUT_DIR = (
    HDBSCAN_OUTPUT_DIR
    / "anomaly_analysis"
    / "top_candidate_diagnostics"
)

SUMMARY_OUTPUT_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "anomaly_analysis"
    / "top_anomaly_quality_diagnostics.csv"
)


# ---------------------------------------------------------------------
# Geographic distance
# ---------------------------------------------------------------------

def haversine_steps_km(
    latitudes,
    longitudes,
):
    """
    Calculate Haversine distance between consecutive points.
    """

    latitudes = np.asarray(
        latitudes,
        dtype="float64",
    )

    longitudes = np.asarray(
        longitudes,
        dtype="float64",
    )

    if len(latitudes) < 2:
        return np.array(
            [],
            dtype="float64",
        )

    earth_radius_km = 6371.0088

    lat1 = np.radians(
        latitudes[:-1]
    )

    lat2 = np.radians(
        latitudes[1:]
    )

    lon1 = np.radians(
        longitudes[:-1]
    )

    lon2 = np.radians(
        longitudes[1:]
    )

    dlat = (
        lat2
        - lat1
    )

    dlon = (
        lon2
        - lon1
    )

    a = (
        np.sin(
            dlat / 2.0
        ) ** 2
        + np.cos(
            lat1
        )
        * np.cos(
            lat2
        )
        * np.sin(
            dlon / 2.0
        ) ** 2
    )

    c = (
        2.0
        * np.arctan2(
            np.sqrt(
                a
            ),
            np.sqrt(
                1.0 - a
            ),
        )
    )

    return (
        earth_radius_km
        * c
    )


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def trajectory_path(
    month,
    file_name,
):
    """
    Resolve processed trajectory path.
    """

    if month == "2026-01":
        return (
            PROCESSED_JANUARY_DIR
            / file_name
        )

    if month == "2026-07":
        return (
            PROCESSED_JULY_DIR
            / file_name
        )

    raise ValueError(
        f"Unknown month: {month}"
    )


def load_trajectory(
    month,
    file_name,
):
    """
    Load one processed trajectory.
    """

    path = trajectory_path(
        month,
        file_name,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing trajectory: {path}"
        )

    trajectory = pd.read_csv(
        path
    )

    required = {
        "time_utc",
        "lat",
        "lon",
        "interpolated",
    }

    missing = (
        required
        - set(
            trajectory.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{file_name} missing columns: "
            f"{sorted(missing)}"
        )

    trajectory[
        "time_utc"
    ] = pd.to_datetime(
        trajectory[
            "time_utc"
        ],
        utc=True,
        errors="coerce",
    )

    return trajectory


# ---------------------------------------------------------------------
# Quality statistics
# ---------------------------------------------------------------------

def interpolated_fraction(
    series,
):
    """
    Robustly interpret the interpolated flag.
    """

    if pd.api.types.is_bool_dtype(
        series
    ):
        return float(
            series.mean()
        )

    values = (
        series
        .astype(
            str
        )
        .str.lower()
        .str.strip()
    )

    flags = values.isin(
        [
            "true",
            "1",
            "yes",
        ]
    )

    return float(
        flags.mean()
    )


def trajectory_diagnostics(
    trajectory,
):
    """
    Calculate trajectory-level geometric and temporal diagnostics.
    """

    valid_time = (
        trajectory[
            "time_utc"
        ]
        .notna()
    )

    clean = (
        trajectory.loc[
            valid_time
        ]
        .copy()
        .sort_values(
            "time_utc"
        )
        .reset_index(
            drop=True
        )
    )

    if len(clean) < 2:
        raise RuntimeError(
            "Trajectory has fewer than two valid observations."
        )

    times = (
        clean[
            "time_utc"
        ]
    )

    time_differences = (
        times
        .diff()
        .dt.total_seconds()
        .to_numpy(
            dtype="float64"
        )[1:]
    )

    step_distances = (
        haversine_steps_km(
            clean[
                "lat"
            ],
            clean[
                "lon"
            ],
        )
    )

    valid_speed = (
        time_differences > 0
    )

    implied_speed_kmh = (
        step_distances[
            valid_speed
        ]
        / time_differences[
            valid_speed
        ]
        * 3600.0
    )

    duration_seconds = (
        (
            times.iloc[-1]
            - times.iloc[0]
        )
        .total_seconds()
    )

    return {
        "observation_count": (
            int(
                len(
                    trajectory
                )
            )
        ),

        "valid_time_count": (
            int(
                len(
                    clean
                )
            )
        ),

        "duration_hours": (
            float(
                duration_seconds
                / 3600.0
            )
        ),

        "maximum_time_gap_sec": (
            float(
                np.max(
                    time_differences
                )
            )
        ),

        "gaps_over_20_sec": (
            int(
                (
                    time_differences
                    > 20.0
                ).sum()
            )
        ),

        "gaps_over_60_sec": (
            int(
                (
                    time_differences
                    > 60.0
                ).sum()
            )
        ),

        "interpolated_percent": (
            float(
                interpolated_fraction(
                    trajectory[
                        "interpolated"
                    ]
                )
                * 100.0
            )
        ),

        "total_path_length_km": (
            float(
                step_distances.sum()
            )
        ),

        "median_step_distance_km": (
            float(
                np.median(
                    step_distances
                )
            )
        ),

        "p95_step_distance_km": (
            float(
                np.percentile(
                    step_distances,
                    95,
                )
            )
        ),

        "maximum_step_distance_km": (
            float(
                np.max(
                    step_distances
                )
            )
        ),

        "maximum_implied_speed_kmh": (
            float(
                np.max(
                    implied_speed_kmh
                )
            )
        ),
    }


# ---------------------------------------------------------------------
# Individual plotting
# ---------------------------------------------------------------------

def plot_candidate(
    candidate,
    trajectory,
    medoids,
):
    """
    Plot one anomaly candidate against all four route medoids.
    """

    figure, axis = plt.subplots(
        figsize=(
            11,
            7,
        )
    )

    for medoid in (
        medoids.itertuples(
            index=False
        )
    ):

        medoid_trajectory = (
            load_trajectory(
                medoid.medoid_month,
                medoid.medoid_file_name,
            )
        )

        is_nearest = (
            int(
                medoid.cluster_label
            )
            == int(
                candidate.nearest_medoid_cluster
            )
        )

        if is_nearest:
            linewidth = 2.5
            alpha = 0.95

        else:
            linewidth = 1.2
            alpha = 0.35

        axis.plot(
            medoid_trajectory[
                "lon"
            ],
            medoid_trajectory[
                "lat"
            ],
            linewidth=linewidth,
            alpha=alpha,
            label=(
                f"Cluster {medoid.cluster_label} medoid"
            ),
        )

    axis.plot(
        trajectory[
            "lon"
        ],
        trajectory[
            "lat"
        ],
        linestyle="--",
        linewidth=2.0,
        color="black",
        label=(
            f"Candidate rank "
            f"{candidate.anomaly_candidate_rank}"
        ),
    )

    axis.set_title(
        (
            f"Anomaly Candidate "
            f"#{candidate.anomaly_candidate_rank}\n"
            f"{candidate.trajectory_id} | "
            f"nearest medoid distance "
            f"{candidate.nearest_medoid_distance_km:.1f} km"
        )
    )

    axis.set_xlabel(
        "Longitude"
    )

    axis.set_ylabel(
        "Latitude"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        loc="best"
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / (
            f"rank_"
            f"{int(candidate.anomaly_candidate_rank):02d}_"
            f"{candidate.trajectory_id}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print(
        "=" * 110
    )

    print(
        "TOP HDBSCAN ANOMALY QUALITY DIAGNOSTICS"
    )

    print(
        "=" * 110
    )

    candidates = pd.read_csv(
        ANOMALY_TABLE_PATH
    )

    candidates = (
        candidates
        .sort_values(
            "anomaly_candidate_rank"
        )
        .head(
            TOP_N
        )
        .reset_index(
            drop=True
        )
    )

    medoids = pd.read_csv(
        MEDOID_TABLE_PATH
    )

    medoids = (
        medoids
        .sort_values(
            "cluster_label"
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_rows = []

    for candidate in (
        candidates.itertuples(
            index=False
        )
    ):

        trajectory = (
            load_trajectory(
                candidate.month,
                candidate.file_name,
            )
        )

        statistics = (
            trajectory_diagnostics(
                trajectory
            )
        )

        plot_candidate(
            candidate,
            trajectory,
            medoids,
        )

        diagnostic_rows.append(
            {
                "anomaly_candidate_rank": (
                    int(
                        candidate.anomaly_candidate_rank
                    )
                ),

                "matrix_index": (
                    int(
                        candidate.matrix_index
                    )
                ),

                "month": (
                    candidate.month
                ),

                "trajectory_id": (
                    candidate.trajectory_id
                ),

                "persistent_noise_count": (
                    int(
                        candidate.persistent_noise_count
                    )
                ),

                "nearest_medoid_cluster": (
                    int(
                        candidate.nearest_medoid_cluster
                    )
                ),

                "nearest_medoid_distance_km": (
                    float(
                        candidate.nearest_medoid_distance_km
                    )
                ),

                "core_distance_k15_km": (
                    float(
                        candidate.core_distance_k15_km
                    )
                ),

                **statistics,
            }
        )

    diagnostics = pd.DataFrame(
        diagnostic_rows
    )

    diagnostics.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    display_columns = [
        "anomaly_candidate_rank",
        "month",
        "trajectory_id",
        "duration_hours",
        "maximum_time_gap_sec",
        "gaps_over_60_sec",
        "interpolated_percent",
        "total_path_length_km",
        "maximum_step_distance_km",
        "maximum_implied_speed_kmh",
    ]

    print(
        "\n",
        diagnostics[
            display_columns
        ]
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        ),
        sep="",
    )

    print(
        "\nDate concentration:"
    )

    date_counts = (
        diagnostics[
            "trajectory_id"
        ]
        .str.slice(
            0,
            8,
        )
        .value_counts()
        .sort_index()
    )

    print(
        date_counts.to_string()
    )

    print(
        "\nSaved summary:"
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        "\nSaved individual figures in:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()