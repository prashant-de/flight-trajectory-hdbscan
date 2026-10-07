"""
Visualize the strongest HDBSCAN anomaly candidates.

The figure overlays the highest-ranked anomaly candidates from
Module 5.2 with the four primary cluster medoid trajectories.

The purpose is qualitative validation of the anomaly ranking:
high-ranked candidates should show substantial geometric deviation
from the recurring route families if the ranking is meaningful.
"""

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

OUTPUT_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "anomaly_analysis"
    / "top_anomaly_candidates.png"
)


def trajectory_path(
    month,
    file_name,
):
    """
    Return processed trajectory path from month label.
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
            f"Trajectory file not found: {path}"
        )

    trajectory = pd.read_csv(
        path
    )

    required_columns = {
        "lat",
        "lon",
    }

    missing = (
        required_columns
        - set(
            trajectory.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{file_name} missing columns: "
            f"{sorted(missing)}"
        )

    return trajectory


def main():

    print(
        "=" * 90
    )

    print(
        "VISUALIZE TOP HDBSCAN ANOMALY CANDIDATES"
    )

    print(
        "=" * 90
    )

    if not ANOMALY_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing anomaly table: {ANOMALY_TABLE_PATH}"
        )

    if not MEDOID_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing medoid table: {MEDOID_TABLE_PATH}"
        )

    candidates = pd.read_csv(
        ANOMALY_TABLE_PATH
    )

    medoids = pd.read_csv(
        MEDOID_TABLE_PATH
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
    # Figure
    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )
    # Plot representative routes
    for row in (
        medoids.itertuples(
            index=False
        )
    ):

        trajectory = load_trajectory(
            row.medoid_month,
            row.medoid_file_name,
        )

        ax.plot(
            trajectory[
                "lon"
            ],
            trajectory[
                "lat"
            ],
            linewidth=2.5,
            alpha=0.95,
            label=(
                f"Cluster {row.cluster_label} medoid"
            ),
        )
    # Plot strongest anomaly candidates
    for row in (
        candidates.itertuples(
            index=False
        )
    ):

        trajectory = load_trajectory(
            row.month,
            row.file_name,
        )

        ax.plot(
            trajectory[
                "lon"
            ],
            trajectory[
                "lat"
            ],
            linewidth=1.25,
            alpha=0.55,
            linestyle="--",
        )

    ax.set_title(
        f"Top {TOP_N} Persistent HDBSCAN Anomaly Candidates\n"
        "Compared with Primary Cluster Medoids"
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )
    # Report plotted candidates
    print(
        f"\nTop {TOP_N} candidates plotted:"
    )

    display_columns = [
        "anomaly_candidate_rank",
        "month",
        "trajectory_id",
        "persistent_noise_count",
        "nearest_medoid_cluster",
        "nearest_medoid_distance_km",
        "core_distance_k15_km",
    ]

    print(
        "\n",
        candidates[
            display_columns
        ]
        .to_string(
            index=False
        ),
        sep="",
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()