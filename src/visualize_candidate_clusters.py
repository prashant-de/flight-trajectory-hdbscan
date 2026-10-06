from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    HDBSCAN_OUTPUT_DIR,
    PROCESSED_JANUARY_DIR,
    PROCESSED_JULY_DIR,
)


CANDIDATE_CONFIGURATIONS = [
    (10, 30),
    (15, 30),
    (20, 30),
]


def load_labels(
    min_samples,
    min_cluster_size,
):
    """
    Load final trajectory labels for one HDBSCAN configuration.
    """

    path = (
        HDBSCAN_OUTPUT_DIR
        / (
            f"labels_"
            f"k{min_samples}_"
            f"mcs{min_cluster_size}.csv"
        )
    )

    return pd.read_csv(
        path
    )


def get_trajectory_path(
    row,
):
    """
    Recover processed trajectory CSV path from month metadata.
    """

    if row["month"] == "2026-01":
        directory = (
            PROCESSED_JANUARY_DIR
        )

    elif row["month"] == "2026-07":
        directory = (
            PROCESSED_JULY_DIR
        )

    else:
        raise ValueError(
            f"Unknown month: {row['month']}"
        )

    return (
        directory
        / row["file_name"]
    )


def plot_configuration(
    labels,
    min_samples,
    min_cluster_size,
):
    """
    Plot trajectories with one consistent color per cluster.

    Noise is shown in light gray so that the selected
    route groups are visually distinguishable.
    """

    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )

    cluster_labels = sorted(
        label
        for label in (
            labels[
                "cluster_label"
            ]
            .unique()
        )
        if label >= 0
    )

    # -------------------------------------------------------------
    # Noise
    # -------------------------------------------------------------

    noise_rows = (
        labels[
            labels[
                "cluster_label"
            ]
            == -1
        ]
    )

    first_noise = True

    for row in (
        noise_rows
        .itertuples(
            index=False
        )
    ):
        trajectory_path = (
            get_trajectory_path(
                row._asdict()
            )
        )

        trajectory = pd.read_csv(
            trajectory_path,
            usecols=[
                "lat",
                "lon",
            ],
        )

        noise_label = None

        if first_noise:
            noise_label = (
                f"Noise "
                f"(n={len(noise_rows)})"
            )

            first_noise = False

        ax.plot(
            trajectory["lon"],
            trajectory["lat"],
            color="lightgray",
            linewidth=0.35,
            alpha=0.20,
            label=noise_label,
            zorder=1,
        )

    # -------------------------------------------------------------
    # Selected clusters
    # -------------------------------------------------------------

    for cluster_label in (
        cluster_labels
    ):

        cluster_rows = (
            labels[
                labels[
                    "cluster_label"
                ]
                == cluster_label
            ]
        )

        # One fixed color for every trajectory in this cluster.
        cluster_color = (
            f"C{cluster_label % 10}"
        )

        first = True

        for row in (
            cluster_rows
            .itertuples(
                index=False
            )
        ):
            trajectory_path = (
                get_trajectory_path(
                    row._asdict()
                )
            )

            trajectory = pd.read_csv(
                trajectory_path,
                usecols=[
                    "lat",
                    "lon",
                ],
            )

            plot_label = None

            if first:
                plot_label = (
                    f"Cluster "
                    f"{cluster_label} "
                    f"(n={len(cluster_rows)})"
                )

                first = False

            ax.plot(
                trajectory["lon"],
                trajectory["lat"],
                color=cluster_color,
                linewidth=0.55,
                alpha=0.22,
                label=plot_label,
                zorder=2,
            )

    ax.set_title(
        (
            "LAX → JFK HDBSCAN Trajectory Clusters\n"
            f"min_samples={min_samples}, "
            f"min_cluster_size={min_cluster_size}"
        )
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    output_path = (
        HDBSCAN_OUTPUT_DIR
        / (
            "trajectory_clusters_"
            f"k{min_samples}_"
            f"mcs{min_cluster_size}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def print_cluster_summary(
    labels,
    min_samples,
    min_cluster_size,
):
    """
    Print cluster and month composition.
    """

    summary = (
        labels
        .groupby(
            [
                "cluster_label",
                "month",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    summary[
        "total"
    ] = (
        summary.sum(
            axis=1
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"k={min_samples}, "
        f"mcs={min_cluster_size}"
    )

    print(
        "=" * 75
    )

    print(
        summary.to_string()
    )


def main():

    HDBSCAN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        min_samples,
        min_cluster_size,
    ) in CANDIDATE_CONFIGURATIONS:

        labels = load_labels(
            min_samples,
            min_cluster_size,
        )

        print_cluster_summary(
            labels,
            min_samples,
            min_cluster_size,
        )

        output_path = (
            plot_configuration(
                labels,
                min_samples,
                min_cluster_size,
            )
        )

        print(
            "\nSaved:",
            output_path,
        )


if __name__ == "__main__":
    main()