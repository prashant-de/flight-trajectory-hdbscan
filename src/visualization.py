"""
Generate the final report-ready visualization set for the frozen
LAX -> JFK HDBSCAN clustering.

Final figures
-------------
1. Geographic map of all final HDBSCAN route clusters.
2. Geographic map of the four representative cluster medoids.
3. January/July composition of clusters and noise.
4. Silhouette-score distributions by cluster.
5. Pairwise Hausdorff distances between representative medoids.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.lines import Line2D

from src.config import (
    DISTANCE_OUTPUT_DIR,
    HDBSCAN_OUTPUT_DIR,
    PROJECT_ROOT,
)

# Paths
REPRESENTATION_PATH = (
    DISTANCE_OUTPUT_DIR
    / "trajectory_representations_100.npy"
)

LABELS_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "final"
    / "labels.csv"
)

MEDOIDS_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "representative_routes_k15_mcs30.csv"
)

SILHOUETTE_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "final_evaluation"
    / "silhouette_samples.csv"
)

MEDOID_DISTANCE_PATH = (
    HDBSCAN_OUTPUT_DIR
    / "final_evaluation"
    / "intercluster_medoid_distances.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "final"
)

# Fixed visual identity
CLUSTER_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
}

NOISE_COLOR = "0.45"

LAX_COLOR = "blue"
JFK_COLOR = "green"

LAX_LAT = 33.9416
LAX_LON = -118.4085

JFK_LAT = 40.6413
JFK_LON = -73.7781

# Shared helpers
def save_figure(
    figure,
    stem,
):
    """
    Save one figure as both PNG and PDF.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURE_DIR
        / f"{stem}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{stem}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return (
        png_path,
        pdf_path,
    )


def load_primary_data():
    """
    Load and validate the frozen primary clustering artifacts.
    """

    representations = np.load(
        REPRESENTATION_PATH
    )

    labels = pd.read_csv(
        LABELS_PATH
    )

    medoids = pd.read_csv(
        MEDOIDS_PATH
    )

    labels = (
        labels
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        representations.shape[0]
        != len(labels)
    ):
        raise RuntimeError(
            "Representation and label counts differ."
        )

    expected_indices = np.arange(
        len(labels)
    )

    actual_indices = (
        labels[
            "matrix_index"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    if not np.array_equal(
        expected_indices,
        actual_indices,
    ):
        raise RuntimeError(
            "Trajectory ordering is inconsistent."
        )

    for _, row in medoids.iterrows():

        medoid_index = int(
            row[
                "medoid_matrix_index"
            ]
        )

        expected_id = (
            labels.loc[
                medoid_index,
                "trajectory_id",
            ]
        )

        if (
            expected_id
            != row[
                "medoid_trajectory_id"
            ]
        ):
            raise RuntimeError(
                "Representative-route alignment failed "
                f"for cluster {row['cluster_label']}."
            )

    return (
        representations,
        labels,
        medoids,
    )

# Geographic helpers
def add_geographic_background(
    axis,
):
    """
    Add a clean technical map background.

    Coastlines and national boundaries are emphasized while
    state boundaries remain lighter.
    """

    axis.set_extent(
        [
            -125,
            -66,
            24,
            49,
        ],
        crs=ccrs.PlateCarree(),
    )

    axis.set_facecolor(
        "white"
    )

    axis.add_feature(
        cfeature.LAND.with_scale(
            "50m"
        ),
        facecolor="#f7f7f7",
        edgecolor="none",
        zorder=0,
    )

    axis.add_feature(
        cfeature.OCEAN.with_scale(
            "50m"
        ),
        facecolor="white",
        edgecolor="none",
        zorder=0,
    )

    axis.add_feature(
        cfeature.COASTLINE.with_scale(
            "50m"
        ),
        linewidth=1.0,
        edgecolor="0.25",
        zorder=1,
    )

    axis.add_feature(
        cfeature.BORDERS.with_scale(
            "50m"
        ),
        linewidth=0.9,
        edgecolor="0.25",
        zorder=1,
    )

    axis.add_feature(
        cfeature.STATES.with_scale(
            "50m"
        ),
        linewidth=0.5,
        edgecolor="0.65",
        zorder=1,
    )

    gridlines = axis.gridlines(
        draw_labels=True,
        linewidth=0.45,
        color="gray",
        alpha=0.45,
        linestyle=":",
        x_inline=False,
        y_inline=False,
    )

    gridlines.top_labels = False
    gridlines.right_labels = False

    gridlines.xlabel_style = {
        "size": 10
    }

    gridlines.ylabel_style = {
        "size": 10
    }


def add_airports(
    axis,
):
    """
    Add LAX and JFK markers.
    """

    axis.scatter(
        LAX_LON,
        LAX_LAT,
        s=55,
        color=LAX_COLOR,
        edgecolor="white",
        linewidth=0.7,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    axis.scatter(
        JFK_LON,
        JFK_LAT,
        s=55,
        color=JFK_COLOR,
        edgecolor="white",
        linewidth=0.7,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    axis.text(
        LAX_LON + 0.7,
        LAX_LAT - 0.8,
        "LAX",
        fontsize=11,
        fontweight="bold",
        color=LAX_COLOR,
        transform=ccrs.PlateCarree(),
        zorder=11,
    )

    axis.text(
        JFK_LON + 0.7,
        JFK_LAT + 0.4,
        "JFK",
        fontsize=11,
        fontweight="bold",
        color=JFK_COLOR,
        transform=ccrs.PlateCarree(),
        zorder=11,
    )

# Figure 1 — All final HDBSCAN clusters
def plot_final_route_clusters_map(
    representations,
    labels,
):
    """
    Plot every trajectory using its final HDBSCAN label.

    IMPORTANT:
    Representative medoids are deliberately NOT overlaid here.
    This figure shows the clustering result itself.
    """

    figure = plt.figure(
        figsize=(
            14,
            7.8,
        )
    )

    axis = plt.axes(
        projection=ccrs.PlateCarree()
    )

    add_geographic_background(
        axis
    )

    cluster_labels = (
        labels[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )
    # HDBSCAN noise
    noise_indices = np.flatnonzero(
        cluster_labels
        == -1
    )

    for index in noise_indices:

        route = (
            representations[
                index
            ]
        )

        axis.plot(
            route[:, 1],
            route[:, 0],
            color=NOISE_COLOR,
            linewidth=0.35,
            alpha=0.08,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
    # Four route clusters
    for cluster_id in [
        0,
        1,
        2,
        3,
    ]:

        member_indices = np.flatnonzero(
            cluster_labels
            == cluster_id
        )

        for index in member_indices:

            route = (
                representations[
                    index
                ]
            )

            axis.plot(
                route[:, 1],
                route[:, 0],
                color=CLUSTER_COLORS[
                    cluster_id
                ],
                linewidth=0.50,
                alpha=0.16,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

    add_airports(
        axis
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=NOISE_COLOR,
            linewidth=1.2,
            label=(
                "HDBSCAN noise "
                f"(n={len(noise_indices)})"
            ),
        )
    ]

    for cluster_id in [
        0,
        1,
        2,
        3,
    ]:

        count = int(
            (
                cluster_labels
                == cluster_id
            ).sum()
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=CLUSTER_COLORS[
                    cluster_id
                ],
                linewidth=1.6,
                label=(
                    f"Cluster {cluster_id} "
                    f"(n={count})"
                ),
            )
        )

    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=LAX_COLOR,
                markeredgecolor="white",
                markersize=7,
                label="LAX origin",
            ),

            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=JFK_COLOR,
                markeredgecolor="white",
                markersize=7,
                label="JFK destination",
            ),
        ]
    )

    axis.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=9,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="0.6",
    )

    axis.set_title(
        "Final HDBSCAN Route Clusters: LAX → JFK",
        fontsize=16,
        pad=16,
    )

    figure.tight_layout()

    return save_figure(
        figure,
        "final_route_clusters_map",
    )

# Figure 2 — Four representative routes
def plot_representative_routes_map(
    representations,
    medoids,
):
    """
    Plot only the four observed cluster medoids.

    This figure represents the recurring route families without
    displaying every individual trajectory.
    """

    figure = plt.figure(
        figsize=(
            14,
            7.8,
        )
    )

    axis = plt.axes(
        projection=ccrs.PlateCarree()
    )

    add_geographic_background(
        axis
    )

    for _, row in (
        medoids
        .sort_values(
            "cluster_label"
        )
        .iterrows()
    ):

        cluster_id = int(
            row[
                "cluster_label"
            ]
        )

        cluster_size = int(
            row[
                "cluster_size"
            ]
        )

        medoid_index = int(
            row[
                "medoid_matrix_index"
            ]
        )

        route = (
            representations[
                medoid_index
            ]
        )

        axis.plot(
            route[:, 1],
            route[:, 0],
            color=CLUSTER_COLORS[
                cluster_id
            ],
            linewidth=1.4,
            alpha=0.95,
            transform=ccrs.PlateCarree(),
            label=(
                f"Cluster {cluster_id} "
                f"(n={cluster_size})"
            ),
            zorder=4,
        )

    add_airports(
        axis
    )

    legend_handles = []

    for _, row in (
        medoids
        .sort_values(
            "cluster_label"
        )
        .iterrows()
    ):

        cluster_id = int(
            row[
                "cluster_label"
            ]
        )

        cluster_size = int(
            row[
                "cluster_size"
            ]
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=CLUSTER_COLORS[
                    cluster_id
                ],
                linewidth=1.6,
                label=(
                    f"Cluster {cluster_id} "
                    f"(n={cluster_size})"
                ),
            )
        )

    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=LAX_COLOR,
                markeredgecolor="white",
                markersize=7,
                label="LAX origin",
            ),

            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=JFK_COLOR,
                markeredgecolor="white",
                markersize=7,
                label="JFK destination",
            ),
        ]
    )

    axis.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=10,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="0.6",
    )

    axis.set_title(
        "Representative Routes of the Four HDBSCAN Clusters: LAX → JFK",
        fontsize=16,
        pad=16,
    )

    figure.tight_layout()

    return save_figure(
        figure,
        "final_representative_routes_map",
    )

# Figure 3 — Month composition
def plot_month_composition(
    labels,
):
    """
    Show January/July membership counts.

    This plot is descriptive and is not interpreted as evidence
    of a seasonal causal effect.
    """

    categories = [
        -1,
        0,
        1,
        2,
        3,
    ]

    names = [
        "Noise",
        "Cluster 0",
        "Cluster 1",
        "Cluster 2",
        "Cluster 3",
    ]

    january = []

    july = []

    for cluster_id in categories:

        subset = (
            labels[
                labels[
                    "cluster_label"
                ]
                == cluster_id
            ]
        )

        january.append(
            int(
                (
                    subset[
                        "month"
                    ]
                    == "2026-01"
                ).sum()
            )
        )

        july.append(
            int(
                (
                    subset[
                        "month"
                    ]
                    == "2026-07"
                ).sum()
            )
        )

    january = np.asarray(
        january
    )

    july = np.asarray(
        july
    )

    totals = (
        january
        + july
    )

    positions = np.arange(
        len(categories)
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.bar(
        positions,
        january,
        label="January 2026",
    )

    axis.bar(
        positions,
        july,
        bottom=january,
        label="July 2026",
    )

    for position, total in zip(
        positions,
        totals,
    ):

        axis.text(
            position,
            total + 8,
            str(
                int(total)
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_xticks(
        positions,
        names,
    )

    axis.set_ylabel(
        "Number of trajectories"
    )

    axis.set_title(
        "Month Composition of Final HDBSCAN Groups"
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.20,
    )

    axis.text(
        0.5,
        -0.13,
        (
            "Month composition is descriptive; "
            "it is not interpreted as a seasonal causal effect."
        ),
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
    )

    figure.tight_layout()

    return save_figure(
        figure,
        "final_month_composition",
    )

# Figure 4 — Silhouette distributions
def plot_silhouette_distribution():
    """
    Plot silhouette-score distributions for the four selected
    route clusters.
    """

    silhouette = pd.read_csv(
        SILHOUETTE_PATH
    )

    cluster_ids = sorted(
        silhouette[
            "cluster_label"
        ]
        .unique()
        .tolist()
    )

    distributions = [
        (
            silhouette.loc[
                silhouette[
                    "cluster_label"
                ]
                == cluster_id,
                "silhouette_score",
            ]
            .to_numpy(
                dtype="float64"
            )
        )
        for cluster_id
        in cluster_ids
    ]

    figure, axis = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    boxplot = axis.boxplot(
        distributions,
        tick_labels=[
            f"Cluster {cluster_id}"
            for cluster_id
            in cluster_ids
        ],
        showmeans=True,
        patch_artist=True,
    )

    for patch, cluster_id in zip(
        boxplot[
            "boxes"
        ],
        cluster_ids,
    ):

        patch.set_facecolor(
            CLUSTER_COLORS[
                cluster_id
            ]
        )

        patch.set_alpha(
            0.50
        )

    axis.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        color="0.35",
    )

    axis.set_ylabel(
        "Silhouette score"
    )

    axis.set_title(
        "Silhouette Distribution of the Four Route Clusters"
    )

    axis.grid(
        axis="y",
        alpha=0.20,
    )

    figure.tight_layout()

    return save_figure(
        figure,
        "final_silhouette_distribution",
    )

# Figure 5 — Medoid-distance heatmap
def plot_medoid_distance_heatmap():
    """
    Plot pairwise Hausdorff distances between the four
    representative route medoids.
    """

    table = pd.read_csv(
        MEDOID_DISTANCE_PATH,
        index_col=0,
    )

    matrix = (
        table
        .to_numpy(
            dtype="float64"
        )
    )

    names = [
        name.replace(
            "cluster_",
            "Cluster ",
        )
        for name
        in table.columns
    ]

    figure, axis = plt.subplots(
        figsize=(
            7,
            6,
        )
    )

    image = axis.imshow(
        matrix,
        aspect="equal",
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Hausdorff distance (km)"
    )

    axis.set_xticks(
        np.arange(
            len(names)
        ),
        names,
    )

    axis.set_yticks(
        np.arange(
            len(names)
        ),
        names,
    )

    axis.set_title(
        "Separation Between Representative Route Medoids"
    )

    midpoint = (
        np.nanmin(matrix)
        + np.nanmax(matrix)
    ) / 2.0

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            value = (
                matrix[
                    row,
                    column,
                ]
            )

            text_color = (
                "white"
                if value > midpoint
                else "black"
            )

            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    figure.tight_layout()

    return save_figure(
        figure,
        "final_medoid_distances",
    )

# Main
def main():

    print(
        "=" * 100
    )

    print(
        "GENERATING FINAL VISUALIZATION SET"
    )

    print(
        "=" * 100
    )

    (
        representations,
        labels,
        medoids,
    ) = load_primary_data()

    print(
        "\nInput validation: PASS"
    )

    print(
        "Representations:",
        representations.shape,
    )

    print(
        "Trajectories:",
        len(labels),
    )

    print(
        "Representative clusters:",
        len(medoids),
    )

    saved_files = []

    print(
        "\n[1/5] Geographic final HDBSCAN clusters..."
    )

    saved_files.extend(
        plot_final_route_clusters_map(
            representations,
            labels,
        )
    )

    print(
        "[2/5] Geographic representative routes..."
    )

    saved_files.extend(
        plot_representative_routes_map(
            representations,
            medoids,
        )
    )

    print(
        "[3/5] Month composition..."
    )

    saved_files.extend(
        plot_month_composition(
            labels
        )
    )

    print(
        "[4/5] Silhouette distributions..."
    )

    saved_files.extend(
        plot_silhouette_distribution()
    )

    print(
        "[5/5] Medoid-distance heatmap..."
    )

    saved_files.extend(
        plot_medoid_distance_heatmap()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SAVED FINAL FIGURES"
    )

    print(
        "=" * 100
    )

    for path in saved_files:

        print(path)

    print(
        "\nFINAL VISUALIZATION SET: PASS"
    )


if __name__ == "__main__":
    main()