import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    DISTANCE_OUTPUT_DIR,
    HDBSCAN_OUTPUT_DIR,
    PROCESSED_JANUARY_DIR,
    PROCESSED_JULY_DIR,
)

# Primary HDBSCAN configuration
MIN_SAMPLES = 15
MIN_CLUSTER_SIZE = 30

# Loading
def load_primary_results():
    """
    Load the primary HDBSCAN labels, synchronized trajectory index,
    and full Hausdorff distance matrix.
    """

    label_path = (
        HDBSCAN_OUTPUT_DIR
        / (
            f"labels_"
            f"k{MIN_SAMPLES}_"
            f"mcs{MIN_CLUSTER_SIZE}.csv"
        )
    )

    matrix_path = (
        DISTANCE_OUTPUT_DIR
        / "hausdorff_100.npy"
    )

    index_path = (
        DISTANCE_OUTPUT_DIR
        / "trajectory_index.csv"
    )

    labels = pd.read_csv(
        label_path
    )

    trajectory_index = pd.read_csv(
        index_path
    )

    distance_matrix = np.load(
        matrix_path
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

    trajectory_index = (
        trajectory_index
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    if not np.array_equal(
        labels[
            "matrix_index"
        ].to_numpy(),
        trajectory_index[
            "matrix_index"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "Label table and trajectory index are not aligned."
        )

    if (
        len(labels)
        != distance_matrix.shape[0]
    ):
        raise RuntimeError(
            "Label table and distance matrix size do not match."
        )

    return (
        labels,
        trajectory_index,
        distance_matrix,
    )

# Medoid
def find_cluster_medoid(
    member_indices,
    distance_matrix,
):
    """
    Find the medoid trajectory of one HDBSCAN cluster.

    The medoid is the real trajectory with the smallest
    mean Hausdorff distance to all other members.
    """

    member_indices = np.asarray(
        member_indices,
        dtype="int64",
    )

    cluster_distances = (
        distance_matrix[
            np.ix_(
                member_indices,
                member_indices,
            )
        ]
    )

    n = len(
        member_indices
    )

    if n == 1:
        return {
            "medoid_index": int(
                member_indices[0]
            ),
            "mean_distance_km": 0.0,
            "median_distance_km": 0.0,
            "p95_distance_km": 0.0,
            "maximum_distance_km": 0.0,
        }

    # Diagonal is zero, so dividing the row sum by n-1
    # gives mean distance to all OTHER cluster members.
    mean_distances = (
        cluster_distances.sum(
            axis=1
        )
        / (
            n - 1
        )
    )

    local_medoid_position = int(
        np.argmin(
            mean_distances
        )
    )

    medoid_index = int(
        member_indices[
            local_medoid_position
        ]
    )

    medoid_distances = (
        cluster_distances[
            local_medoid_position
        ]
    )

    # Remove its zero self-distance.
    other_distances = np.delete(
        medoid_distances,
        local_medoid_position,
    )

    return {
        "medoid_index": (
            medoid_index
        ),

        "mean_distance_km": float(
            np.mean(
                other_distances
            )
        ),

        "median_distance_km": float(
            np.median(
                other_distances
            )
        ),

        "p95_distance_km": float(
            np.percentile(
                other_distances,
                95,
            )
        ),

        "maximum_distance_km": float(
            np.max(
                other_distances
            )
        ),
    }

# Processed trajectory loading
def get_trajectory_path(
    month,
    file_name,
):
    """
    Recover one processed trajectory path.
    """

    if month == "2026-01":
        directory = (
            PROCESSED_JANUARY_DIR
        )

    elif month == "2026-07":
        directory = (
            PROCESSED_JULY_DIR
        )

    else:
        raise ValueError(
            f"Unknown month: {month}"
        )

    return (
        directory
        / file_name
    )

# Representative-route analysis
def calculate_representatives(
    labels,
    trajectory_index,
    distance_matrix,
):
    """
    Calculate one medoid for every selected HDBSCAN cluster.
    """

    cluster_labels = sorted(
        int(
            label
        )
        for label in (
            labels[
                "cluster_label"
            ]
            .unique()
        )
        if label >= 0
    )

    records = []

    for cluster_label in (
        cluster_labels
    ):

        member_indices = (
            labels.loc[
                labels[
                    "cluster_label"
                ]
                == cluster_label,
                "matrix_index",
            ]
            .to_numpy(
                dtype="int64"
            )
        )

        result = find_cluster_medoid(
            member_indices,
            distance_matrix,
        )

        medoid_index = (
            result[
                "medoid_index"
            ]
        )

        medoid_row = (
            trajectory_index.loc[
                trajectory_index[
                    "matrix_index"
                ]
                == medoid_index
            ]
            .iloc[0]
        )

        cluster_rows = (
            labels[
                labels[
                    "cluster_label"
                ]
                == cluster_label
            ]
        )

        january_count = int(
            (
                cluster_rows[
                    "month"
                ]
                == "2026-01"
            ).sum()
        )

        july_count = int(
            (
                cluster_rows[
                    "month"
                ]
                == "2026-07"
            ).sum()
        )

        records.append(
            {
                "cluster_label": (
                    cluster_label
                ),

                "cluster_size": (
                    len(
                        member_indices
                    )
                ),

                "january_count": (
                    january_count
                ),

                "july_count": (
                    july_count
                ),

                "medoid_matrix_index": (
                    medoid_index
                ),

                "medoid_month": (
                    medoid_row[
                        "month"
                    ]
                ),

                "medoid_trajectory_id": (
                    medoid_row[
                        "trajectory_id"
                    ]
                ),

                "medoid_file_name": (
                    medoid_row[
                        "file_name"
                    ]
                ),

                "mean_distance_to_members_km": (
                    result[
                        "mean_distance_km"
                    ]
                ),

                "median_distance_to_members_km": (
                    result[
                        "median_distance_km"
                    ]
                ),

                "p95_distance_to_members_km": (
                    result[
                        "p95_distance_km"
                    ]
                ),

                "maximum_distance_to_members_km": (
                    result[
                        "maximum_distance_km"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        records
    )

# Plot
def plot_representative_routes(
    representatives,
):
    """
    Plot only the four representative medoid trajectories.
    """

    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )

    for row in (
        representatives
        .itertuples(
            index=False
        )
    ):

        trajectory_path = (
            get_trajectory_path(
                row.medoid_month,
                row.medoid_file_name,
            )
        )

        trajectory = pd.read_csv(
            trajectory_path,
            usecols=[
                "lat",
                "lon",
            ],
        )

        ax.plot(
            trajectory[
                "lon"
            ],
            trajectory[
                "lat"
            ],
            linewidth=2.2,
            label=(
                f"Cluster "
                f"{row.cluster_label} "
                f"(n={row.cluster_size})"
            ),
        )

    ax.set_title(
        (
            "Representative LAX → JFK Route "
            "for Each HDBSCAN Cluster\n"
            "Medoid trajectory — "
            "min_samples=15, "
            "min_cluster_size=30"
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
        / "representative_routes_k15_mcs30.png"
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

# Main
def main():

    (
        labels,
        trajectory_index,
        distance_matrix,
    ) = load_primary_results()

    representatives = (
        calculate_representatives(
            labels,
            trajectory_index,
            distance_matrix,
        )
    )

    output_table = (
        HDBSCAN_OUTPUT_DIR
        / (
            "representative_routes_"
            "k15_mcs30.csv"
        )
    )

    representatives.to_csv(
        output_table,
        index=False,
    )

    figure_path = (
        plot_representative_routes(
            representatives
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "HDBSCAN REPRESENTATIVE ROUTES"
    )

    print(
        "=" * 100
    )

    print(
        "\nPrimary configuration:"
    )

    print(
        "min_samples = 15"
    )

    print(
        "min_cluster_size = 30"
    )

    print(
        "\nCluster medoids:"
    )

    print(
        representatives.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        output_table
    )

    print(
        figure_path
    )


if __name__ == "__main__":
    main()