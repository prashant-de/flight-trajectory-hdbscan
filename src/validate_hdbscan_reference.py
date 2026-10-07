"""
Validate the final manual HDBSCAN result against
scikit-learn's reference HDBSCAN implementation.

Manual configuration
--------------------
min_samples = 15
min_cluster_size = 30

Important convention
--------------------
Our manual implementation defines min_samples=15 as
the 15th nearest OTHER trajectory.

scikit-learn HDBSCAN counts the point itself, so the
equivalent reference value is:

    min_samples = 16

Both implementations use:
    precomputed distance matrix
    EOM cluster selection
    allow_single_cluster = False
"""

import json

import numpy as np
import pandas as pd
import sklearn

from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

from src.config import (
    DISTANCE_OUTPUT_DIR,
    HDBSCAN_OUTPUT_DIR,
)

# Parameters
MANUAL_MIN_SAMPLES = 15

REFERENCE_MIN_SAMPLES = (
    MANUAL_MIN_SAMPLES + 1
)

MIN_CLUSTER_SIZE = 30

# Paths
FINAL_MANUAL_DIR = (
    HDBSCAN_OUTPUT_DIR
    / "final"
)

REFERENCE_OUTPUT_DIR = (
    HDBSCAN_OUTPUT_DIR
    / "reference_validation"
)

# Loading
def load_inputs():
    """
    Load the frozen distance matrix and final manual labels.
    """

    distance_matrix = np.load(
        DISTANCE_OUTPUT_DIR
        / "hausdorff_100.npy"
    )

    manual_labels_table = pd.read_csv(
        FINAL_MANUAL_DIR
        / "labels.csv"
    )

    manual_labels_table = (
        manual_labels_table
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            manual_labels_table
        )
        != distance_matrix.shape[0]
    ):
        raise RuntimeError(
            "Manual labels and distance matrix "
            "have different sizes."
        )

    return (
        distance_matrix,
        manual_labels_table,
    )

# Basic statistics
def summarize_labels(
    labels,
):
    """
    Summarize one clustering result.
    """

    labels = np.asarray(
        labels,
        dtype="int64",
    )

    clustered_labels = (
        labels[
            labels >= 0
        ]
    )

    unique_clusters = np.unique(
        clustered_labels
    )

    cluster_sizes = []

    for cluster_label in (
        unique_clusters
    ):
        cluster_sizes.append(
            int(
                (
                    labels
                    == cluster_label
                ).sum()
            )
        )

    cluster_sizes = sorted(
        cluster_sizes
    )

    noise_count = int(
        (
            labels == -1
        ).sum()
    )

    return {
        "cluster_count": (
            int(
                len(
                    unique_clusters
                )
            )
        ),

        "cluster_sizes": (
            cluster_sizes
        ),

        "noise_count": (
            noise_count
        ),

        "noise_percent": (
            float(
                noise_count
                / len(
                    labels
                )
                * 100
            )
        ),
    }

# Comparison metrics
def compare_clusterings(
    manual_labels,
    reference_labels,
):
    """
    Compare manual and reference partitions.

    Cluster numeric IDs are arbitrary, so comparison uses
    label-invariant metrics rather than direct label equality.
    """

    manual_labels = np.asarray(
        manual_labels,
        dtype="int64",
    )

    reference_labels = np.asarray(
        reference_labels,
        dtype="int64",
    )

    if (
        manual_labels.shape
        != reference_labels.shape
    ):
        raise ValueError(
            "Label arrays have different shapes."
        )
    # Overall Adjusted Rand Index
    ari_all = float(
        adjusted_rand_score(
            manual_labels,
            reference_labels,
        )
    )
    # Compare route grouping only where BOTH implementations
    # classify the trajectory into a cluster.
    common_clustered = (
        (manual_labels >= 0)
        & (reference_labels >= 0)
    )

    common_clustered_count = int(
        common_clustered.sum()
    )

    if (
        common_clustered_count
        >= 2
    ):
        ari_common_clustered = float(
            adjusted_rand_score(
                manual_labels[
                    common_clustered
                ],
                reference_labels[
                    common_clustered
                ],
            )
        )

    else:
        ari_common_clustered = (
            np.nan
        )
    # Noise comparison
    manual_noise = (
        manual_labels == -1
    )

    reference_noise = (
        reference_labels == -1
    )

    noise_intersection = int(
        (
            manual_noise
            & reference_noise
        ).sum()
    )

    noise_union = int(
        (
            manual_noise
            | reference_noise
        ).sum()
    )

    if noise_union == 0:
        noise_jaccard = 1.0

    else:
        noise_jaccard = float(
            noise_intersection
            / noise_union
        )

    noise_mask_agreement = float(
        (
            manual_noise
            == reference_noise
        ).mean()
    )

    manual_noise_reference_cluster = int(
        (
            manual_noise
            & ~reference_noise
        ).sum()
    )

    manual_cluster_reference_noise = int(
        (
            ~manual_noise
            & reference_noise
        ).sum()
    )

    return {
        "ari_all": (
            ari_all
        ),

        "ari_common_clustered": (
            ari_common_clustered
        ),

        "common_clustered_count": (
            common_clustered_count
        ),

        "noise_intersection": (
            noise_intersection
        ),

        "noise_union": (
            noise_union
        ),

        "noise_jaccard": (
            noise_jaccard
        ),

        "noise_mask_agreement": (
            noise_mask_agreement
        ),

        "manual_noise_reference_cluster": (
            manual_noise_reference_cluster
        ),

        "manual_cluster_reference_noise": (
            manual_cluster_reference_noise
        ),
    }

# Membership cross-tab
def build_membership_crosstab(
    manual_labels,
    reference_labels,
):
    """
    Build a contingency table showing how the manual
    clusters map to reference clusters.

    Noise is retained as label -1.
    """

    table = pd.DataFrame(
        {
            "manual_label": (
                manual_labels
            ),

            "reference_label": (
                reference_labels
            ),
        }
    )

    return pd.crosstab(
        table[
            "manual_label"
        ],
        table[
            "reference_label"
        ],
        margins=True,
    )

# Main
def main():

    print(
        "=" * 90
    )

    print(
        "MANUAL HDBSCAN VS SCIKIT-LEARN REFERENCE"
    )

    print(
        "=" * 90
    )

    print(
        "\nscikit-learn version:",
        sklearn.__version__,
    )

    print(
        "\nManual configuration:"
    )

    print(
        "min_samples =",
        MANUAL_MIN_SAMPLES,
        "(15 nearest OTHER trajectories)",
    )

    print(
        "min_cluster_size =",
        MIN_CLUSTER_SIZE,
    )

    print(
        "\nReference configuration:"
    )

    print(
        "min_samples =",
        REFERENCE_MIN_SAMPLES,
        "(scikit-learn counts self)",
    )

    print(
        "min_cluster_size =",
        MIN_CLUSTER_SIZE,
    )

    print(
        "metric = precomputed"
    )

    print(
        "cluster_selection_method = eom"
    )

    print(
        "allow_single_cluster = False"
    )
    # Load
    (
        distance_matrix,
        manual_table,
    ) = load_inputs()

    manual_labels = (
        manual_table[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    print(
        "\nDistance matrix:",
        distance_matrix.shape,
    )
    # Run reference HDBSCAN
    print(
        "\nRunning scikit-learn HDBSCAN..."
    )

    model = HDBSCAN(
        min_cluster_size=(
            MIN_CLUSTER_SIZE
        ),

        min_samples=(
            REFERENCE_MIN_SAMPLES
        ),

        metric="precomputed",

        algorithm="brute",

        cluster_selection_method="eom",

        allow_single_cluster=False,

        cluster_selection_epsilon=0.0,

        alpha=1.0,

        copy=True,
    )

    reference_labels = (
        model.fit_predict(
            distance_matrix
        )
    )

    reference_labels = np.asarray(
        reference_labels,
        dtype="int64",
    )
    # Sanity-check reference outlier labels
    unexpected_negative_labels = (
        set(
            np.unique(
                reference_labels[
                    reference_labels < 0
                ]
            )
        )
        - {
            -1,
        }
    )

    if unexpected_negative_labels:
        raise RuntimeError(
            "Reference implementation returned unexpected "
            f"negative labels: {unexpected_negative_labels}"
        )
    # Summaries
    manual_summary = (
        summarize_labels(
            manual_labels
        )
    )

    reference_summary = (
        summarize_labels(
            reference_labels
        )
    )

    comparison = (
        compare_clusterings(
            manual_labels,
            reference_labels,
        )
    )

    crosstab = (
        build_membership_crosstab(
            manual_labels,
            reference_labels,
        )
    )
    # Save reference result
    REFERENCE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_table = (
        manual_table[
            [
                "matrix_index",
                "month",
                "trajectory_id",
                "file_name",
            ]
        ]
        .copy()
    )

    reference_table[
        "reference_cluster_label"
    ] = (
        reference_labels
    )

    reference_table[
        "reference_probability"
    ] = (
        model.probabilities_
    )

    reference_table.to_csv(
        REFERENCE_OUTPUT_DIR
        / "reference_labels.csv",
        index=False,
    )

    crosstab.to_csv(
        REFERENCE_OUTPUT_DIR
        / "manual_reference_crosstab.csv"
    )

    report = {
        "scikit_learn_version": (
            sklearn.__version__
        ),

        "manual_parameters": {
            "min_samples": (
                MANUAL_MIN_SAMPLES
            ),

            "min_cluster_size": (
                MIN_CLUSTER_SIZE
            ),
        },

        "reference_parameters": {
            "min_samples": (
                REFERENCE_MIN_SAMPLES
            ),

            "min_cluster_size": (
                MIN_CLUSTER_SIZE
            ),

            "metric": (
                "precomputed"
            ),

            "cluster_selection_method": (
                "eom"
            ),

            "allow_single_cluster": (
                False
            ),

            "alpha": (
                1.0
            ),
        },

        "manual_summary": (
            manual_summary
        ),

        "reference_summary": (
            reference_summary
        ),

        "comparison": (
            comparison
        ),
    }

    with open(
        REFERENCE_OUTPUT_DIR
        / "validation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )
    # Report
    print(
        "\n"
        + "=" * 90
    )

    print(
        "CLUSTERING SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        "\nManual:"
    )

    print(
        "Clusters:",
        manual_summary[
            "cluster_count"
        ],
    )

    print(
        "Cluster sizes:",
        manual_summary[
            "cluster_sizes"
        ],
    )

    print(
        "Noise:",
        manual_summary[
            "noise_count"
        ],
        (
            f"({manual_summary['noise_percent']:.3f}%)"
        ),
    )

    print(
        "\nReference:"
    )

    print(
        "Clusters:",
        reference_summary[
            "cluster_count"
        ],
    )

    print(
        "Cluster sizes:",
        reference_summary[
            "cluster_sizes"
        ],
    )

    print(
        "Noise:",
        reference_summary[
            "noise_count"
        ],
        (
            f"({reference_summary['noise_percent']:.3f}%)"
        ),
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PARTITION AGREEMENT"
    )

    print(
        "=" * 90
    )

    print(
        "\nAdjusted Rand Index (all trajectories):",
        f"{comparison['ari_all']:.6f}",
    )

    print(
        "ARI among trajectories clustered by both:",
        (
            f"{comparison['ari_common_clustered']:.6f}"
        ),
    )

    print(
        "Trajectories clustered by both:",
        comparison[
            "common_clustered_count"
        ],
    )

    print(
        "\nNoise-set Jaccard:",
        f"{comparison['noise_jaccard']:.6f}",
    )

    print(
        "Noise-mask agreement:",
        (
            f"{comparison['noise_mask_agreement']:.6f}"
        ),
    )

    print(
        "Noise in both:",
        comparison[
            "noise_intersection"
        ],
    )

    print(
        "Manual noise but reference clustered:",
        comparison[
            "manual_noise_reference_cluster"
        ],
    )

    print(
        "Manual clustered but reference noise:",
        comparison[
            "manual_cluster_reference_noise"
        ],
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MANUAL ↔ REFERENCE MEMBERSHIP TABLE"
    )

    print(
        "=" * 90
    )

    print(
        "\n",
        crosstab.to_string(),
        sep="",
    )

    print(
        "\nSaved:"
    )

    print(
        "outputs/clusters/reference_validation/"
    )

    print(
        "  reference_labels.csv"
    )

    print(
        "  manual_reference_crosstab.csv"
    )

    print(
        "  validation_summary.json"
    )


if __name__ == "__main__":
    main()