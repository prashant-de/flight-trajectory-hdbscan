import numpy as np
import pandas as pd

from src.config import (
    HDBSCAN_OUTPUT_DIR,
)


MIN_SAMPLES = 30

MCS_VALUES = [
    10,
    20,
    30,
]


def load_case(
    min_cluster_size,
):
    """
    Load all saved outputs needed to independently inspect
    one k=30 EOM configuration.
    """

    condensed_tree = pd.read_csv(
        HDBSCAN_OUTPUT_DIR
        / (
            f"condensed_tree_"
            f"k{MIN_SAMPLES}_"
            f"mcs{min_cluster_size}.csv"
        )
    )

    cluster_table = pd.read_csv(
        HDBSCAN_OUTPUT_DIR
        / (
            f"cluster_stability_"
            f"k{MIN_SAMPLES}_"
            f"mcs{min_cluster_size}.csv"
        )
    )

    labels = pd.read_csv(
        HDBSCAN_OUTPUT_DIR
        / (
            f"labels_"
            f"k{MIN_SAMPLES}_"
            f"mcs{min_cluster_size}.csv"
        )
    )

    return (
        condensed_tree,
        cluster_table,
        labels,
    )


def selected_mask(
    series,
):
    """
    Handle either Boolean or string-form selected columns.
    """

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series

    return (
        series
        .astype(str)
        .str.lower()
        .eq("true")
    )


def independently_evaluate_eom(
    condensed_tree,
    cluster_table,
):
    """
    Independently reproduce the EOM parent-vs-descendant
    comparison from the saved condensed tree.

    This does NOT call our original EOM function.
    """

    stability = (
        cluster_table
        .set_index(
            "cluster_id"
        )[
            "stability"
        ]
        .to_dict()
    )

    cluster_ids = (
        cluster_table[
            "cluster_id"
        ]
        .astype(int)
        .tolist()
    )

    children = {
        cluster_id: []
        for cluster_id
        in cluster_ids
    }

    cluster_edges = (
        condensed_tree[
            condensed_tree[
                "child_type"
            ]
            == "cluster"
        ]
    )

    for row in (
        cluster_edges
        .itertuples(
            index=False
        )
    ):
        parent = int(
            row.parent_cluster
        )

        child = int(
            row.child_id
        )

        children[
            parent
        ].append(
            child
        )

    decision_records = []

    def solve(
        cluster_id,
    ):
        """
        Returns:
            selected clusters in this subtree
            effective EOM stability
        """

        child_clusters = (
            children[
                cluster_id
            ]
        )

        own_stability = float(
            stability[
                cluster_id
            ]
        )

        if not child_clusters:

            if cluster_id == 0:
                return (
                    set(),
                    0.0,
                )

            decision_records.append(
                {
                    "cluster_id": (
                        cluster_id
                    ),
                    "own_stability": (
                        own_stability
                    ),
                    "descendant_effective_stability": (
                        0.0
                    ),
                    "margin": (
                        own_stability
                    ),
                    "decision": (
                        "SELECT_LEAF"
                    ),
                }
            )

            return (
                {
                    cluster_id
                },
                own_stability,
            )

        selected_descendants = set()

        descendant_effective = 0.0

        for child in (
            child_clusters
        ):
            (
                child_selected,
                child_effective,
            ) = solve(
                child
            )

            selected_descendants.update(
                child_selected
            )

            descendant_effective += (
                child_effective
            )

        # Root is deliberately not selectable.
        if cluster_id == 0:

            decision_records.append(
                {
                    "cluster_id": 0,
                    "own_stability": (
                        own_stability
                    ),
                    "descendant_effective_stability": (
                        descendant_effective
                    ),
                    "margin": (
                        own_stability
                        - descendant_effective
                    ),
                    "decision": (
                        "ROOT_EXCLUDED"
                    ),
                }
            )

            return (
                selected_descendants,
                descendant_effective,
            )

        margin = (
            own_stability
            - descendant_effective
        )

        if (
            own_stability
            >= descendant_effective
        ):

            decision = (
                "SELECT_PARENT"
            )

            return_selected = {
                cluster_id
            }

            return_stability = (
                own_stability
            )

        else:

            decision = (
                "KEEP_DESCENDANTS"
            )

            return_selected = (
                selected_descendants
            )

            return_stability = (
                descendant_effective
            )

        decision_records.append(
            {
                "cluster_id": (
                    cluster_id
                ),
                "own_stability": (
                    own_stability
                ),
                "descendant_effective_stability": (
                    descendant_effective
                ),
                "margin": (
                    margin
                ),
                "decision": (
                    decision
                ),
            }
        )

        return (
            return_selected,
            return_stability,
        )

    independently_selected, effective_stability = (
        solve(
            0
        )
    )

    decisions = pd.DataFrame(
        decision_records
    )

    return (
        independently_selected,
        effective_stability,
        decisions,
        children,
    )


def analyze_case(
    min_cluster_size,
):
    """
    Inspect one k=30 configuration.
    """

    (
        condensed_tree,
        cluster_table,
        labels,
    ) = load_case(
        min_cluster_size
    )

    (
        independent_selected,
        independent_stability,
        decisions,
        children,
    ) = independently_evaluate_eom(
        condensed_tree,
        cluster_table,
    )

    saved_selected = set(
        cluster_table.loc[
            selected_mask(
                cluster_table[
                    "selected"
                ]
            ),
            "cluster_id",
        ]
        .astype(int)
        .tolist()
    )

    selection_match = (
        saved_selected
        == independent_selected
    )

    # -------------------------------------------------------------
    # Assigned trajectory count per selected condensed cluster
    # -------------------------------------------------------------

    assigned_counts = (
        labels[
            labels[
                "selected_cluster_id"
            ]
            >= 0
        ]
        .groupby(
            "selected_cluster_id"
        )
        .size()
        .to_dict()
    )

    cluster_table = (
        cluster_table
        .copy()
    )

    cluster_table[
        "assigned_trajectories"
    ] = (
        cluster_table[
            "cluster_id"
        ]
        .map(
            assigned_counts
        )
        .fillna(
            0
        )
        .astype(int)
    )

    cluster_table[
        "direct_child_clusters"
    ] = (
        cluster_table[
            "cluster_id"
        ]
        .map(
            lambda x: len(
                children[
                    int(x)
                ]
            )
        )
    )

    decision_lookup = (
        decisions
        .set_index(
            "cluster_id"
        )
    )

    cluster_table[
        "descendant_effective_stability"
    ] = (
        cluster_table[
            "cluster_id"
        ]
        .map(
            decision_lookup[
                "descendant_effective_stability"
            ]
        )
    )

    cluster_table[
        "eom_margin"
    ] = (
        cluster_table[
            "cluster_id"
        ]
        .map(
            decision_lookup[
                "margin"
            ]
        )
    )

    cluster_table[
        "independent_decision"
    ] = (
        cluster_table[
            "cluster_id"
        ]
        .map(
            decision_lookup[
                "decision"
            ]
        )
    )

    noise_count = int(
        (
            labels[
                "cluster_label"
            ]
            == -1
        )
        .sum()
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"k=30, min_cluster_size="
        f"{min_cluster_size}"
    )

    print(
        "=" * 90
    )

    print(
        "\nSaved selected clusters:",
        sorted(
            saved_selected
        ),
    )

    print(
        "Independent selected clusters:",
        sorted(
            independent_selected
        ),
    )

    print(
        "Independent EOM matches saved result:",
        selection_match,
    )

    print(
        "Independent selected stability:",
        f"{independent_stability:.6f}",
    )

    print(
        "Noise count:",
        noise_count,
    )

    print(
        "\nCandidate cluster decision table:"
    )

    display_columns = [
        "cluster_id",
        "birth_size",
        "birth_lambda",
        "death_lambda",
        "stability",
        "direct_child_clusters",
        "descendant_effective_stability",
        "eom_margin",
        "independent_decision",
        "assigned_trajectories",
    ]

    print(
        cluster_table[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nSelected clusters and assigned sizes:"
    )

    selected_table = (
        cluster_table[
            cluster_table[
                "cluster_id"
            ]
            .isin(
                saved_selected
            )
        ][
            display_columns
        ]
    )

    print(
        selected_table.to_string(
            index=False
        )
    )

    return labels


def compare_neighboring_cases(
    labels_10,
    labels_20,
    labels_30,
):
    """
    Determine where the very large mcs=20 cluster comes from
    under the neighboring mcs=10 and mcs=30 solutions.
    """

    labels_10 = (
        labels_10
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    labels_20 = (
        labels_20
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    labels_30 = (
        labels_30
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    if not np.array_equal(
        labels_10[
            "matrix_index"
        ].to_numpy(),
        labels_20[
            "matrix_index"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "mcs10 and mcs20 label rows are not aligned."
        )

    if not np.array_equal(
        labels_20[
            "matrix_index"
        ].to_numpy(),
        labels_30[
            "matrix_index"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "mcs20 and mcs30 label rows are not aligned."
        )

    # -------------------------------------------------------------
    # Identify largest selected cluster at mcs=20
    # -------------------------------------------------------------

    clustered_20 = (
        labels_20[
            labels_20[
                "cluster_label"
            ]
            >= 0
        ]
    )

    largest_label = int(
        clustered_20[
            "cluster_label"
        ]
        .value_counts()
        .idxmax()
    )

    largest_mask = (
        labels_20[
            "cluster_label"
        ]
        == largest_label
    )

    largest_size = int(
        largest_mask.sum()
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MEMBERSHIP ORIGIN OF THE LARGE mcs=20 CLUSTER"
    )

    print(
        "=" * 90
    )

    print(
        "\nmcs=20 largest final label:",
        largest_label,
    )

    print(
        "mcs=20 largest cluster size:",
        largest_size,
    )

    comparison = pd.DataFrame(
        {
            "mcs10_label": (
                labels_10.loc[
                    largest_mask,
                    "cluster_label",
                ]
                .to_numpy()
            ),

            "mcs30_label": (
                labels_30.loc[
                    largest_mask,
                    "cluster_label",
                ]
                .to_numpy()
            ),
        }
    )

    print(
        "\nThose same trajectories under mcs=10:"
    )

    print(
        comparison[
            "mcs10_label"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()
    )

    print(
        "\nThose same trajectories under mcs=30:"
    )

    print(
        comparison[
            "mcs30_label"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()
    )

    # -------------------------------------------------------------
    # Noise transitions
    # -------------------------------------------------------------

    noise_10 = (
        labels_10[
            "cluster_label"
        ]
        == -1
    )

    noise_20 = (
        labels_20[
            "cluster_label"
        ]
        == -1
    )

    noise_30 = (
        labels_30[
            "cluster_label"
        ]
        == -1
    )

    print(
        "\nNoise-transition counts:"
    )

    print(
        "mcs10 noise -> mcs20 cluster:",
        int(
            (
                noise_10
                & ~noise_20
            ).sum()
        ),
    )

    print(
        "mcs10 cluster -> mcs20 noise:",
        int(
            (
                ~noise_10
                & noise_20
            ).sum()
        ),
    )

    print(
        "mcs20 cluster -> mcs30 noise:",
        int(
            (
                ~noise_20
                & noise_30
            ).sum()
        ),
    )

    print(
        "mcs20 noise -> mcs30 cluster:",
        int(
            (
                noise_20
                & ~noise_30
            ).sum()
        ),
    )


def main():

    label_results = {}

    for min_cluster_size in (
        MCS_VALUES
    ):
        labels = analyze_case(
            min_cluster_size
        )

        label_results[
            min_cluster_size
        ] = labels

    compare_neighboring_cases(
        label_results[
            10
        ],
        label_results[
            20
        ],
        label_results[
            30
        ],
    )


if __name__ == "__main__":
    main()