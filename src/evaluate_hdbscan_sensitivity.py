import numpy as np
import pandas as pd

from src.config import (
    HDBSCAN_OUTPUT_DIR,
)


MIN_SAMPLES_CANDIDATES = [
    5,
    10,
    15,
    20,
    30,
]

MIN_CLUSTER_SIZE_CANDIDATES = [
    5,
    10,
    20,
    30,
    50,
    75,
]


def load_labels(
    min_samples,
    min_cluster_size,
):
    """
    Load one final HDBSCAN labeling.
    """

    path = (
        HDBSCAN_OUTPUT_DIR
        / (
            f"labels_"
            f"k{min_samples}_"
            f"mcs{min_cluster_size}.csv"
        )
    )

    table = pd.read_csv(
        path
    )

    table = (
        table
        .sort_values(
            "matrix_index"
        )
        .reset_index(
            drop=True
        )
    )

    return table


def comb2(
    values,
):
    """
    Number of unordered pairs from each count.
    """

    values = np.asarray(
        values,
        dtype="float64",
    )

    return (
        values
        * (
            values - 1
        )
        / 2
    )


def adjusted_rand_index(
    labels_a,
    labels_b,
):
    """
    Manual Adjusted Rand Index implementation.

    Cluster label numbers themselves do not need to match.
    """

    labels_a = np.asarray(
        labels_a
    )

    labels_b = np.asarray(
        labels_b
    )

    if (
        labels_a.shape
        != labels_b.shape
    ):
        raise ValueError(
            "Label arrays must have the same shape."
        )

    n = len(
        labels_a
    )

    if n < 2:
        return 1.0

    unique_a, inverse_a = np.unique(
        labels_a,
        return_inverse=True,
    )

    unique_b, inverse_b = np.unique(
        labels_b,
        return_inverse=True,
    )

    contingency = np.zeros(
        (
            len(
                unique_a
            ),
            len(
                unique_b
            ),
        ),
        dtype="int64",
    )

    np.add.at(
        contingency,
        (
            inverse_a,
            inverse_b,
        ),
        1,
    )

    sum_comb_cells = (
        comb2(
            contingency
        )
        .sum()
    )

    row_sums = (
        contingency
        .sum(
            axis=1
        )
    )

    column_sums = (
        contingency
        .sum(
            axis=0
        )
    )

    sum_comb_rows = (
        comb2(
            row_sums
        )
        .sum()
    )

    sum_comb_columns = (
        comb2(
            column_sums
        )
        .sum()
    )

    total_pairs = (
        n
        * (
            n - 1
        )
        / 2
    )

    expected_index = (
        sum_comb_rows
        * sum_comb_columns
        / total_pairs
    )

    maximum_index = (
        0.5
        * (
            sum_comb_rows
            + sum_comb_columns
        )
    )

    denominator = (
        maximum_index
        - expected_index
    )

    numerator = (
        sum_comb_cells
        - expected_index
    )

    if abs(
        denominator
    ) < 1e-15:
        return 1.0

    return float(
        numerator
        / denominator
    )


def compare_labelings(
    table_a,
    table_b,
):
    """
    Compare two final HDBSCAN results.
    """

    indices_a = (
        table_a[
            "matrix_index"
        ]
        .to_numpy()
    )

    indices_b = (
        table_b[
            "matrix_index"
        ]
        .to_numpy()
    )

    if not np.array_equal(
        indices_a,
        indices_b,
    ):
        raise RuntimeError(
            "Trajectory indices are not aligned."
        )

    labels_a = (
        table_a[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    labels_b = (
        table_b[
            "cluster_label"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    # -------------------------------------------------------------
    # ARI using all trajectories.
    #
    # Noise (-1) is treated as one label here.
    # -------------------------------------------------------------

    ari_all = (
        adjusted_rand_index(
            labels_a,
            labels_b,
        )
    )

    # -------------------------------------------------------------
    # Compare cluster organization among trajectories that are
    # clustered in BOTH configurations.
    # -------------------------------------------------------------

    common_clustered = (
        (labels_a >= 0)
        & (labels_b >= 0)
    )

    common_clustered_count = int(
        common_clustered.sum()
    )

    if (
        common_clustered_count
        >= 2
    ):
        ari_common_clustered = (
            adjusted_rand_index(
                labels_a[
                    common_clustered
                ],
                labels_b[
                    common_clustered
                ],
            )
        )
    else:
        ari_common_clustered = (
            np.nan
        )

    # -------------------------------------------------------------
    # Noise-set agreement
    # -------------------------------------------------------------

    noise_a = (
        labels_a == -1
    )

    noise_b = (
        labels_b == -1
    )

    noise_intersection = int(
        (
            noise_a
            & noise_b
        ).sum()
    )

    noise_union = int(
        (
            noise_a
            | noise_b
        ).sum()
    )

    if noise_union == 0:
        noise_jaccard = 1.0

    else:
        noise_jaccard = (
            noise_intersection
            / noise_union
        )

    noise_mask_agreement = float(
        (
            noise_a
            == noise_b
        ).mean()
    )

    return {
        "ari_all": (
            ari_all
        ),

        "ari_common_clustered": (
            ari_common_clustered
        ),

        "common_clustered": (
            common_clustered_count
        ),

        "noise_jaccard": (
            noise_jaccard
        ),

        "noise_mask_agreement": (
            noise_mask_agreement
        ),
    }


def main():

    # -------------------------------------------------------------
    # Load all 30 configurations.
    # -------------------------------------------------------------

    results = {}

    for k in (
        MIN_SAMPLES_CANDIDATES
    ):
        for mcs in (
            MIN_CLUSTER_SIZE_CANDIDATES
        ):
            results[
                (
                    k,
                    mcs,
                )
            ] = load_labels(
                k,
                mcs,
            )

    # -------------------------------------------------------------
    # A. Adjacent min_cluster_size comparisons
    # while min_samples is fixed.
    # -------------------------------------------------------------

    mcs_records = []

    for k in (
        MIN_SAMPLES_CANDIDATES
    ):

        for i in range(
            len(
                MIN_CLUSTER_SIZE_CANDIDATES
            )
            - 1
        ):
            mcs_a = (
                MIN_CLUSTER_SIZE_CANDIDATES[
                    i
                ]
            )

            mcs_b = (
                MIN_CLUSTER_SIZE_CANDIDATES[
                    i + 1
                ]
            )

            metrics = compare_labelings(
                results[
                    (
                        k,
                        mcs_a,
                    )
                ],
                results[
                    (
                        k,
                        mcs_b,
                    )
                ],
            )

            mcs_records.append(
                {
                    "min_samples": (
                        k
                    ),

                    "mcs_a": (
                        mcs_a
                    ),

                    "mcs_b": (
                        mcs_b
                    ),

                    **metrics,
                }
            )

    mcs_summary = pd.DataFrame(
        mcs_records
    )

    # -------------------------------------------------------------
    # B. Adjacent min_samples comparisons
    # while min_cluster_size is fixed.
    # -------------------------------------------------------------

    k_records = []

    for mcs in (
        MIN_CLUSTER_SIZE_CANDIDATES
    ):

        for i in range(
            len(
                MIN_SAMPLES_CANDIDATES
            )
            - 1
        ):
            k_a = (
                MIN_SAMPLES_CANDIDATES[
                    i
                ]
            )

            k_b = (
                MIN_SAMPLES_CANDIDATES[
                    i + 1
                ]
            )

            metrics = compare_labelings(
                results[
                    (
                        k_a,
                        mcs,
                    )
                ],
                results[
                    (
                        k_b,
                        mcs,
                    )
                ],
            )

            k_records.append(
                {
                    "min_cluster_size": (
                        mcs
                    ),

                    "k_a": (
                        k_a
                    ),

                    "k_b": (
                        k_b
                    ),

                    **metrics,
                }
            )

    k_summary = pd.DataFrame(
        k_records
    )

    # -------------------------------------------------------------
    # C. Detect exact-equivalent partitions.
    #
    # ARI=1 alone is not enough because noise masks could differ
    # in unusual degenerate cases, so require both.
    # -------------------------------------------------------------

    equivalent_records = []

    configurations = list(
        results.keys()
    )

    for i in range(
        len(
            configurations
        )
    ):
        config_a = (
            configurations[
                i
            ]
        )

        for j in range(
            i + 1,
            len(
                configurations
            )
        ):
            config_b = (
                configurations[
                    j
                ]
            )

            metrics = compare_labelings(
                results[
                    config_a
                ],
                results[
                    config_b
                ],
            )

            if (
                np.isclose(
                    metrics[
                        "ari_all"
                    ],
                    1.0,
                    atol=1e-12,
                )
                and np.isclose(
                    metrics[
                        "noise_mask_agreement"
                    ],
                    1.0,
                    atol=1e-12,
                )
            ):
                equivalent_records.append(
                    {
                        "k_a": (
                            config_a[
                                0
                            ]
                        ),

                        "mcs_a": (
                            config_a[
                                1
                            ]
                        ),

                        "k_b": (
                            config_b[
                                0
                            ]
                        ),

                        "mcs_b": (
                            config_b[
                                1
                            ]
                        ),
                    }
                )

    equivalent = pd.DataFrame(
        equivalent_records
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    mcs_summary.to_csv(
        HDBSCAN_OUTPUT_DIR
        / "sensitivity_adjacent_mcs.csv",
        index=False,
    )

    k_summary.to_csv(
        HDBSCAN_OUTPUT_DIR
        / "sensitivity_adjacent_min_samples.csv",
        index=False,
    )

    equivalent.to_csv(
        HDBSCAN_OUTPUT_DIR
        / "sensitivity_equivalent_partitions.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "HDBSCAN FINAL-LABEL SENSITIVITY"
    )

    print(
        "=" * 100
    )

    print(
        "\nA. Adjacent min_cluster_size comparisons"
    )

    print(
        mcs_summary.to_string(
            index=False
        )
    )

    print(
        "\nB. Adjacent min_samples comparisons"
    )

    print(
        k_summary.to_string(
            index=False
        )
    )

    print(
        "\nC. Exact-equivalent parameter pairs"
    )

    if len(
        equivalent
    ) == 0:
        print(
            "None"
        )

    else:
        print(
            equivalent.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()