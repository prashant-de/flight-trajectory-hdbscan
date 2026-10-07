"""
Manual HDBSCAN implementation for a precomputed distance matrix.

Pipeline
--------
1. Core distances
2. Mutual-reachability distances
3. Minimum spanning tree using Prim's algorithm
4. Single-linkage hierarchy
5. Condensed hierarchy using min_cluster_size
6. Cluster stability
7. Excess-of-Mass (EOM) cluster selection
8. Final trajectory labels

Important convention
--------------------
In this project, min_samples = k means the distance to the
k-th nearest OTHER sample.

Because the input distance matrix contains zero self-distance,
the required value is therefore sorted_distances[k].

This convention must remain consistent throughout the project.
"""

import numpy as np
import pandas as pd


# Distance-matrix validation

def validate_distance_matrix(
    distance_matrix,
):
    """
    Validate a precomputed distance matrix.

    Returns
    -------
    list[str]
        Empty list when validation passes.
    """

    problems = []

    if distance_matrix.ndim != 2:
        problems.append(
            "distance_matrix_not_2d"
        )

        return problems

    if (
        distance_matrix.shape[0]
        != distance_matrix.shape[1]
    ):
        problems.append(
            "distance_matrix_not_square"
        )

        return problems

    if not np.all(
        np.isfinite(
            distance_matrix
        )
    ):
        problems.append(
            "nonfinite_distances"
        )

    if np.any(
        distance_matrix < 0
    ):
        problems.append(
            "negative_distances"
        )

    if not np.allclose(
        distance_matrix,
        distance_matrix.T,
        atol=1e-12,
    ):
        problems.append(
            "distance_matrix_not_symmetric"
        )

    if not np.allclose(
        np.diag(
            distance_matrix
        ),
        0.0,
        atol=1e-12,
    ):
        problems.append(
            "nonzero_diagonal"
        )

    return problems


# 1. Core distances

def compute_core_distances(
    distance_matrix,
    min_samples,
):
    """
    Compute the core distance for every sample.

    Project convention:
        min_samples = k
        -> distance to the k-th nearest OTHER sample.

    Since self-distance is zero and occupies position 0,
    partition index k gives the required neighbor distance.
    """

    n = distance_matrix.shape[0]

    if min_samples < 1:
        raise ValueError(
            "min_samples must be at least 1."
        )

    if min_samples >= n:
        raise ValueError(
            "min_samples must be smaller than "
            "the number of samples."
        )

    partitioned = np.partition(
        distance_matrix,
        kth=min_samples,
        axis=1,
    )

    return (
        partitioned[
            :,
            min_samples
        ]
        .astype(
            "float64",
            copy=False,
        )
    )


# 2. Mutual reachability

def compute_mutual_reachability(
    distance_matrix,
    core_distances,
):
    """
    Compute the complete mutual-reachability matrix.

    mreach(a, b) =
        max(
            core(a),
            core(b),
            distance(a, b)
        )
    """

    n = distance_matrix.shape[0]

    if len(
        core_distances
    ) != n:
        raise ValueError(
            "Core-distance length does not match "
            "distance-matrix size."
        )

    mutual = np.maximum(
        distance_matrix,
        core_distances[
            :,
            None,
        ],
    )

    mutual = np.maximum(
        mutual,
        core_distances[
            None,
            :,
        ],
    )

    return mutual


def validate_mutual_reachability(
    distance_matrix,
    mutual_reachability,
    core_distances,
):
    """
    Validate the mutual-reachability matrix.
    """

    problems = []

    if (
        mutual_reachability.shape
        != distance_matrix.shape
    ):
        problems.append(
            "shape_mismatch"
        )

        return problems

    if not np.all(
        np.isfinite(
            mutual_reachability
        )
    ):
        problems.append(
            "nonfinite_values"
        )

    if not np.allclose(
        mutual_reachability,
        mutual_reachability.T,
        atol=1e-12,
    ):
        problems.append(
            "not_symmetric"
        )

    if np.any(
        mutual_reachability
        + 1e-12
        < distance_matrix
    ):
        problems.append(
            "smaller_than_original_distance"
        )

    if not np.allclose(
        np.diag(
            mutual_reachability
        ),
        core_distances,
        atol=1e-12,
    ):
        problems.append(
            "incorrect_diagonal"
        )

    return problems


# 3. Minimum spanning tree

def compute_mst_prim(
    mutual_reachability,
):
    """
    Construct the minimum spanning tree manually using
    Prim's algorithm.

    Returns
    -------
    pandas.DataFrame

    Columns
    -------
    source
    target
    weight_km
    """

    n = mutual_reachability.shape[0]

    if (
        mutual_reachability.ndim != 2
        or mutual_reachability.shape[1] != n
    ):
        raise ValueError(
            "Mutual-reachability matrix must be square."
        )

    in_tree = np.zeros(
        n,
        dtype=bool,
    )

    best_weight = np.full(
        n,
        np.inf,
        dtype="float64",
    )

    parent = np.full(
        n,
        -1,
        dtype="int64",
    )

    # Deterministic starting vertex.
    best_weight[0] = 0.0

    edges = []

    for _ in range(
        n
    ):

        candidate_weights = np.where(
            in_tree,
            np.inf,
            best_weight,
        )

        u = int(
            np.argmin(
                candidate_weights
            )
        )

        if not np.isfinite(
            candidate_weights[u]
        ):
            raise RuntimeError(
                "Graph is disconnected; "
                "MST cannot be completed."
            )

        in_tree[u] = True

        if parent[u] != -1:
            edges.append(
                {
                    "source": int(
                        parent[u]
                    ),

                    "target": int(
                        u
                    ),

                    "weight_km": float(
                        best_weight[u]
                    ),
                }
            )

        row = mutual_reachability[
            u
        ]

        update_mask = (
            (~in_tree)
            & (
                row
                < best_weight
            )
        )

        best_weight[
            update_mask
        ] = row[
            update_mask
        ]

        parent[
            update_mask
        ] = u

    return pd.DataFrame(
        edges
    )


def validate_mst(
    mst_edges,
    n_vertices,
):
    """
    Validate edge count, connectivity and acyclicity.
    """

    problems = []

    if len(
        mst_edges
    ) != (
        n_vertices - 1
    ):
        problems.append(
            "incorrect_edge_count"
        )

        return problems

    if not np.all(
        np.isfinite(
            mst_edges[
                "weight_km"
            ]
        )
    ):
        problems.append(
            "nonfinite_edge_weight"
        )

    if np.any(
        mst_edges[
            "weight_km"
        ]
        < 0
    ):
        problems.append(
            "negative_edge_weight"
        )

    # Manual union-find validation

    parent = np.arange(
        n_vertices,
        dtype="int64",
    )

    def find(
        x,
    ):
        while (
            parent[x] != x
        ):
            parent[x] = (
                parent[
                    parent[x]
                ]
            )

            x = parent[x]

        return x

    def union(
        a,
        b,
    ):
        root_a = find(
            a
        )

        root_b = find(
            b
        )

        if root_a == root_b:
            return False

        parent[
            root_b
        ] = root_a

        return True

    for row in (
        mst_edges.itertuples(
            index=False
        )
    ):
        if not union(
            int(
                row.source
            ),
            int(
                row.target
            ),
        ):
            problems.append(
                "cycle_detected"
            )

            break

    roots = {
        find(
            vertex
        )
        for vertex
        in range(
            n_vertices
        )
    }

    if len(
        roots
    ) != 1:
        problems.append(
            "mst_not_connected"
        )

    return problems


# 4. Single-linkage hierarchy

def build_single_linkage_hierarchy(
    mst_edges,
    n_vertices,
):
    """
    Convert the MST into a binary single-linkage hierarchy.

    Leaf IDs:
        0 ... n_vertices - 1

    Internal cluster IDs:
        n_vertices ... 2*n_vertices - 2
    """

    if len(
        mst_edges
    ) != (
        n_vertices - 1
    ):
        raise ValueError(
            "MST must contain exactly N - 1 edges."
        )

    edges = (
        mst_edges
        .sort_values(
            [
                "weight_km",
                "source",
                "target",
            ],
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    uf_parent = np.arange(
        n_vertices,
        dtype="int64",
    )

    component_node = np.arange(
        n_vertices,
        dtype="int64",
    )

    component_size = np.ones(
        n_vertices,
        dtype="int64",
    )

    def find(
        vertex,
    ):
        root = vertex

        while (
            uf_parent[root]
            != root
        ):
            root = (
                uf_parent[root]
            )

        while (
            uf_parent[vertex]
            != vertex
        ):
            next_vertex = (
                uf_parent[
                    vertex
                ]
            )

            uf_parent[
                vertex
            ] = root

            vertex = (
                next_vertex
            )

        return root

    next_cluster_id = (
        n_vertices
    )

    hierarchy_rows = []

    for edge in (
        edges.itertuples(
            index=False
        )
    ):

        source = int(
            edge.source
        )

        target = int(
            edge.target
        )

        distance = float(
            edge.weight_km
        )

        root_a = find(
            source
        )

        root_b = find(
            target
        )

        if root_a == root_b:
            raise RuntimeError(
                "MST attempted to merge an already "
                "connected hierarchy component."
            )

        child_a = int(
            component_node[
                root_a
            ]
        )

        child_b = int(
            component_node[
                root_b
            ]
        )

        size_a = int(
            component_size[
                root_a
            ]
        )

        size_b = int(
            component_size[
                root_b
            ]
        )

        parent_cluster = (
            next_cluster_id
        )

        parent_size = (
            size_a
            + size_b
        )

        if distance > 0:
            lambda_value = (
                1.0
                / distance
            )
        else:
            lambda_value = (
                np.inf
            )

        hierarchy_rows.append(
            {
                "parent": (
                    parent_cluster
                ),

                "left_child": (
                    child_a
                ),

                "right_child": (
                    child_b
                ),

                "distance_km": (
                    distance
                ),

                "lambda_value": (
                    lambda_value
                ),

                "left_size": (
                    size_a
                ),

                "right_size": (
                    size_b
                ),

                "cluster_size": (
                    parent_size
                ),
            }
        )

        uf_parent[
            root_b
        ] = root_a

        component_node[
            root_a
        ] = parent_cluster

        component_size[
            root_a
        ] = parent_size

        next_cluster_id += 1

    return pd.DataFrame(
        hierarchy_rows
    )


def validate_single_linkage_hierarchy(
    hierarchy,
    n_vertices,
):
    """
    Validate the binary hierarchy.
    """

    problems = []

    if len(
        hierarchy
    ) != (
        n_vertices - 1
    ):
        problems.append(
            "incorrect_merge_count"
        )

        return problems

    expected_parent_ids = np.arange(
        n_vertices,
        2 * n_vertices - 1,
        dtype="int64",
    )

    actual_parent_ids = (
        hierarchy[
            "parent"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    if not np.array_equal(
        actual_parent_ids,
        expected_parent_ids,
    ):
        problems.append(
            "incorrect_parent_ids"
        )

    distances = (
        hierarchy[
            "distance_km"
        ]
        .to_numpy(
            dtype="float64"
        )
    )

    if np.any(
        np.diff(
            distances
        )
        < -1e-12
    ):
        problems.append(
            "merge_distances_not_monotonic"
        )

    if (
        int(
            hierarchy[
                "cluster_size"
            ].iloc[-1]
        )
        != n_vertices
    ):
        problems.append(
            "incorrect_root_size"
        )

    children = np.concatenate(
        [
            hierarchy[
                "left_child"
            ].to_numpy(
                dtype="int64"
            ),

            hierarchy[
                "right_child"
            ].to_numpy(
                dtype="int64"
            ),
        ]
    )

    unique_children, counts = np.unique(
        children,
        return_counts=True,
    )

    if np.any(
        counts != 1
    ):
        problems.append(
            "child_used_multiple_times"
        )

    expected_children = np.arange(
        0,
        2 * n_vertices - 2,
        dtype="int64",
    )

    if not np.array_equal(
        unique_children,
        expected_children,
    ):
        problems.append(
            "missing_or_invalid_child_nodes"
        )

    return problems


# 5. Condensed hierarchy

def build_condensed_tree(
    hierarchy,
    n_vertices,
    min_cluster_size,
):
    """
    Condense the single-linkage hierarchy according to
    min_cluster_size.

    Cluster 0 is the root condensed cluster.
    """

    if min_cluster_size < 2:
        raise ValueError(
            "min_cluster_size must be at least 2."
        )

    total_nodes = (
        2 * n_vertices - 1
    )

    left_child = np.full(
        total_nodes,
        -1,
        dtype="int64",
    )

    right_child = np.full(
        total_nodes,
        -1,
        dtype="int64",
    )

    node_size = np.ones(
        total_nodes,
        dtype="int64",
    )

    split_lambda = np.full(
        total_nodes,
        np.nan,
        dtype="float64",
    )

    for row in (
        hierarchy.itertuples(
            index=False
        )
    ):

        parent = int(
            row.parent
        )

        left_child[
            parent
        ] = int(
            row.left_child
        )

        right_child[
            parent
        ] = int(
            row.right_child
        )

        node_size[
            parent
        ] = int(
            row.cluster_size
        )

        split_lambda[
            parent
        ] = float(
            row.lambda_value
        )

    root_node = (
        2 * n_vertices - 2
    )

    def collect_leaves(
        start_node,
    ):
        """
        Collect original trajectory IDs beneath a hierarchy node.
        """

        leaves = []

        stack = [
            int(
                start_node
            )
        ]

        while stack:

            node = (
                stack.pop()
            )

            if node < n_vertices:
                leaves.append(
                    node
                )

            else:
                stack.append(
                    int(
                        right_child[
                            node
                        ]
                    )
                )

                stack.append(
                    int(
                        left_child[
                            node
                        ]
                    )
                )

        return leaves

    events = []

    cluster_records = {
        0: {
            "cluster_id": 0,
            "birth_lambda": 0.0,
            "death_lambda": np.nan,
            "birth_size": n_vertices,
            "hierarchy_node": root_node,
        }
    }

    next_cluster_id = 1

    traversal_stack = [
        (
            root_node,
            0,
        )
    ]

    while traversal_stack:

        (
            node,
            current_cluster,
        ) = traversal_stack.pop()

        if node < n_vertices:
            raise RuntimeError(
                "A surviving condensed cluster "
                "unexpectedly reached one leaf."
            )

        left = int(
            left_child[
                node
            ]
        )

        right = int(
            right_child[
                node
            ]
        )

        lambda_value = float(
            split_lambda[
                node
            ]
        )

        left_size = int(
            node_size[
                left
            ]
        )

        right_size = int(
            node_size[
                right
            ]
        )

        left_survives = (
            left_size
            >= min_cluster_size
        )

        right_survives = (
            right_size
            >= min_cluster_size
        )

        # ---------------------------------------------------------
        # Both branches survive:
        # parent ends and two child clusters are born.
        # ---------------------------------------------------------

        if (
            left_survives
            and right_survives
        ):

            left_cluster = (
                next_cluster_id
            )

            next_cluster_id += 1

            right_cluster = (
                next_cluster_id
            )

            next_cluster_id += 1

            for (
                child_cluster,
                child_node,
                child_size,
            ) in [
                (
                    left_cluster,
                    left,
                    left_size,
                ),
                (
                    right_cluster,
                    right,
                    right_size,
                ),
            ]:

                events.append(
                    {
                        "parent_cluster": (
                            current_cluster
                        ),

                        "child_type": (
                            "cluster"
                        ),

                        "child_id": (
                            child_cluster
                        ),

                        "child_hierarchy_node": (
                            child_node
                        ),

                        "lambda_value": (
                            lambda_value
                        ),

                        "child_size": (
                            child_size
                        ),
                    }
                )

                cluster_records[
                    child_cluster
                ] = {
                    "cluster_id": (
                        child_cluster
                    ),

                    "birth_lambda": (
                        lambda_value
                    ),

                    "death_lambda": (
                        np.nan
                    ),

                    "birth_size": (
                        child_size
                    ),

                    "hierarchy_node": (
                        child_node
                    ),
                }

            cluster_records[
                current_cluster
            ][
                "death_lambda"
            ] = lambda_value

            traversal_stack.append(
                (
                    right,
                    right_cluster,
                )
            )

            traversal_stack.append(
                (
                    left,
                    left_cluster,
                )
            )

        # ---------------------------------------------------------
        # Only left survives.
        # ---------------------------------------------------------

        elif (
            left_survives
            and not right_survives
        ):

            for point in (
                collect_leaves(
                    right
                )
            ):

                events.append(
                    {
                        "parent_cluster": (
                            current_cluster
                        ),

                        "child_type": (
                            "point"
                        ),

                        "child_id": (
                            point
                        ),

                        "child_hierarchy_node": (
                            point
                        ),

                        "lambda_value": (
                            lambda_value
                        ),

                        "child_size": 1,
                    }
                )

            traversal_stack.append(
                (
                    left,
                    current_cluster,
                )
            )

        # ---------------------------------------------------------
        # Only right survives.
        # ---------------------------------------------------------

        elif (
            right_survives
            and not left_survives
        ):

            for point in (
                collect_leaves(
                    left
                )
            ):

                events.append(
                    {
                        "parent_cluster": (
                            current_cluster
                        ),

                        "child_type": (
                            "point"
                        ),

                        "child_id": (
                            point
                        ),

                        "child_hierarchy_node": (
                            point
                        ),

                        "lambda_value": (
                            lambda_value
                        ),

                        "child_size": 1,
                    }
                )

            traversal_stack.append(
                (
                    right,
                    current_cluster,
                )
            )

        # ---------------------------------------------------------
        # Neither branch survives.
        # ---------------------------------------------------------

        else:

            fallen_points = (
                collect_leaves(
                    left
                )
                + collect_leaves(
                    right
                )
            )

            for point in (
                fallen_points
            ):

                events.append(
                    {
                        "parent_cluster": (
                            current_cluster
                        ),

                        "child_type": (
                            "point"
                        ),

                        "child_id": (
                            point
                        ),

                        "child_hierarchy_node": (
                            point
                        ),

                        "lambda_value": (
                            lambda_value
                        ),

                        "child_size": 1,
                    }
                )

            cluster_records[
                current_cluster
            ][
                "death_lambda"
            ] = lambda_value

    condensed_tree = pd.DataFrame(
        events
    )

    cluster_table = (
        pd.DataFrame(
            list(
                cluster_records.values()
            )
        )
        .sort_values(
            "cluster_id"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        condensed_tree,
        cluster_table,
    )


def validate_condensed_tree(
    condensed_tree,
    cluster_table,
    n_vertices,
    min_cluster_size,
):
    """
    Validate the condensed hierarchy.
    """

    problems = []

    point_rows = (
        condensed_tree[
            condensed_tree[
                "child_type"
            ]
            == "point"
        ]
    )

    cluster_rows = (
        condensed_tree[
            condensed_tree[
                "child_type"
            ]
            == "cluster"
        ]
    )

    point_ids = (
        point_rows[
            "child_id"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    if len(
        point_ids
    ) != n_vertices:
        problems.append(
            "incorrect_point_fallout_count"
        )

    unique_points, counts = np.unique(
        point_ids,
        return_counts=True,
    )

    if not np.array_equal(
        unique_points,
        np.arange(
            n_vertices
        ),
    ):
        problems.append(
            "missing_or_invalid_points"
        )

    if np.any(
        counts != 1
    ):
        problems.append(
            "point_falls_out_multiple_times"
        )

    if len(
        cluster_rows
    ) != (
        len(
            cluster_table
        )
        - 1
    ):
        problems.append(
            "incorrect_cluster_parent_count"
        )

    if (
        len(
            cluster_rows
        ) > 0
        and np.any(
            cluster_rows[
                "child_size"
            ]
            < min_cluster_size
        )
    ):
        problems.append(
            "undersized_condensed_cluster"
        )

    expected_cluster_ids = np.arange(
        len(
            cluster_table
        ),
        dtype="int64",
    )

    actual_cluster_ids = (
        cluster_table[
            "cluster_id"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    if not np.array_equal(
        expected_cluster_ids,
        actual_cluster_ids,
    ):
        problems.append(
            "nonsequential_cluster_ids"
        )

    if (
        cluster_table[
            "death_lambda"
        ]
        .isna()
        .any()
    ):
        problems.append(
            "cluster_missing_death_lambda"
        )

    return problems


# 6. Cluster stability

def compute_cluster_stability(
    condensed_tree,
    cluster_table,
):
    """
    Compute HDBSCAN cluster stability.

    Each departure event contributes:

        child_size *
        (lambda_event - lambda_birth(parent))
    """

    result = (
        cluster_table
        .copy()
    )

    birth_lambda = (
        result
        .set_index(
            "cluster_id"
        )[
            "birth_lambda"
        ]
        .to_dict()
    )

    events = (
        condensed_tree
        .copy()
    )

    events[
        "parent_birth_lambda"
    ] = (
        events[
            "parent_cluster"
        ]
        .map(
            birth_lambda
        )
    )

    if (
        events[
            "parent_birth_lambda"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Unknown parent cluster in condensed tree."
        )

    events[
        "stability_contribution"
    ] = (
        events[
            "child_size"
        ]
        * (
            events[
                "lambda_value"
            ]
            - events[
                "parent_birth_lambda"
            ]
        )
    )

    tiny_negative = (
        (
            events[
                "stability_contribution"
            ]
            < 0
        )
        & (
            events[
                "stability_contribution"
            ]
            > -1e-12
        )
    )

    events.loc[
        tiny_negative,
        "stability_contribution",
    ] = 0.0

    if np.any(
        events[
            "stability_contribution"
        ]
        < 0
    ):
        raise RuntimeError(
            "Negative stability contribution found."
        )

    stability = (
        events
        .groupby(
            "parent_cluster"
        )[
            "stability_contribution"
        ]
        .sum()
    )

    result[
        "stability"
    ] = (
        result[
            "cluster_id"
        ]
        .map(
            stability
        )
        .fillna(
            0.0
        )
    )

    return result


# 7. Excess-of-Mass cluster selection

def select_clusters_eom(
    condensed_tree,
    cluster_table,
    allow_single_cluster=False,
):
    """
    Select clusters using the Excess-of-Mass rule.
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

    cluster_edges = (
        condensed_tree[
            condensed_tree[
                "child_type"
            ]
            == "cluster"
        ]
    )

    children = {
        int(
            cluster_id
        ): []
        for cluster_id
        in cluster_table[
            "cluster_id"
        ]
    }

    for row in (
        cluster_edges.itertuples(
            index=False
        )
    ):
        children[
            int(
                row.parent_cluster
            )
        ].append(
            int(
                row.child_id
            )
        )

    def solve(
        cluster_id,
    ):

        child_clusters = (
            children[
                cluster_id
            ]
        )

        if not child_clusters:

            if (
                cluster_id == 0
                and not allow_single_cluster
            ):
                return (
                    set(),
                    0.0,
                )

            return (
                {
                    cluster_id
                },
                float(
                    stability[
                        cluster_id
                    ]
                ),
            )

        selected_descendants = set()

        descendant_stability = 0.0

        for child in (
            child_clusters
        ):

            (
                selected,
                effective_stability,
            ) = solve(
                child
            )

            selected_descendants.update(
                selected
            )

            descendant_stability += (
                effective_stability
            )

        if (
            cluster_id == 0
            and not allow_single_cluster
        ):
            return (
                selected_descendants,
                descendant_stability,
            )

        own_stability = float(
            stability[
                cluster_id
            ]
        )

        if (
            own_stability
            >= descendant_stability
        ):
            return (
                {
                    cluster_id
                },
                own_stability,
            )

        return (
            selected_descendants,
            descendant_stability,
        )

    return solve(
        0
    )


# 8. Final labels

def assign_selected_cluster_labels(
    condensed_tree,
    selected_clusters,
    n_vertices,
):
    """
    Assign selected HDBSCAN clusters to original trajectories.

    Noise receives label -1.
    """

    cluster_edges = (
        condensed_tree[
            condensed_tree[
                "child_type"
            ]
            == "cluster"
        ]
    )

    point_edges = (
        condensed_tree[
            condensed_tree[
                "child_type"
            ]
            == "point"
        ]
    )

    parent_of_cluster = {}

    for row in (
        cluster_edges.itertuples(
            index=False
        )
    ):
        parent_of_cluster[
            int(
                row.child_id
            )
        ] = int(
            row.parent_cluster
        )

    selected_clusters = {
        int(
            cluster_id
        )
        for cluster_id
        in selected_clusters
    }

    ordered_selected = sorted(
        selected_clusters
    )

    cluster_to_label = {
        cluster_id: label
        for (
            label,
            cluster_id,
        )
        in enumerate(
            ordered_selected
        )
    }

    labels = np.full(
        n_vertices,
        -1,
        dtype="int64",
    )

    selected_cluster_ids = np.full(
        n_vertices,
        -1,
        dtype="int64",
    )

    for row in (
        point_edges.itertuples(
            index=False
        )
    ):

        point = int(
            row.child_id
        )

        current_cluster = int(
            row.parent_cluster
        )

        assigned_cluster = None

        while True:

            if (
                current_cluster
                in selected_clusters
            ):
                assigned_cluster = (
                    current_cluster
                )

                break

            if (
                current_cluster
                not in parent_of_cluster
            ):
                break

            current_cluster = (
                parent_of_cluster[
                    current_cluster
                ]
            )

        if assigned_cluster is not None:

            labels[
                point
            ] = cluster_to_label[
                assigned_cluster
            ]

            selected_cluster_ids[
                point
            ] = assigned_cluster

    return (
        labels,
        selected_cluster_ids,
        cluster_to_label,
    )


def validate_final_labels(
    labels,
    selected_clusters,
    n_vertices,
):
    """
    Validate final selected-cluster labels.
    """

    problems = []

    if len(
        labels
    ) != n_vertices:
        problems.append(
            "incorrect_label_count"
        )

        return problems

    clustered = (
        labels[
            labels >= 0
        ]
    )

    expected_cluster_count = (
        len(
            selected_clusters
        )
    )

    if expected_cluster_count == 0:

        if len(
            clustered
        ) > 0:
            problems.append(
                "labels_without_selected_clusters"
            )

        return problems

    unique_labels = np.unique(
        clustered
    )

    expected_labels = np.arange(
        expected_cluster_count
    )

    if not np.array_equal(
        unique_labels,
        expected_labels,
    ):
        problems.append(
            "nonsequential_cluster_labels"
        )

    return problems


# Complete manual HDBSCAN pipeline

def fit_hdbscan_precomputed(
    distance_matrix,
    min_samples,
    min_cluster_size,
    allow_single_cluster=False,
):
    """
    Run the complete manual HDBSCAN pipeline on a
    precomputed distance matrix.

    Returns
    -------
    dict
        Contains all major intermediate and final artifacts.
    """

    distance_problems = (
        validate_distance_matrix(
            distance_matrix
        )
    )

    if distance_problems:
        raise RuntimeError(
            "Distance-matrix validation failed: "
            + ", ".join(
                distance_problems
            )
        )

    n = distance_matrix.shape[
        0
    ]

    # Core distances

    core_distances = (
        compute_core_distances(
            distance_matrix,
            min_samples,
        )
    )

    # Mutual reachability

    mutual_reachability = (
        compute_mutual_reachability(
            distance_matrix,
            core_distances,
        )
    )

    mutual_problems = (
        validate_mutual_reachability(
            distance_matrix,
            mutual_reachability,
            core_distances,
        )
    )

    if mutual_problems:
        raise RuntimeError(
            "Mutual-reachability validation failed: "
            + ", ".join(
                mutual_problems
            )
        )

    # MST

    mst = compute_mst_prim(
        mutual_reachability
    )

    mst_problems = (
        validate_mst(
            mst,
            n,
        )
    )

    if mst_problems:
        raise RuntimeError(
            "MST validation failed: "
            + ", ".join(
                mst_problems
            )
        )

    # Single-linkage hierarchy

    hierarchy = (
        build_single_linkage_hierarchy(
            mst,
            n,
        )
    )

    hierarchy_problems = (
        validate_single_linkage_hierarchy(
            hierarchy,
            n,
        )
    )

    if hierarchy_problems:
        raise RuntimeError(
            "Hierarchy validation failed: "
            + ", ".join(
                hierarchy_problems
            )
        )

    # Condensed hierarchy

    (
        condensed_tree,
        cluster_table,
    ) = build_condensed_tree(
        hierarchy,
        n,
        min_cluster_size,
    )

    condensed_problems = (
        validate_condensed_tree(
            condensed_tree,
            cluster_table,
            n,
            min_cluster_size,
        )
    )

    if condensed_problems:
        raise RuntimeError(
            "Condensed-tree validation failed: "
            + ", ".join(
                condensed_problems
            )
        )

    # Stability

    cluster_table = (
        compute_cluster_stability(
            condensed_tree,
            cluster_table,
        )
    )

    # EOM selection

    (
        selected_clusters,
        selected_stability,
    ) = select_clusters_eom(
        condensed_tree,
        cluster_table,
        allow_single_cluster=(
            allow_single_cluster
        ),
    )

    # Labels

    (
        labels,
        selected_cluster_ids,
        cluster_to_label,
    ) = assign_selected_cluster_labels(
        condensed_tree,
        selected_clusters,
        n,
    )

    label_problems = (
        validate_final_labels(
            labels,
            selected_clusters,
            n,
        )
    )

    if label_problems:
        raise RuntimeError(
            "Final-label validation failed: "
            + ", ".join(
                label_problems
            )
        )

    cluster_table = (
        cluster_table
        .copy()
    )

    cluster_table[
        "selected"
    ] = (
        cluster_table[
            "cluster_id"
        ]
        .isin(
            selected_clusters
        )
    )

    return {
        "core_distances": (
            core_distances
        ),

        "mutual_reachability": (
            mutual_reachability
        ),

        "mst": (
            mst
        ),

        "single_linkage": (
            hierarchy
        ),

        "condensed_tree": (
            condensed_tree
        ),

        "cluster_table": (
            cluster_table
        ),

        "selected_clusters": (
            selected_clusters
        ),

        "selected_stability": (
            selected_stability
        ),

        "labels": (
            labels
        ),

        "selected_cluster_ids": (
            selected_cluster_ids
        ),

        "cluster_to_label": (
            cluster_to_label
        ),
    }