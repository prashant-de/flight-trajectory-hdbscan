import numpy as np
import pandas as pd

from src.config import (
    DISTANCE_OUTPUT_DIR,
    HDBSCAN_OUTPUT_DIR,
)


# Initial HDBSCAN sensitivity values

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


# Data loading

def load_distance_data():
    """
    Load the precomputed Hausdorff distance matrix and the
    synchronized trajectory index.
    """

    matrix_path = (
        DISTANCE_OUTPUT_DIR
        / "hausdorff_100.npy"
    )

    index_path = (
        DISTANCE_OUTPUT_DIR
        / "trajectory_index.csv"
    )

    distance_matrix = np.load(
        matrix_path
    )

    trajectory_index = pd.read_csv(
        index_path
    )

    return (
        distance_matrix,
        trajectory_index,
    )


# Validation

def validate_distance_data(
    distance_matrix,
    trajectory_index,
):
    """
    Verify that the matrix and trajectory index are synchronized.
    """

    problems = []

    if (
        distance_matrix.ndim != 2
        or distance_matrix.shape[0]
        != distance_matrix.shape[1]
    ):
        problems.append(
            "distance_matrix_not_square"
        )

        return problems

    n = distance_matrix.shape[0]

    if len(
        trajectory_index
    ) != n:
        problems.append(
            "trajectory_index_length_mismatch"
        )

    expected_indices = np.arange(
        n
    )

    actual_indices = (
        trajectory_index[
            "matrix_index"
        ]
        .to_numpy()
    )

    if not np.array_equal(
        expected_indices,
        actual_indices,
    ):
        problems.append(
            "matrix_index_not_sequential"
        )

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

    return problems


# Core distance

def compute_core_distances(
    distance_matrix,
    min_samples,
):
    """
    Compute HDBSCAN core distance for every trajectory.

    For one trajectory, the sorted row looks like:

        0,
        nearest neighbor,
        second-nearest neighbor,
        ...

    The zero is the trajectory's distance to itself.

    Therefore for min_samples = k, the core distance is
    sorted_distances[k], which corresponds to the k-th
    nearest OTHER trajectory.
    """

    n = distance_matrix.shape[0]

    if min_samples < 1:
        raise ValueError(
            "min_samples must be at least 1."
        )

    if min_samples >= n:
        raise ValueError(
            "min_samples must be smaller than "
            "the number of trajectories."
        )

    # np.partition is more efficient than sorting every
    # complete row because we only need one neighbor rank.
    partitioned = np.partition(
        distance_matrix,
        kth=min_samples,
        axis=1,
    )

    core_distances = (
        partitioned[
            :,
            min_samples
        ]
    )

    return core_distances

def compute_mutual_reachability(
    distance_matrix,
    core_distances,
):
    """
    Compute the complete mutual-reachability matrix.

    For trajectories a and b:

        mreach(a, b) =
            max(
                core(a),
                core(b),
                distance(a, b)
            )

    The diagonal is left according to the mathematical
    definition. It is not used as an edge during MST
    construction.
    """

    if (
        distance_matrix.shape[0]
        != len(core_distances)
    ):
        raise ValueError(
            "Core-distance length does not match "
            "distance-matrix size."
        )

    mutual_reachability = np.maximum(
        distance_matrix,
        core_distances[:, None],
    )

    mutual_reachability = np.maximum(
        mutual_reachability,
        core_distances[None, :],
    )

    return mutual_reachability

def validate_mutual_reachability(
    distance_matrix,
    mutual_reachability,
    core_distances,
):
    """
    Validate properties required before constructing the MST.
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

    # Mutual reachability can never be smaller than
    # the original trajectory distance.
    if np.any(
        mutual_reachability
        + 1e-12
        < distance_matrix
    ):
        problems.append(
            "smaller_than_raw_distance"
        )

    expected_diagonal = (
        core_distances
    )

    if not np.allclose(
        np.diag(
            mutual_reachability
        ),
        expected_diagonal,
        atol=1e-12,
    ):
        problems.append(
            "incorrect_diagonal"
        )

    return problems

def compute_mst_prim(
    mutual_reachability,
):
    """
    Construct a minimum spanning tree using Prim's algorithm.

    The input is the complete dense mutual-reachability matrix.

    Returns a DataFrame with exactly N - 1 edges:
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

    # Whether each trajectory has already entered the tree.
    in_tree = np.zeros(
        n,
        dtype=bool,
    )

    # Cheapest known edge connecting each outside vertex
    # to the current tree.
    best_weight = np.full(
        n,
        np.inf,
        dtype="float64",
    )

    # Parent vertex providing that cheapest connection.
    parent = np.full(
        n,
        -1,
        dtype="int64",
    )

    # Start from trajectory 0.
    best_weight[0] = 0.0

    edges = []

    for _ in range(n):

        # Select the not-yet-added trajectory with the
        # smallest connection to the current tree.
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
                "Graph is disconnected; MST cannot be completed."
            )

        in_tree[u] = True

        # The first vertex has no parent.
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

        # Update the cheapest known connection for every
        # trajectory still outside the tree.
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
    Validate basic MST structural properties.
    """

    problems = []

    expected_edges = (
        n_vertices - 1
    )

    if len(
        mst_edges
    ) != expected_edges:
        problems.append(
            "incorrect_edge_count"
        )

    if len(
        mst_edges
    ) == 0:
        problems.append(
            "empty_mst"
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

    # Connectivity validation with a small manual union-find.

    parent = np.arange(
        n_vertices,
        dtype="int64",
    )

    def find(x):
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

    def union(a, b):
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

    cycle_detected = False

    for row in (
        mst_edges.itertuples(
            index=False
        )
    ):
        merged = union(
            int(
                row.source
            ),
            int(
                row.target
            ),
        )

        if not merged:
            cycle_detected = True

    if cycle_detected:
        problems.append(
            "cycle_detected"
        )

    roots = {
        find(
            vertex
        )
        for vertex in range(
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

def mst_edge_set(
    mst_edges,
):
    """
    Return MST edges as unordered vertex pairs.

    Used to compare tree structure across min_samples settings.
    """

    edges = set()

    for row in (
        mst_edges.itertuples(
            index=False
        )
    ):
        a = int(
            row.source
        )

        b = int(
            row.target
        )

        edges.add(
            (
                min(
                    a,
                    b,
                ),
                max(
                    a,
                    b,
                ),
            )
        )

    return edges

def build_single_linkage_hierarchy(
    mst_edges,
    n_vertices,
):
    """
    Build a binary single-linkage hierarchy from an MST.

    Leaf node IDs:
        0 ... n_vertices - 1

    Internal cluster IDs:
        n_vertices ... 2*n_vertices - 2

    Each MST edge merges two currently separate components.
    Edges are processed from smallest to largest weight.
    """

    if len(mst_edges) != n_vertices - 1:
        raise ValueError(
            "An MST must contain exactly N - 1 edges."
        )

    # Deterministic ordering is useful when several MST
    # edges have exactly the same weight.
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

    # Union-find state

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

    def find(vertex):
        """
        Find the current union-find component root.
        """

        root = vertex

        while (
            uf_parent[root]
            != root
        ):
            root = uf_parent[root]

        # Path compression.
        while (
            uf_parent[vertex]
            != vertex
        ):
            next_vertex = (
                uf_parent[vertex]
            )

            uf_parent[vertex] = root

            vertex = next_vertex

        return root

    next_cluster_id = (
        n_vertices
    )

    hierarchy_rows = []

    # Merge components in increasing MST-edge order

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

        distance_km = float(
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
                "MST edge joined vertices that were already "
                "inside the same hierarchy component."
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

        if distance_km > 0:
            lambda_value = (
                1.0
                / distance_km
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
                    distance_km
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

        # Union the two graph components.
        # We can attach root_b to root_a because union-find shape
        # is unrelated to the actual clustering hierarchy.

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
    Validate the manually constructed single-linkage tree.
    """

    problems = []

    expected_merges = (
        n_vertices - 1
    )

    if len(
        hierarchy
    ) != expected_merges:
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

    cluster_sizes = (
        hierarchy[
            "cluster_size"
        ]
        .to_numpy(
            dtype="int64"
        )
    )

    if np.any(
        cluster_sizes < 2
    ):
        problems.append(
            "invalid_cluster_size"
        )

    if (
        cluster_sizes[-1]
        != n_vertices
    ):
        problems.append(
            "final_cluster_does_not_contain_all_points"
        )

    # Every node except the final root should appear exactly
    # once as a child in the binary hierarchy.
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

    lambda_values = (
        hierarchy[
            "lambda_value"
        ]
        .to_numpy(
            dtype="float64"
        )
    )

    finite_lambda = lambda_values[
        np.isfinite(
            lambda_values
        )
    ]

    if (
        len(
            finite_lambda
        ) > 1
        and np.any(
            np.diff(
                finite_lambda
            )
            > 1e-12
        )
    ):
        problems.append(
            "lambda_not_nonincreasing"
        )

    return problems

def build_condensed_tree(
    hierarchy,
    n_vertices,
    min_cluster_size,
):
    """
    Condense a binary single-linkage hierarchy using the
    HDBSCAN min_cluster_size rule.

    Condensed cluster IDs are separate from the original
    hierarchy node IDs.

    Cluster 0 is the root condensed cluster.

    Events record either:

        parent cluster -> child cluster

    or:

        parent cluster -> individual point

    at the lambda value where that transition occurs.
    """

    if min_cluster_size < 2:
        raise ValueError(
            "min_cluster_size must be at least 2."
        )

    total_nodes = (
        2 * n_vertices - 1
    )

    # Store the original binary hierarchy in arrays.

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

    for row in hierarchy.itertuples(
        index=False
    ):
        parent = int(
            row.parent
        )

        left_child[parent] = int(
            row.left_child
        )

        right_child[parent] = int(
            row.right_child
        )

        node_size[parent] = int(
            row.cluster_size
        )

        split_lambda[parent] = float(
            row.lambda_value
        )

    root_node = (
        2 * n_vertices - 2
    )

    # Helper: collect all original trajectory leaves under a
    # hierarchy node without recursion.
    #
    # This avoids Python recursion-depth problems for highly
    # unbalanced trees.

    def collect_leaves(
        start_node,
    ):
        leaves = []

        stack = [
            int(
                start_node
            )
        ]

        while stack:
            node = stack.pop()

            if node < n_vertices:
                leaves.append(
                    node
                )

            else:
                stack.append(
                    int(
                        right_child[node]
                    )
                )

                stack.append(
                    int(
                        left_child[node]
                    )
                )

        return leaves

    # Condensed tree state

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

    # Explicit stack avoids recursion-depth problems.
    #
    # Each item:
    #   original hierarchy node,
    #   current condensed cluster ID
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
                "A surviving condensed cluster unexpectedly "
                "reached a single leaf."
            )

        left = int(
            left_child[node]
        )

        right = int(
            right_child[node]
        )

        lambda_value = float(
            split_lambda[node]
        )

        left_size = int(
            node_size[left]
        )

        right_size = int(
            node_size[right]
        )

        left_survives = (
            left_size
            >= min_cluster_size
        )

        right_survives = (
            right_size
            >= min_cluster_size
        )

        # Case 1:
        # Both children are large enough.
        #
        # The parent condensed cluster dies and two new
        # candidate clusters are born.

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

            events.append(
                {
                    "parent_cluster": (
                        current_cluster
                    ),

                    "child_type": (
                        "cluster"
                    ),

                    "child_id": (
                        left_cluster
                    ),

                    "child_hierarchy_node": (
                        left
                    ),

                    "lambda_value": (
                        lambda_value
                    ),

                    "child_size": (
                        left_size
                    ),
                }
            )

            events.append(
                {
                    "parent_cluster": (
                        current_cluster
                    ),

                    "child_type": (
                        "cluster"
                    ),

                    "child_id": (
                        right_cluster
                    ),

                    "child_hierarchy_node": (
                        right
                    ),

                    "lambda_value": (
                        lambda_value
                    ),

                    "child_size": (
                        right_size
                    ),
                }
            )

            cluster_records[
                current_cluster
            ][
                "death_lambda"
            ] = lambda_value

            cluster_records[
                left_cluster
            ] = {
                "cluster_id": (
                    left_cluster
                ),

                "birth_lambda": (
                    lambda_value
                ),

                "death_lambda": (
                    np.nan
                ),

                "birth_size": (
                    left_size
                ),

                "hierarchy_node": (
                    left
                ),
            }

            cluster_records[
                right_cluster
            ] = {
                "cluster_id": (
                    right_cluster
                ),

                "birth_lambda": (
                    lambda_value
                ),

                "death_lambda": (
                    np.nan
                ),

                "birth_size": (
                    right_size
                ),

                "hierarchy_node": (
                    right
                ),
            }

            # Push right first so left is processed first.
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

        # Case 2:
        # Left survives, right is too small.
        #
        # Right-side trajectories fall out.
        # Left continues as the SAME condensed cluster.

        elif (
            left_survives
            and not right_survives
        ):
            fallen_points = (
                collect_leaves(
                    right
                )
            )

            for point in fallen_points:
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

        # Case 3:
        # Right survives, left is too small.

        elif (
            right_survives
            and not left_survives
        ):
            fallen_points = (
                collect_leaves(
                    left
                )
            )

            for point in fallen_points:
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

        # Case 4:
        # Neither child reaches min_cluster_size.
        #
        # All remaining points leave the current cluster.
        # The cluster dies here.

        else:
            fallen_points = (
                collect_leaves(
                    left
                )
                + collect_leaves(
                    right
                )
            )

            for point in fallen_points:
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

    cluster_table = pd.DataFrame(
        list(
            cluster_records.values()
        )
    ).sort_values(
        "cluster_id"
    ).reset_index(
        drop=True
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
    Validate core structural properties of a condensed tree.
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

    # Every original trajectory should eventually fall out
    # exactly once.
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

    if (
        len(
            unique_points
        )
        != n_vertices
    ):
        problems.append(
            "missing_points"
        )

    elif not np.array_equal(
        unique_points,
        np.arange(
            n_vertices
        ),
    ):
        problems.append(
            "invalid_point_ids"
        )

    if (
        len(
            counts
        ) > 0
        and np.any(
            counts != 1
        )
    ):
        problems.append(
            "point_falls_out_multiple_times"
        )

    # Every non-root condensed cluster must have one parent.
    expected_nonroot_clusters = (
        len(
            cluster_table
        )
        - 1
    )

    if len(
        cluster_rows
    ) != expected_nonroot_clusters:
        problems.append(
            "incorrect_cluster_parent_count"
        )

    if len(
        cluster_rows
    ) > 0:
        if np.any(
            cluster_rows[
                "child_size"
            ]
            < min_cluster_size
        ):
            problems.append(
                "undersized_condensed_cluster"
            )

    # Cluster IDs must be sequential.
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

    # Every condensed cluster must have a valid death lambda.
    if cluster_table[
        "death_lambda"
    ].isna().any():
        problems.append(
            "cluster_missing_death_lambda"
        )

    if np.any(
        cluster_table[
            "death_lambda"
        ].to_numpy()
        + 1e-12
        <
        cluster_table[
            "birth_lambda"
        ].to_numpy()
    ):
        problems.append(
            "death_before_birth"
        )

    return problems

def compute_cluster_stability(
    condensed_tree,
    cluster_table,
):
    """
    Compute HDBSCAN stability for every condensed cluster.

    For each event leaving parent cluster C:

        contribution =
            child_size *
            (lambda_event - lambda_birth(C))

    This is equivalent to summing each trajectory's
    lifetime inside the cluster.
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

    event_table = (
        condensed_tree
        .copy()
    )

    event_table[
        "parent_birth_lambda"
    ] = (
        event_table[
            "parent_cluster"
        ]
        .map(
            birth_lambda
        )
    )

    if event_table[
        "parent_birth_lambda"
    ].isna().any():
        raise RuntimeError(
            "A condensed-tree event references "
            "an unknown parent cluster."
        )

    event_table[
        "stability_contribution"
    ] = (
        event_table[
            "child_size"
        ]
        * (
            event_table[
                "lambda_value"
            ]
            - event_table[
                "parent_birth_lambda"
            ]
        )
    )

    # Tiny negative floating-point differences should not
    # contribute negative stability.
    tiny_negative = (
        (
            event_table[
                "stability_contribution"
            ]
            < 0
        )
        & (
            event_table[
                "stability_contribution"
            ]
            > -1e-12
        )
    )

    event_table.loc[
        tiny_negative,
        "stability_contribution",
    ] = 0.0

    if np.any(
        event_table[
            "stability_contribution"
        ]
        < 0
    ):
        raise RuntimeError(
            "Negative cluster stability contribution found."
        )

    stability_by_cluster = (
        event_table
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
            stability_by_cluster
        )
        .fillna(
            0.0
        )
    )

    return result

def select_clusters_eom(
    condensed_tree,
    cluster_table,
    allow_single_cluster=False,
):
    """
    Select final clusters using the HDBSCAN
    Excess-of-Mass (EOM) rule.

    A candidate cluster is selected when its own
    stability is greater than or equal to the
    effective stability of its selected descendants.

    By default the root cluster is never selected.
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
        for cluster_id in (
            cluster_table[
                "cluster_id"
            ]
        )
    }

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

    def solve(
        cluster_id,
    ):
        """
        Return:
            selected cluster IDs in this subtree,
            effective EOM stability of the subtree
        """

        child_clusters = (
            children[
                cluster_id
            ]
        )

        # Leaf candidate cluster.
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

        descendant_selected = set()

        descendant_stability = 0.0

        for child in (
            child_clusters
        ):
            (
                child_selected,
                child_stability,
            ) = solve(
                child
            )

            descendant_selected.update(
                child_selected
            )

            descendant_stability += (
                child_stability
            )

        # Root is excluded unless explicitly requested.
        if (
            cluster_id == 0
            and not allow_single_cluster
        ):
            return (
                descendant_selected,
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
            descendant_selected,
            descendant_stability,
        )

    selected_clusters, selected_stability = (
        solve(
            0
        )
    )

    return (
        selected_clusters,
        selected_stability,
    )

def assign_selected_cluster_labels(
    condensed_tree,
    selected_clusters,
    n_vertices,
):
    """
    Assign every trajectory either to an EOM-selected
    cluster or to noise (-1).

    A trajectory belongs to a selected cluster when the
    condensed cluster from which it falls out is that
    selected cluster or one of its descendants.
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
        cluster_edges
        .itertuples(
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

    selected_clusters = set(
        int(
            cluster_id
        )
        for cluster_id in (
            selected_clusters
        )
    )

    # Stable human-readable labels 0,1,2,...
    ordered_selected = sorted(
        selected_clusters
    )

    cluster_to_label = {
        cluster_id: label
        for label, cluster_id
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
        point_edges
        .itertuples(
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

        if (
            assigned_cluster
            is not None
        ):
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
    Validate the final EOM labeling result.
    """

    problems = []

    if len(
        labels
    ) != n_vertices:
        problems.append(
            "incorrect_label_count"
        )

        return problems

    valid_labels = (
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
            valid_labels
        ) != 0:
            problems.append(
                "labels_exist_without_selected_clusters"
            )

        return problems

    unique_labels = np.unique(
        valid_labels
    )

    expected_labels = np.arange(
        expected_cluster_count
    )

    if not np.array_equal(
        unique_labels,
        expected_labels,
    ):
        problems.append(
            "nonsequential_final_labels"
        )

    counts = np.bincount(
        valid_labels,
        minlength=expected_cluster_count,
    )

    if np.any(
        counts <= 0
    ):
        problems.append(
            "empty_selected_cluster"
        )

    return problems


def summarize_hierarchy(
    hierarchy,
    min_samples,
):
    """
    Summarize merge levels and tied edge weights.
    """

    distances = (
        hierarchy[
            "distance_km"
        ]
        .to_numpy(
            dtype="float64"
        )
    )

    distance_series = pd.Series(
        distances
    )

    duplicated = (
        distance_series
        .duplicated(
            keep=False
        )
        .to_numpy()
    )

    value_counts = (
        distance_series
        .value_counts()
    )

    return {
        "min_samples": (
            min_samples
        ),

        "merge_count": (
            len(
                hierarchy
            )
        ),

        "minimum_merge_km": (
            np.min(
                distances
            )
        ),

        "median_merge_km": (
            np.median(
                distances
            )
        ),

        "p95_merge_km": (
            np.percentile(
                distances,
                95,
            )
        ),

        "maximum_merge_km": (
            np.max(
                distances
            )
        ),

        "unique_merge_levels": (
            distance_series
            .nunique()
        ),

        "tied_merge_rows": (
            int(
                duplicated.sum()
            )
        ),

        "tied_merge_percent": (
            duplicated.mean()
            * 100
        ),

        "largest_tie_group": (
            int(
                value_counts.max()
            )
        ),

        "root_cluster_size": (
            int(
                hierarchy[
                    "cluster_size"
                ].iloc[-1]
            )
        ),
    }




# Core-distance sensitivity summary
def summarize_core_distances(
    core_distances,
    min_samples,
):
    """
    Return distribution statistics for one min_samples value.
    """

    return {
        "min_samples": (
            min_samples
        ),

        "minimum_km": (
            np.min(
                core_distances
            )
        ),

        "p05_km": (
            np.percentile(
                core_distances,
                5,
            )
        ),

        "p25_km": (
            np.percentile(
                core_distances,
                25,
            )
        ),

        "median_km": (
            np.median(
                core_distances
            )
        ),

        "p75_km": (
            np.percentile(
                core_distances,
                75,
            )
        ),

        "p95_km": (
            np.percentile(
                core_distances,
                95,
            )
        ),

        "maximum_km": (
            np.max(
                core_distances
            )
        ),
    }


# Main
def main():
    (
        distance_matrix,
        trajectory_index,
    ) = load_distance_data()

    problems = validate_distance_data(
        distance_matrix,
        trajectory_index,
    )

    if problems:
        raise RuntimeError(
            "Distance-data validation failed: "
            + ", ".join(
                problems
            )
        )

    n = distance_matrix.shape[
        0
    ]

    HDBSCAN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Trajectories:",
        n,
    )

    print(
        "\nRunning EOM sensitivity across:"
    )

    print(
        "min_samples:",
        MIN_SAMPLES_CANDIDATES,
    )

    print(
        "min_cluster_size:",
        MIN_CLUSTER_SIZE_CANDIDATES,
    )

    summary_records = []

    for min_samples in (
        MIN_SAMPLES_CANDIDATES
    ):

        for min_cluster_size in (
            MIN_CLUSTER_SIZE_CANDIDATES
        ):

            condensed_path = (
                HDBSCAN_OUTPUT_DIR
                / (
                    "condensed_tree_"
                    f"k{min_samples}_"
                    f"mcs{min_cluster_size}.csv"
                )
            )

            cluster_path = (
                HDBSCAN_OUTPUT_DIR
                / (
                    "condensed_clusters_"
                    f"k{min_samples}_"
                    f"mcs{min_cluster_size}.csv"
                )
            )

            condensed_tree = (
                pd.read_csv(
                    condensed_path
                )
            )

            cluster_table = (
                pd.read_csv(
                    cluster_path
                )
            )

            # Stability

            cluster_table = (
                compute_cluster_stability(
                    condensed_tree,
                    cluster_table,
                )
            )

            if not np.all(
                np.isfinite(
                    cluster_table[
                        "stability"
                    ]
                )
            ):
                raise RuntimeError(
                    "Non-finite cluster stability found."
                )

            if np.any(
                cluster_table[
                    "stability"
                ]
                < 0
            ):
                raise RuntimeError(
                    "Negative cluster stability found."
                )

            # EOM

            (
                selected_clusters,
                selected_stability,
            ) = select_clusters_eom(
                condensed_tree,
                cluster_table,
                allow_single_cluster=False,
            )

            (
                labels,
                selected_cluster_ids,
                cluster_to_label,
            ) = assign_selected_cluster_labels(
                condensed_tree,
                selected_clusters,
                n_vertices=n,
            )

            label_validation = (
                validate_final_labels(
                    labels,
                    selected_clusters,
                    n_vertices=n,
                )
            )

            if label_validation:
                raise RuntimeError(
                    "Final-label validation failed "
                    f"for k={min_samples}, "
                    f"mcs={min_cluster_size}: "
                    + ", ".join(
                        label_validation
                    )
                )

            # Final cluster statistics

            clustered_mask = (
                labels >= 0
            )

            noise_count = int(
                (
                    labels == -1
                ).sum()
            )

            noise_percent = (
                noise_count
                / n
                * 100
            )

            if len(
                selected_clusters
            ) > 0:

                cluster_sizes = (
                    np.bincount(
                        labels[
                            clustered_mask
                        ],
                        minlength=len(
                            selected_clusters
                        ),
                    )
                )

                minimum_cluster_size = int(
                    cluster_sizes.min()
                )

                median_cluster_size = float(
                    np.median(
                        cluster_sizes
                    )
                )

                maximum_cluster_size = int(
                    cluster_sizes.max()
                )

            else:
                cluster_sizes = np.array(
                    [],
                    dtype="int64",
                )

                minimum_cluster_size = 0
                median_cluster_size = 0.0
                maximum_cluster_size = 0

            summary_records.append(
                {
                    "min_samples": (
                        min_samples
                    ),

                    "min_cluster_size": (
                        min_cluster_size
                    ),

                    "selected_clusters": (
                        len(
                            selected_clusters
                        )
                    ),

                    "noise_count": (
                        noise_count
                    ),

                    "noise_percent": (
                        noise_percent
                    ),

                    "minimum_final_cluster": (
                        minimum_cluster_size
                    ),

                    "median_final_cluster": (
                        median_cluster_size
                    ),

                    "maximum_final_cluster": (
                        maximum_cluster_size
                    ),

                    "selected_stability": (
                        selected_stability
                    ),
                }
            )

            # Save cluster-selection table

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

            stability_path = (
                HDBSCAN_OUTPUT_DIR
                / (
                    "cluster_stability_"
                    f"k{min_samples}_"
                    f"mcs{min_cluster_size}.csv"
                )
            )

            cluster_table.to_csv(
                stability_path,
                index=False,
            )

            # Save final trajectory labels

            label_table = (
                trajectory_index
                .copy()
            )

            label_table[
                "cluster_label"
            ] = labels

            label_table[
                "selected_cluster_id"
            ] = selected_cluster_ids

            label_path = (
                HDBSCAN_OUTPUT_DIR
                / (
                    "labels_"
                    f"k{min_samples}_"
                    f"mcs{min_cluster_size}.csv"
                )
            )

            label_table.to_csv(
                label_path,
                index=False,
            )

    summary = pd.DataFrame(
        summary_records
    )

    summary_path = (
        HDBSCAN_OUTPUT_DIR
        / "eom_sensitivity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "HDBSCAN EOM CLUSTER-SELECTION SENSITIVITY"
    )

    print(
        "=" * 90
    )

    print(
        "\nFinal cluster summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nAll EOM label validations: PASS"
    )

    print(
        "\nSaved:"
    )

    print(
        "outputs/clusters/"
        "eom_sensitivity_summary.csv"
    )


if __name__ == "__main__":
    main()