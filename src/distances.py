import numpy as np
import pandas as pd
from time import perf_counter

from src.config import (
    PROCESSED_JANUARY_DIR,
    PROCESSED_JULY_DIR,
)


EARTH_RADIUS_KM = 6371.0088


# Candidate representation sizes

REPRESENTATION_POINT_COUNTS = [
    25,
    50,
    100,
    200,
]


# Geographic helpers

def consecutive_haversine_km(
    latitude,
    longitude,
):
    """
    Calculate great-circle distance between consecutive points.
    """

    latitude = np.asarray(
        latitude,
        dtype="float64",
    )

    longitude = np.asarray(
        longitude,
        dtype="float64",
    )

    n = len(latitude)

    distances = np.zeros(
        n,
        dtype="float64",
    )

    if n <= 1:
        return distances

    lat_rad = np.radians(
        latitude
    )

    lon_rad = np.radians(
        longitude
    )

    lat1 = lat_rad[:-1]
    lat2 = lat_rad[1:]

    lon1 = lon_rad[:-1]
    lon2 = lon_rad[1:]

    dlat = (
        lat2 - lat1
    )

    dlon = (
        lon2 - lon1
    )

    a = (
        np.sin(
            dlat / 2
        ) ** 2
        + np.cos(
            lat1
        )
        * np.cos(
            lat2
        )
        * np.sin(
            dlon / 2
        ) ** 2
    )

    a = np.clip(
        a,
        0,
        1,
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(
            1 - a
        ),
    )

    distances[1:] = (
        EARTH_RADIUS_KM * c
    )

    return distances


def pointwise_haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate great-circle distance between corresponding points.
    """

    lat1 = np.radians(
        np.asarray(
            lat1,
            dtype="float64",
        )
    )

    lon1 = np.radians(
        np.asarray(
            lon1,
            dtype="float64",
        )
    )

    lat2 = np.radians(
        np.asarray(
            lat2,
            dtype="float64",
        )
    )

    lon2 = np.radians(
        np.asarray(
            lon2,
            dtype="float64",
        )
    )

    dlat = (
        lat2 - lat1
    )

    dlon = (
        lon2 - lon1
    )

    a = (
        np.sin(
            dlat / 2
        ) ** 2
        + np.cos(
            lat1
        )
        * np.cos(
            lat2
        )
        * np.sin(
            dlon / 2
        ) ** 2
    )

    a = np.clip(
        a,
        0,
        1,
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(
            1 - a
        ),
    )

    return (
        EARTH_RADIUS_KM * c
    )


# Arc-length trajectory representation

def build_arc_length_representation(
    trajectory,
    n_points,
):
    """
    Represent a trajectory using a fixed number of points
    equally spaced along cumulative geographic path length.

    This representation focuses on route geometry rather than
    flight duration or aircraft speed.
    """

    latitude = (
        trajectory["lat"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    longitude = (
        trajectory["lon"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    if len(latitude) < 2:
        raise ValueError(
            "Trajectory must contain at least two points."
        )

    segment_distance = (
        consecutive_haversine_km(
            latitude,
            longitude,
        )
    )

    cumulative_distance = (
        np.cumsum(
            segment_distance
        )
    )

    total_distance = (
        cumulative_distance[-1]
    )

    if (
        not np.isfinite(
            total_distance
        )
        or total_distance <= 0
    ):
        raise ValueError(
            "Trajectory has zero or invalid path length."
        )

    # np.interp expects increasing x values.
    # Remove points that contribute zero additional path distance.
    unique_mask = np.concatenate(
        (
            [True],
            np.diff(
                cumulative_distance
            ) > 1e-9,
        )
    )

    cumulative_unique = (
        cumulative_distance[
            unique_mask
        ]
    )

    latitude_unique = (
        latitude[
            unique_mask
        ]
    )

    longitude_unique = (
        longitude[
            unique_mask
        ]
    )

    if len(
        cumulative_unique
    ) < 2:
        raise ValueError(
            "Trajectory does not contain enough unique geometry."
        )

    target_distance = np.linspace(
        0.0,
        total_distance,
        n_points,
    )

    represented_latitude = np.interp(
        target_distance,
        cumulative_unique,
        latitude_unique,
    )

    represented_longitude = np.interp(
        target_distance,
        cumulative_unique,
        longitude_unique,
    )

    path_fraction = (
        target_distance
        / total_distance
    )

    representation = pd.DataFrame(
        {
            "point_index": np.arange(
                n_points,
                dtype="int64",
            ),

            "path_fraction": (
                path_fraction
            ),

            "lat": (
                represented_latitude
            ),

            "lon": (
                represented_longitude
            ),
        }
    )

    return (
        representation,
        cumulative_distance,
        total_distance,
    )

def pairwise_haversine_matrix_km(
    trajectory_a,
    trajectory_b,
):
    """
    Compute all great-circle distances between two
    fixed-point trajectory representations.

    Returns a matrix with shape:
        len(A) x len(B)
    """

    lat_a = np.radians(
        trajectory_a["lat"].to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    lon_a = np.radians(
        trajectory_a["lon"].to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    lat_b = np.radians(
        trajectory_b["lat"].to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    lon_b = np.radians(
        trajectory_b["lon"].to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    lat1 = lat_a[:, None]
    lon1 = lon_a[:, None]

    lat2 = lat_b[None, :]
    lon2 = lon_b[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(
        a,
        0,
        1,
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    return (
        EARTH_RADIUS_KM * c
    )


def hausdorff_distance_km(
    trajectory_a,
    trajectory_b,
):
    """
    Symmetric Hausdorff distance between two trajectories.

    H(A, B) = max(
        max_a min_b d(a,b),
        max_b min_a d(a,b)
    )
    """

    distance_matrix = (
        pairwise_haversine_matrix_km(
            trajectory_a,
            trajectory_b,
        )
    )

    directed_a_to_b = (
        distance_matrix
        .min(axis=1)
        .max()
    )

    directed_b_to_a = (
        distance_matrix
        .min(axis=0)
        .max()
    )

    return max(
        directed_a_to_b,
        directed_b_to_a,
    )
def discrete_frechet_distance_km(
    trajectory_a,
    trajectory_b,
):
    """
    Compute the discrete Frechet distance between two trajectories.

    Unlike Hausdorff distance, discrete Frechet preserves the
    sequential ordering of points along each trajectory.

    The implementation uses dynamic programming and only keeps
    two rows of the DP table in memory.
    """

    distance_matrix = (
        pairwise_haversine_matrix_km(
            trajectory_a,
            trajectory_b,
        )
    )

    n, m = distance_matrix.shape

    previous = np.empty(
        m,
        dtype="float64",
    )

    current = np.empty(
        m,
        dtype="float64",
    )

    # First row.
    previous[0] = (
        distance_matrix[0, 0]
    )

    for j in range(
        1,
        m,
    ):
        previous[j] = max(
            previous[j - 1],
            distance_matrix[0, j],
        )

    # Remaining rows.
    for i in range(
        1,
        n,
    ):
        current[0] = max(
            previous[0],
            distance_matrix[i, 0],
        )

        for j in range(
            1,
            m,
        ):
            predecessor = min(
                previous[j],
                previous[j - 1],
                current[j - 1],
            )

            current[j] = max(
                predecessor,
                distance_matrix[i, j],
            )

        previous, current = (
            current,
            previous,
        )

    return previous[
        m - 1
    ]


def normalized_dtw_distance_km(
    trajectory_a,
    trajectory_b,
):
    """
    Compute normalized Dynamic Time Warping distance.

    Standard DTW accumulates pointwise distances along the best
    warping path. Because raw accumulated DTW depends on path
    length, this implementation divides the accumulated distance
    by the number of matched point pairs along that path.

    The returned value can therefore be interpreted approximately
    as average spatial separation along the optimal alignment.
    """

    distance_matrix = (
        pairwise_haversine_matrix_km(
            trajectory_a,
            trajectory_b,
        )
    )

    n, m = distance_matrix.shape

    previous_cost = np.empty(
        m,
        dtype="float64",
    )

    current_cost = np.empty(
        m,
        dtype="float64",
    )

    previous_steps = np.empty(
        m,
        dtype="int64",
    )

    current_steps = np.empty(
        m,
        dtype="int64",
    )

    # -------------------------------------------------------------
    # First row
    # -------------------------------------------------------------

    previous_cost[0] = (
        distance_matrix[0, 0]
    )

    previous_steps[0] = 1

    for j in range(
        1,
        m,
    ):
        previous_cost[j] = (
            previous_cost[j - 1]
            + distance_matrix[0, j]
        )

        previous_steps[j] = (
            previous_steps[j - 1]
            + 1
        )

    # -------------------------------------------------------------
    # Remaining rows
    # -------------------------------------------------------------

    for i in range(
        1,
        n,
    ):
        current_cost[0] = (
            previous_cost[0]
            + distance_matrix[i, 0]
        )

        current_steps[0] = (
            previous_steps[0]
            + 1
        )

        for j in range(
            1,
            m,
        ):
            predecessor_costs = (
                previous_cost[j],
                current_cost[j - 1],
                previous_cost[j - 1],
            )

            predecessor_choice = int(
                np.argmin(
                    predecessor_costs
                )
            )

            if predecessor_choice == 0:
                best_cost = (
                    previous_cost[j]
                )

                best_steps = (
                    previous_steps[j]
                )

            elif predecessor_choice == 1:
                best_cost = (
                    current_cost[j - 1]
                )

                best_steps = (
                    current_steps[j - 1]
                )

            else:
                best_cost = (
                    previous_cost[j - 1]
                )

                best_steps = (
                    previous_steps[j - 1]
                )

            current_cost[j] = (
                best_cost
                + distance_matrix[i, j]
            )

            current_steps[j] = (
                best_steps + 1
            )

        previous_cost, current_cost = (
            current_cost,
            previous_cost,
        )

        previous_steps, current_steps = (
            current_steps,
            previous_steps,
        )

    total_cost = previous_cost[
        m - 1
    ]

    path_length = previous_steps[
        m - 1
    ]

    return (
        total_cost
        / path_length
    )


def build_metric_distance_matrix(
    representations,
    metric_function,
):
    """
    Build a complete symmetric trajectory-distance matrix.
    """

    n = len(
        representations
    )

    matrix = np.zeros(
        (n, n),
        dtype="float64",
    )

    start = perf_counter()

    for i in range(n):

        for j in range(
            i + 1,
            n,
        ):
            distance = metric_function(
                representations[i],
                representations[j],
            )

            matrix[
                i,
                j,
            ] = distance

            matrix[
                j,
                i,
            ] = distance

    elapsed = (
        perf_counter()
        - start
    )

    return matrix, elapsed


def simple_rank_correlation(
    values_a,
    values_b,
):
    """
    Calculate correlation between simple rank orders.

    Pairwise trajectory distances are effectively continuous,
    so exact ties are unlikely in this experiment.
    """

    rank_a = np.argsort(
        np.argsort(
            values_a
        )
    )

    rank_b = np.argsort(
        np.argsort(
            values_b
        )
    )

    return np.corrcoef(
        rank_a,
        rank_b,
    )[0, 1]


def compare_neighbor_sets(
    matrix_a,
    matrix_b,
    k,
):
    """
    Calculate top-k neighborhood overlap between two metrics.
    """

    n = matrix_a.shape[0]

    overlaps = []

    exact_nearest_matches = 0

    for i in range(n):

        ranking_a = np.argsort(
            matrix_a[i]
        )

        ranking_a = ranking_a[
            ranking_a != i
        ]

        ranking_b = np.argsort(
            matrix_b[i]
        )

        ranking_b = ranking_b[
            ranking_b != i
        ]

        if (
            ranking_a[0]
            == ranking_b[0]
        ):
            exact_nearest_matches += 1

        neighbors_a = set(
            ranking_a[
                :k
            ]
        )

        neighbors_b = set(
            ranking_b[
                :k
            ]
        )

        overlap = (
            len(
                neighbors_a
                & neighbors_b
            )
            / k
            * 100
        )

        overlaps.append(
            overlap
        )

    return {
        "exact_nn_percent": (
            exact_nearest_matches
            / n
            * 100
        ),

        "mean_overlap_percent": (
            np.mean(
                overlaps
            )
        ),

        "median_overlap_percent": (
            np.median(
                overlaps
            )
        ),

        "minimum_overlap_percent": (
            np.min(
                overlaps
            )
        ),
    }

# Representation-quality evaluation

def evaluate_representation(
    trajectory,
    n_points,
):
    """
    Measure how closely a fixed-point representation reproduces
    the original validated 10-second trajectory geometry.
    """

    (
        representation,
        cumulative_distance,
        total_distance,
    ) = build_arc_length_representation(
        trajectory,
        n_points,
    )

    original_fraction = (
        cumulative_distance
        / total_distance
    )

    represented_fraction = (
        representation[
            "path_fraction"
        ]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    represented_latitude = (
        representation["lat"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    represented_longitude = (
        representation["lon"]
        .to_numpy(
            dtype="float64",
            copy=False,
        )
    )

    reconstructed_latitude = np.interp(
        original_fraction,
        represented_fraction,
        represented_latitude,
    )

    reconstructed_longitude = np.interp(
        original_fraction,
        represented_fraction,
        represented_longitude,
    )

    errors = pointwise_haversine_km(
        trajectory["lat"],
        trajectory["lon"],
        reconstructed_latitude,
        reconstructed_longitude,
    )

    original_rows = len(
        trajectory
    )

    reduction_percent = (
        (
            1
            - (
                n_points
                / original_rows
            )
        )
        * 100
    )

    return {
        "original_rows": (
            original_rows
        ),

        "representation_points": (
            n_points
        ),

        "path_length_km": (
            total_distance
        ),

        "reduction_percent": (
            reduction_percent
        ),

        "median_error_km": (
            np.median(
                errors
            )
        ),

        "p95_error_km": (
            np.percentile(
                errors,
                95,
            )
        ),

        "max_error_km": (
            np.max(
                errors
            )
        ),
    }


# Dataset-wide experiment

def main():
    """
    Compare Hausdorff, discrete Frechet, and normalized DTW
    using the selected 100-point arc-length representation.

    This is a development benchmark rather than the final
    full 1,354 x 1,354 distance calculation.
    """

    january_files = sorted(
        PROCESSED_JANUARY_DIR.glob(
            "*.csv"
        )
    )

    july_files = sorted(
        PROCESSED_JULY_DIR.glob(
            "*.csv"
        )
    )

    # -------------------------------------------------------------
    # Deterministic sample:
    # 15 January + 15 July = 30 trajectories.
    # -------------------------------------------------------------

    january_indices = np.linspace(
        0,
        len(january_files) - 1,
        15,
        dtype=int,
    )

    july_indices = np.linspace(
        0,
        len(july_files) - 1,
        15,
        dtype=int,
    )

    selected_files = []

    for index in january_indices:
        selected_files.append(
            (
                "2026-01",
                january_files[
                    index
                ],
            )
        )

    for index in july_indices:
        selected_files.append(
            (
                "2026-07",
                july_files[
                    index
                ],
            )
        )

    print(
        "Benchmark trajectories:",
        len(
            selected_files
        ),
    )

    representations = []

    names = []

    for month, file_path in (
        selected_files
    ):
        trajectory = pd.read_csv(
            file_path,
            usecols=[
                "lat",
                "lon",
            ],
        )

        representation, _, _ = (
            build_arc_length_representation(
                trajectory,
                100,
            )
        )

        representations.append(
            representation
        )

        names.append(
            f"{month}:{file_path.name}"
        )

    n = len(
        representations
    )

    pair_count = (
        n
        * (
            n - 1
        )
        // 2
    )

    print(
        "Unique benchmark pairs:",
        pair_count,
    )

    # -------------------------------------------------------------
    # Hausdorff
    # -------------------------------------------------------------

    print(
        "\nComputing Hausdorff..."
    )

    hausdorff_matrix, hausdorff_time = (
        build_metric_distance_matrix(
            representations,
            hausdorff_distance_km,
        )
    )

    # -------------------------------------------------------------
    # Discrete Frechet
    # -------------------------------------------------------------

    print(
        "Computing discrete Frechet..."
    )

    frechet_matrix, frechet_time = (
        build_metric_distance_matrix(
            representations,
            discrete_frechet_distance_km,
        )
    )

    # -------------------------------------------------------------
    # Normalized DTW
    # -------------------------------------------------------------

    print(
        "Computing normalized DTW..."
    )

    dtw_matrix, dtw_time = (
        build_metric_distance_matrix(
            representations,
            normalized_dtw_distance_km,
        )
    )

    upper = np.triu_indices(
        n,
        k=1,
    )

    hausdorff_values = (
        hausdorff_matrix[
            upper
        ]
    )

    frechet_values = (
        frechet_matrix[
            upper
        ]
    )

    dtw_values = (
        dtw_matrix[
            upper
        ]
    )

    # -------------------------------------------------------------
    # Global relationship between metrics
    # -------------------------------------------------------------

    metric_pairs = [
        (
            "Hausdorff",
            hausdorff_values,
            hausdorff_matrix,
            "Frechet",
            frechet_values,
            frechet_matrix,
        ),

        (
            "Hausdorff",
            hausdorff_values,
            hausdorff_matrix,
            "DTW",
            dtw_values,
            dtw_matrix,
        ),

        (
            "Frechet",
            frechet_values,
            frechet_matrix,
            "DTW",
            dtw_values,
            dtw_matrix,
        ),
    ]

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAJECTORY DISTANCE METRIC BENCHMARK"
    )

    print(
        "=" * 70
    )

    print(
        "\nPairs evaluated:"
    )

    print(
        pair_count
    )

    print(
        "\nRuntime:"
    )

    print(
        "Hausdorff:",
        f"{hausdorff_time:.4f} s"
    )

    print(
        "Discrete Frechet:",
        f"{frechet_time:.4f} s"
    )

    print(
        "Normalized DTW:",
        f"{dtw_time:.4f} s"
    )

    print(
        "\nDistance distributions:"
    )

    distribution_table = (
        pd.DataFrame(
            [
                {
                    "metric": "Hausdorff",
                    "median_km": np.median(
                        hausdorff_values
                    ),
                    "p95_km": np.percentile(
                        hausdorff_values,
                        95,
                    ),
                    "maximum_km": np.max(
                        hausdorff_values
                    ),
                },

                {
                    "metric": "Discrete Frechet",
                    "median_km": np.median(
                        frechet_values
                    ),
                    "p95_km": np.percentile(
                        frechet_values,
                        95,
                    ),
                    "maximum_km": np.max(
                        frechet_values
                    ),
                },

                {
                    "metric": "Normalized DTW",
                    "median_km": np.median(
                        dtw_values
                    ),
                    "p95_km": np.percentile(
                        dtw_values,
                        95,
                    ),
                    "maximum_km": np.max(
                        dtw_values
                    ),
                },
            ]
        )
    )

    print(
        distribution_table.to_string(
            index=False
        )
    )

    for (
        name_a,
        values_a,
        matrix_a,
        name_b,
        values_b,
        matrix_b,
    ) in metric_pairs:

        pearson = np.corrcoef(
            values_a,
            values_b,
        )[0, 1]

        rank_correlation = (
            simple_rank_correlation(
                values_a,
                values_b,
            )
        )

        top5 = (
            compare_neighbor_sets(
                matrix_a,
                matrix_b,
                k=5,
            )
        )

        top10 = (
            compare_neighbor_sets(
                matrix_a,
                matrix_b,
                k=10,
            )
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"{name_a} vs {name_b}"
        )

        print(
            "-" * 70
        )

        print(
            "Pearson distance correlation:",
            f"{pearson:.6f}"
        )

        print(
            "Rank correlation:",
            f"{rank_correlation:.6f}"
        )

        print(
            "Exact nearest-neighbor agreement:",
            f"{top5['exact_nn_percent']:.1f}%"
        )

        print(
            "Top-5 mean neighborhood overlap:",
            f"{top5['mean_overlap_percent']:.2f}%"
        )

        print(
            "Top-5 minimum overlap:",
            f"{top5['minimum_overlap_percent']:.2f}%"
        )

        print(
            "Top-10 mean neighborhood overlap:",
            f"{top10['mean_overlap_percent']:.2f}%"
        )

        print(
            "Top-10 minimum overlap:",
            f"{top10['minimum_overlap_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()