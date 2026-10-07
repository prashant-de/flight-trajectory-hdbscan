"""
Command-line entry point for the flight-trajectory HDBSCAN project.

This interface intentionally exposes only the finalized and
reasonably inexpensive project stages.

Expensive preprocessing and sensitivity computations are kept as
separate modules and are not automatically rerun from this file.

Examples
--------
Show project summary:

    python main.py

Run the frozen manual HDBSCAN solution:

    python main.py cluster

Run final quantitative evaluation:

    python main.py evaluate

Generate final figures:

    python main.py visualize

Validate the manual implementation against scikit-learn:

    python main.py validate

Run the final reproducible analysis sequence:

    python main.py final
"""

import argparse
import sys

# Project summary

def print_project_summary():
    """
    Print the frozen primary project configuration and available commands.
    """

    print("=" * 88)
    print("FLIGHT TRAJECTORY CLUSTERING USING HDBSCAN")
    print("=" * 88)

    print(
        """
Study:
    LAX -> JFK flight-trajectory clustering

Dataset:
    January 2026: 554 trajectories
    July 2026:    800 trajectories
    Total:       1,354 trajectories

Frozen primary configuration:
    sampling:          validated 10-second trajectories
    representation:    100 arc-length points
    point distance:    Haversine
    trajectory metric: symmetric Hausdorff
    HDBSCAN:
        min_samples = 15
        min_cluster_size = 30

Final result:
    clusters: 4
    Cluster 0: 73 trajectories
    Cluster 1: 558 trajectories
    Cluster 2: 88 trajectories
    Cluster 3: 340 trajectories
    clustered trajectories: 1,059
    noise/anomaly candidates: 295

Available commands:
    python main.py cluster
    python main.py evaluate
    python main.py visualize
    python main.py validate
    python main.py final

Use:
    python main.py --help

for command descriptions.
"""
    )

# Command runners
def run_cluster():
    """
    Reproduce the frozen manual HDBSCAN clustering from the saved
    primary Hausdorff distance matrix.
    """

    print("\nRunning frozen manual HDBSCAN clustering...\n")

    from src.run_manual_hdbscan import main as cluster_main

    cluster_main()


def run_evaluation():
    """
    Run final quantitative evaluation.
    """

    print("\nRunning final cluster evaluation...\n")

    from src.evaluation import main as evaluation_main

    evaluation_main()


def run_visualization():
    """
    Generate the final five-figure visualization set.
    """

    print("\nGenerating final visualization set...\n")

    from src.visualization import main as visualization_main

    visualization_main()


def run_reference_validation():
    """
    Validate the manual HDBSCAN implementation against the
    equivalent scikit-learn HDBSCAN configuration.
    """

    print(
        "\nRunning manual-vs-scikit-learn "
        "reference validation...\n"
    )

    from src.validate_hdbscan_reference import (
        main as validation_main,
    )

    validation_main()


def run_final_pipeline():
    """
    Run the finalized reproducible analysis stages.

    This deliberately does NOT rebuild raw datasets, the full
    Hausdorff matrix, the 200-point sensitivity matrix, or the
    discrete Frechet matrix.
    """

    print("=" * 88)
    print("FINAL REPRODUCIBLE ANALYSIS")
    print("=" * 88)

    print(
        "\nThis sequence uses existing processed data and the "
        "frozen primary distance matrix."
    )

    print(
        "It does not rerun expensive preprocessing or sensitivity "
        "matrix construction.\n"
    )

    print("-" * 88)
    print("STEP 1/4 — Manual HDBSCAN")
    print("-" * 88)

    run_cluster()

    print("\n" + "-" * 88)
    print("STEP 2/4 — Reference validation")
    print("-" * 88)

    run_reference_validation()

    print("\n" + "-" * 88)
    print("STEP 3/4 — Final evaluation")
    print("-" * 88)

    run_evaluation()

    print("\n" + "-" * 88)
    print("STEP 4/4 — Final visualization")
    print("-" * 88)

    run_visualization()

    print("\n" + "=" * 88)
    print("FINAL REPRODUCIBLE ANALYSIS: PASS")
    print("=" * 88)

# CLI
def build_parser():
    """
    Build the project command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Flight-trajectory HDBSCAN project interface. "
            "Expensive preprocessing and sensitivity-matrix "
            "construction are intentionally excluded."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    subparsers.add_parser(
        "cluster",
        help=(
            "Run the frozen manual HDBSCAN clustering."
        ),
    )

    subparsers.add_parser(
        "evaluate",
        help=(
            "Run final quantitative cluster evaluation."
        ),
    )

    subparsers.add_parser(
        "visualize",
        help=(
            "Generate the final report visualization set."
        ),
    )

    subparsers.add_parser(
        "validate",
        help=(
            "Validate manual HDBSCAN against scikit-learn."
        ),
    )

    subparsers.add_parser(
        "final",
        help=(
            "Run clustering, reference validation, "
            "evaluation, and visualization."
        ),
    )

    return parser


def main():
    """
    Parse the requested command and dispatch the corresponding
    finalized project stage.
    """

    parser = build_parser()

    args = parser.parse_args()

    if args.command is None:
        print_project_summary()
        return

    if args.command == "cluster":
        run_cluster()

    elif args.command == "evaluate":
        run_evaluation()

    elif args.command == "visualize":
        run_visualization()

    elif args.command == "validate":
        run_reference_validation()

    elif args.command == "final":
        run_final_pipeline()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()