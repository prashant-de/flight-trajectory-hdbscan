from pathlib import Path


# Project root directory: HDBSCAN CLUSTERING/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Original source data
RAW_JANUARY_DIR = PROJECT_ROOT / "data" / "raw" / "january" / "OriginalJan"
RAW_JULY_DIR = PROJECT_ROOT / "data" / "raw" / "july"


#  reduced (~10 s) data
REDUCED_JANUARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "reduced"
    / "january"
    / "trajectory_january"
)

REDUCED_JULY_DIR = (
    PROJECT_ROOT
    / "data"
    / "reduced"
    / "july"
    / "trajectory_july"
)


# Outputs produced by our preprocessing pipeline
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_JANUARY_DIR = (
    PROCESSED_DIR / "january"
)

PROCESSED_JULY_DIR = (
    PROCESSED_DIR / "july"
)

QUALITY_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "quality"
)

DISTANCE_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "distances"
)