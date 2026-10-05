import pandas as pd

from src.config import RAW_JANUARY_DIR


SAMPLE_FILE = RAW_JANUARY_DIR / "20260106T062826Z_a8584a_FFT2504.csv"


def load_flight_csv(file_path):
    """Load one flight trajectory CSV into a pandas DataFrame."""
    return pd.read_csv(file_path)


if __name__ == "__main__":
    flight = load_flight_csv(SAMPLE_FILE)

    print("File:", SAMPLE_FILE.name)
    print("Shape:", flight.shape)

    print("\nColumns:")
    print(flight.columns.tolist())

    print("\nData types:")
    print(flight.dtypes)

    print("\nFirst 3 rows:")
    print(flight.head(3))