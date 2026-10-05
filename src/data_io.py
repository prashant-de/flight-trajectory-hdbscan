import pandas as pd

from src.config import RAW_JANUARY_DIR


SAMPLE_FILE = RAW_JANUARY_DIR / "20260106T062826Z_a8584a_FFT2504.csv"


def load_flight_csv(file_path):
    """Load one flight trajectory CSV into a pandas DataFrame."""
    return pd.read_csv(file_path)


def find_flight_csvs(directory):
    """Find all CSV flight files in a directory."""

    # glob("*.csv") finds CSV files directly inside the given folder.
    flight_files = directory.glob("*.csv")

    # Sort them by filename so our ordering is always reproducible.
    return sorted(flight_files)

def load_flight_metadata(file_path):
    """Extract and verify basic identifiers for one flight file."""

    # Force identifier columns to strings.
    # This prevents values such as ICAO24 "4247e0"
    # from being interpreted as scientific notation.
    metadata = pd.read_csv(
        file_path,
        usecols=["flight_key", "icao24", "callsign"],
        dtype={
            "flight_key": "string",
            "icao24": "string",
            "callsign": "string",
        },
    )

    # Remove leading/trailing whitespace from identifier fields.
    metadata["flight_key"] = metadata["flight_key"].str.strip()
    metadata["icao24"] = metadata["icao24"].str.strip()
    metadata["callsign"] = metadata["callsign"].str.strip()

    # Treat empty strings as missing values.
    metadata["flight_key"] = metadata["flight_key"].replace("", pd.NA)
    metadata["icao24"] = metadata["icao24"].replace("", pd.NA)
    metadata["callsign"] = metadata["callsign"].replace("", pd.NA)

    # Find distinct non-missing values observed in the whole file.
    flight_keys = metadata["flight_key"].dropna().unique()
    icao24_values = metadata["icao24"].dropna().unique()
    callsigns = metadata["callsign"].dropna().unique()

    return {
        "file_name": file_path.name,
        "flight_key": flight_keys[0] if len(flight_keys) == 1 else None,
        "icao24": icao24_values[0] if len(icao24_values) == 1 else None,
        "callsign": callsigns[0] if len(callsigns) == 1 else None,
        "flight_key_count": len(flight_keys),
        "icao24_count": len(icao24_values),
        "callsign_count": len(callsigns),
    }

def get_unique_callsigns(file_path):
    """Return distinct non-missing callsigns observed in one flight file."""

    # Read callsign as text so pandas does not alter identifier formatting.
    callsign_data = pd.read_csv(
        file_path,
        usecols=["callsign"],
        dtype={"callsign": "string"},
    )

    # Normalize whitespace and remove missing/empty values.
    callsigns = callsign_data["callsign"].str.strip()
    callsigns = callsigns.replace("", pd.NA).dropna()

    return callsigns.unique().tolist()

if __name__ == "__main__":
    # Find all original January trajectory files.
    flight_files = find_flight_csvs(RAW_JANUARY_DIR)

    metadata_records = []

    # Extract identifier information from every flight file.
    for file_path in flight_files:
        metadata_records.append(load_flight_metadata(file_path))

    # Combine all flight metadata into one table.
    metadata_table = pd.DataFrame(metadata_records)

    print("Files discovered:", len(flight_files))
    print("Metadata records:", len(metadata_table))
    print("Unique flight keys:", metadata_table["flight_key"].nunique())

    # Each file should ideally contain one unique flight key.
    print("\nFiles with flight_key count != 1:")
    print((metadata_table["flight_key_count"] != 1).sum())

    # Each file should ideally correspond to one aircraft identifier.
    print("\nFiles with icao24 count != 1:")
    print((metadata_table["icao24_count"] != 1).sum())

    # Check whether callsign changes or is completely unavailable.
    print("\nFiles with callsign count != 1:")
    print((metadata_table["callsign_count"] != 1).sum())

    print("\nFiles with missing resolved callsign:")
    print(metadata_table["callsign"].isna().sum())

    # Separate completely missing callsigns from files
    # where more than one callsign appears.
    print("\nFiles with no callsign observed:")
    print((metadata_table["callsign_count"] == 0).sum())

    print("\nFiles with multiple callsigns observed:")
    print((metadata_table["callsign_count"] > 1).sum())

    print("\nFiles without a resolved callsign:")
    print(
        metadata_table.loc[
            metadata_table["callsign"].isna(),
            ["file_name", "flight_key", "icao24", "callsign_count"]
        ].to_string(index=False)
    )

    print("\nDistinct callsigns in files with multiple callsigns:")

    multiple_callsign_files = metadata_table[
        metadata_table["callsign_count"] > 1
    ]

    for file_name in multiple_callsign_files["file_name"]:
        file_path = RAW_JANUARY_DIR / file_name
        callsigns = get_unique_callsigns(file_path)

        print(f"{file_name}: {callsigns}")

    print("\nFirst 5 metadata records:")
    print(metadata_table.head())