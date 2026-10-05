## Module 1 — Setup and Loading

**Status:** Completed and locally verified.

- Project structure created with separate `raw`, `reduced`, `processed`, output, and source-code directories.
- Python 3.14 virtual environment configured.
- Current dependencies: NumPy 2.5.3, pandas 3.0.6, Matplotlib 3.11.2.
- Dataset paths centralized in `src/config.py`.
- `src/data_io.py` can load individual trajectory CSVs and discover multiple flight files.
- Original January dataset contains **566 CSV files**.
- Metadata extraction produced **566 records and 566 unique `flight_key` values**.
- Every file currently has exactly one non-missing `flight_key` and one `icao24`.
- ICAO24 fields are explicitly loaded as strings to prevent hexadecimal identifiers such as `4247e0` from being misinterpreted numerically.
- **27 files contain two distinct callsigns.** These are retained and flagged for later quality investigation rather than corrected or removed.
- Callsign is treated as descriptive metadata; `flight_key` remains the primary file/trajectory identifier.
- No trajectory cleaning, filtering, or resampling has been performed yet.

**Next:** Module 2.1 — Timestamp and freshness checks using original January observations.