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

## Step 2.1 — Timestamp and Freshness Checks

**Status:** Completed and verified on all original January trajectories.

- Checked **566 original January flight files** containing **9,671,132 rows**.
- All trajectory `time` values parsed successfully as UTC timestamps.
- Pandas loaded parsed timestamps at microsecond resolution; conversion to Unix seconds is performed explicitly before freshness calculations.
- Recalculated freshness:
  - `position_age = row Unix time - lastposupdate`
  - `contact_age = row Unix time - lastcontact`
- No negative recalculated position or contact ages were observed in the development flights examined.
- `lastcontact` was available throughout the January dataset.
- **188 flights** contain at least one missing `lastposupdate`.
- Total rows with missing `lastposupdate`: **4,829**, approximately **0.0499%** of all January observations.
- No flight has all `lastposupdate` values missing.
- Missing `lastposupdate` is treated as **unknown position freshness**, not as fresh or stale.
- Stale position observations occur across many flights; reporting thresholds of 10 s, 30 s, and 60 s were used diagnostically only and are **not cleaning thresholds**.
- The largest observed recalculated position age was **637.953 s** in `20260130T072141Z_add734_JBU324.csv`.
- The supplied `position_age_seconds` and `contact_age_seconds` fields are systematically invalid:
  - all 566 flights contain negative supplied position-age values;
  - **9,666,303** supplied position-age rows are negative and the remaining **4,829** are missing;
  - all **9,671,132** supplied contact-age values are negative.
- Supplied age columns will therefore be preserved for traceability but will **not** be used for quality decisions. Recalculated age values will be used instead.
- No rows have been removed, interpolated, or modified during Step 2.1.

**Next:** Step 2.2 — Gaps and coordinate checks.