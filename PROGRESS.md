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

## Step 2.2 — Gaps and Coordinate Quality Checks

**Status:** Completed sufficiently for the MVP and verified on the original January dataset.

### Dataset-wide checks

- Checked **566 original January trajectory files**.
- No duplicate row timestamps were found.
- No negative/backward row-time gaps were found.
- **280 flights** contain at least one row gap greater than 5 seconds.
- **221 flights** contain at least one row gap greater than 30 seconds.
- **186 flights** contain at least one row gap greater than 60 seconds.
- **59 flights** contain a maximum row gap greater than 5 minutes.
- **12 flights** contain a maximum row gap greater than 1 hour.
- The largest observed row gap was **53,519 seconds (~14.87 hours)**.

### Coordinate availability and validity

- **188 flights** contain at least one row with missing latitude or longitude.
- Total rows with missing coordinates: **4,829**.
- These counts match the missing-`lastposupdate` observations identified in Step 2.1.
- No latitude values outside `[-90, 90]` were found.
- No longitude values outside `[-180, 180]` were found.

### Repeated and stale positions

- Exact consecutive repeated coordinates are common in the source observations.
- **469 flights** contain an identical-coordinate run longer than 60 seconds.
- The longest identical-coordinate run was **637 seconds** in `20260130T072141Z_add734_JBU324.csv`.
- This same flight had the largest recalculated position age in Step 2.1: approximately **637.953 seconds**.
- Development-flight tests showed close agreement between long repeated-coordinate runs and recalculated `lastposupdate` freshness ages.
- Exact repeated coordinates are therefore treated as a complementary quality indicator, while `lastposupdate` remains the primary position-freshness indicator when available.
- Short repeated-coordinate runs are not automatically considered bad data.

### Severe multi-segment trajectory candidates

Module 1 identified **27 files containing more than one observed callsign**. Step 2.2 showed that these do not all represent the same type of problem.

A distinct group of **12 files** shows all of the following:

- more than one plausible callsign;
- an internal row-time gap greater than 1 hour;
- unusually long total file durations of approximately 10–33 hours.

All **12 / 12** flights with a maximum row gap greater than 1 hour also contain multiple observed callsigns.

Detailed inspection of `20260121T043329Z_e8021d_LPE2690.csv` showed:

- the same aircraft identifier (`icao24 = e8021d`);
- an early segment under callsign `LPE2477`;
- a **53,519-second** discontinuity;
- a later geographically separate segment under callsign `LPE2690`.

This provides strong evidence that some source CSV files contain observations from multiple operational flight segments associated with the same aircraft.

Therefore:

- `icao24` is treated as an aircraft identifier, not a unique flight identifier;
- one source CSV is **not automatically assumed to equal one valid operational trajectory**;
- multiple callsigns alone are not sufficient reason to reject a flight;
- the combination of extreme temporal discontinuity, unusually long duration, and multiple plausible callsigns is treated as a severe multi-segment quality flag.

### MVP handling of the 12 severe files

For the initial clustering pipeline, the **12 severe multi-segment candidate files will be excluded from the clustering input rather than manually repaired or split**.

The original source files are **not deleted or modified**. They remain in:

`data/raw/january/OriginalJan`

The exclusion decision and reason will be preserved in the preprocessing/quality report.

These files may later be revisited as:

- a robustness analysis;
- an extension of the project;
- or a trajectory-segmentation study if the original data-extraction logic is investigated in more detail.

### Spatial discontinuities

Basic coordinate-range validation alone is not sufficient because geographically valid coordinates can still form physically impossible trajectory jumps.

Development-flight inspection revealed examples such as:

- `FFT2504`: maximum consecutive position-update distance ≈ **32 km**
- `JBU524`: ≈ **109 km**
- `JBU1724`: ≈ **678 km**
- `AAL2`: ≈ **693 km**

Detailed inspection of `AAL2` showed several large back-and-forth position jumps within only a few minutes, including a **~693 km jump over ~5.9 seconds**, while callsign, ICAO24, altitude, velocity, and heading remained broadly consistent.

This behavior cannot represent physical aircraft movement and is treated as a spatial data-quality problem rather than a genuine route deviation.

Weather or turbulence may cause real detours, altitude changes, speed changes, or holding patterns, but such changes should still form a physically continuous trajectory. Therefore, impossible instantaneous spatial jumps are not interpreted as weather effects.

Receiver metadata (`serials`) differed across some suspicious AAL2 observations, suggesting that receiver/data-fusion effects may contribute, but the exact technical cause has **not been established** and is not required for the MVP.

### Step 2.2 decision

Step 2.2 identified three distinct quality problems that preprocessing must handle:

1. **Stale / carried-forward position observations**
2. **Severe multi-segment trajectory files**
3. **Physically impossible spatial discontinuities within otherwise continuous flights**

No raw source files were modified during Step 2.2.

**Next:** Step 2.3 — Define and implement reproducible cleaning rules for the MVP.