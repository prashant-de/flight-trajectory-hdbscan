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

## Step 2.3 — Cleaning Rules and Base Preprocessing

**Status:** Completed and verified across all original January trajectories.

### Cleaning policy

The MVP preprocessing pipeline applies the following reproducible rules:

1. **Severe multi-segment trajectory files**
   - Exclude a source file from the initial clustering dataset when all three conditions are satisfied:
     - maximum internal row gap > **1 hour**;
     - total file duration > **8 hours**;
     - more than one observed callsign.
   - January analysis identified **12 files** satisfying this combination.
   - These raw files are not deleted or modified. They remain in `data/raw/january/OriginalJan` and may later be revisited as a robustness, segmentation, or extension analysis.

2. **Unusable geometry**
   - Remove observations with:
     - unparseable timestamps;
     - missing latitude or longitude;
     - latitude outside `[-90, 90]`;
     - longitude outside `[-180, 180]`;
     - missing `lastposupdate`.

3. **Carried-forward position observations**
   - Consecutive rows carrying the same `lastposupdate` are reduced to the first observation representing that position update.
   - This removes repeated/stale copies without inventing new coordinates.

4. **Physically impossible spatial discontinuities**
   - A candidate position update is rejected only when both:
     - distance from the last accepted position is greater than **10 km**; and
     - implied movement speed is greater than **3000 km/h**.
   - These deliberately conservative thresholds target only clearly impossible spatial discontinuities rather than normal route deviations.
   - Threshold sensitivity testing showed that a speed-only rule was too aggressive, while the combined distance-and-speed rule preserved normal development trajectories and still detected known corrupted intervals.

5. **Raw data preservation**
   - Preprocessing never modifies the original January CSV files.
   - Retained observations preserve the original source-row index for traceability.

### January-wide preprocessing result

- Original January source files: **566**
- Flights retained: **554**
- Severe multi-segment candidates excluded: **12**
- Original rows in retained flights: **9,422,067**
- Unusable rows removed: **1,939**
- Carried-forward rows removed: **1,008,635**
- Spatially inconsistent position updates removed: **5,160**
- Final cleaned position-update rows: **8,406,333**
- Flights requiring at least one spatial removal: **83**
- Flights with non-positive position-update timing: **0**

The largest number of spatial removals in one retained trajectory was **1,172**.

The preprocessing implementation was optimized so that each source CSV is read once and most spatial screening is performed using vectorized NumPy operations. This preserves the cleaning logic while substantially reducing January-wide processing time.

**Next:** Step 2.4 — Evaluate and select the working trajectory representation/resampling interval for distance calculation.

## Step 2.4 — Trajectory Resampling and Month Harmonization

**Status:** Completed.

### Working temporal resolution

Candidate trajectory representations at **5 s, 10 s, and 30 s** were compared against cleaned high-resolution January trajectories.

The 10-second representation was selected as the working temporal resolution because it provided a strong balance between geometric preservation and computational reduction.

Across all 554 retained January trajectories:

- Cleaned high-resolution position updates: **8,406,333**
- Final 10-second trajectory points: **915,582**
- Overall point reduction: **89.108%**
- Minimum geometric coverage: **98.764%**
- Median geometric coverage: **99.885%**
- Mean geometric coverage: **99.857%**
- Median of per-flight median reconstruction errors: **0.012890 km**
- Median per-flight 95th-percentile reconstruction error: **0.059591 km**
- Largest per-flight 95th-percentile error: **0.138371 km**

Large missing-data gaps are preserved rather than interpolated through. Interpolation is only performed between cleaned observations separated by at most 60 seconds.

### July compatibility

The externally supplied July dataset was verified to be a genuine approximately 10-second dataset:

- Source trajectories: **810**
- Median timestamp gap: **10.000 s**
- Median proportion of timestamp gaps between 8–12 s: **99.835%**
- Duplicate timestamps: **0 flights**
- Non-positive timestamp gaps: **0 flights**

Because July does not contain `lastposupdate` or callsign, the January raw-data cleaning procedure cannot be reproduced identically. July is therefore harmonized conservatively using the available timestamp and coordinate data.

The July harmonization process removes invalid geometry, repeated consecutive coordinates, severe spatial discontinuities, and obvious multi-segment trajectories before re-gridding onto the same exact 10-second working representation used for January.

### Final trajectory-level eligibility rule

A trajectory is excluded from MVP clustering if more than **20% of its unique position updates** must be rejected as physically impossible spatial observations.

The same rule is applied to both months.

January:
- Source trajectories: **566**
- Severe multi-segment exclusions: **12**
- Severe spatial-corruption exclusions: **0**
- Final retained trajectories: **554**
- Largest retained spatial-removal fraction: **12.084%**

July:
- Source trajectories: **810**
- Severe multi-segment exclusions: **5**
- Severe spatial-corruption exclusions: **5**
- Final retained trajectories: **800**
- Largest retained spatial-removal fraction: **14.664%**

The existing July 10-second dataset is therefore sufficient for the MVP. Re-downloading the full July 1-second dataset is not required.

---

## Step 2.5 — Final Processed Dataset

**Status:** Completed and structurally validated.

Final generated structure:

```text
data/processed/
├── january/
│   └── 554 processed trajectory CSV files
└── july/
    └── 800 processed trajectory CSV files

outputs/quality/
├── january_preprocessing_manifest.csv
├── july_preprocessing_manifest.csv
├── processed_dataset_manifest.csv
└── processed_dataset_summary.json
```

Every final trajectory uses the common schema:

- `sample_index`
- `sample_time`
- `time_utc`
- `lat`
- `lon`
- `interpolated`

Final dataset size:

- January trajectories: **554**
- July trajectories: **800**
- Combined trajectories: **1,354**
- January processed points: **915,582**
- July processed points: **1,394,558**
- Combined processed points: **2,310,140**

All generated trajectory files passed structural validation:

- finite timestamps;
- finite and valid latitude/longitude;
- strictly increasing timestamps;
- timestamps aligned to the common 10-second grid;
- valid sequential sample indices.

Raw and externally reduced source files remain unchanged.

### Module 2 conclusion

The project now has a reproducible, quality-controlled, month-harmonized trajectory dataset containing **1,354 LAX → JFK flight trajectories** at a common **10-second temporal resolution**.

This dataset is the input to **Module 3 — Trajectory Representation and Distance Calculation**.

## Module 3 — Trajectory Representation and Distance Calculation

**Status:** Completed for the primary MVP configuration.

### Step 3.1 — Fixed trajectory representation

The validated 10-second trajectories contain approximately 1,500–1,700 points per flight, which is unnecessarily large for full pairwise trajectory-distance calculations.

Fixed-size representations were therefore evaluated using points equally spaced along cumulative geographic path length rather than time. This preserves route geometry while removing differences caused only by flight speed.

Candidate representation sizes:

- 25 points
- 50 points
- 100 points
- 200 points

Across all 1,354 trajectories:

| Points | Median p95 reconstruction error | Worst p95 error | Median reduction |
|---:|---:|---:|---:|
| 25 | 34.689 km | 71.074 km | 98.54% |
| 50 | 15.514 km | 38.857 km | 97.08% |
| 100 | 4.828 km | 16.050 km | 94.16% |
| 200 | 1.119 km | 8.413 km | 88.32% |

The 25-point and 50-point representations were considered too coarse.

A direct 100-versus-200-point Hausdorff comparison showed:

- Pairwise distance correlation: **0.999981**
- Median absolute distance difference: **0.467 km**
- 95th-percentile absolute difference: **2.254 km**
- Exact nearest-neighbor agreement: **95%**
- 100-point nearest neighbor contained in 200-point top three: **100%**
- Mean top-5 neighborhood overlap: **98.0%**
- Mean top-10 neighborhood overlap: **98.75%**
- Mean top-15 neighborhood overlap: **98.5%**
- 200-point runtime approximately **4.5×** the 100-point runtime

Therefore, **100 arc-length points** were selected as the primary MVP representation.

The **200-point representation is retained as a sensitivity configuration** for later robustness analysis.

---

### Step 3.2 — Trajectory distance metric

Three manually implemented trajectory-distance candidates were benchmarked using the selected 100-point representation:

1. Symmetric Hausdorff distance
2. Discrete Fréchet distance
3. Normalized Dynamic Time Warping (DTW)

All point-to-point geographic distances use the Haversine great-circle distance.

Benchmark results:

- Hausdorff runtime: **0.1392 s**
- Discrete Fréchet runtime: **1.7198 s**
- Normalized DTW runtime: **11.4276 s**

Hausdorff and discrete Fréchet produced nearly identical pairwise structures:

- Pearson correlation: **0.999980**
- Rank correlation: **0.999960**
- Exact nearest-neighbor agreement: **100%**
- Mean top-5 neighborhood overlap: **98.0%**
- Mean top-10 neighborhood overlap: **98.67%**

DTW produced a meaningfully different neighborhood structure and was substantially more expensive.

Therefore, the primary MVP trajectory metric is:

**Symmetric Hausdorff distance using Haversine point distance.**

Discrete Fréchet is retained as a later metric-sensitivity configuration.

---

### Step 3.3 — Full pairwise distance matrix

The primary configuration was applied to all final trajectories:

- January trajectories: **554**
- July trajectories: **800**
- Combined trajectories: **1,354**

Number of unique trajectory pairs:

**915,981**

The resulting distance matrix has shape:

**1,354 × 1,354**

Pairwise Hausdorff distance distribution:

- Median: **313.826 km**
- 95th percentile: **715.055 km**
- Maximum: **1,641.603 km**

The matrix passed structural validation:

- square matrix;
- finite values;
- non-negative distances;
- zero diagonal;
- symmetric values.

Saved synchronized artifacts:

```text
outputs/distances/
├── trajectory_representations_100.npy
├── trajectory_index.csv
├── hausdorff_100.npy
└── hausdorff_100_metadata.json
```

`trajectory_index.csv` defines the exact row/column ordering used by the matrix and will be preserved throughout clustering so that HDBSCAN labels can always be mapped back to source flights.

### Primary configuration

```text
10-second validated trajectory
        ↓
100-point arc-length representation
        ↓
Haversine point distance
        ↓
symmetric Hausdorff trajectory distance
        ↓
precomputed distance matrix
        ↓
manual HDBSCAN
```

### Planned sensitivity configurations

The primary configuration is not treated as the only possible modeling choice. Later robustness analysis will include, where computationally appropriate:

- **100 vs 200 arc-length points**
- **Hausdorff vs discrete Fréchet**
- **5 s / 10 s / 30 s temporal preprocessing evidence**
- multiple HDBSCAN `min_samples` values
- multiple HDBSCAN `min_cluster_size` values
- clustering-membership and noise-label stability
- month-specific January/July cluster composition

Sensitivity analysis will therefore evaluate whether the final operational-route conclusions remain stable under reasonable alternative modeling choices.

**Next:** Module 4 — Manual HDBSCAN implementation using the precomputed Hausdorff distance matrix.