# Liljegren WBGT implementation analysis

Reproducible analysis supporting a comparison of Liljegren wet-bulb globe
temperature implementations over HeatStressBench ERA5 runs. Code under `src/`
contains testable computation; notebooks provide paper-facing analysis; scripts
provide hardened headless execution.

## Analysis design

For each date, valid WBGT values are matched by run, cycle, and row identity.
Signed differences are `implementation_a - implementation_b`, with names
ordered lexicographically so each unordered pair appears once.

The v4 production query touches the full-resolution ERA5 results once per date.
Pairwise scope uses one implementation self-join. Complete-case scope aggregates
the seven values once and expands the 21 fixed pairs without a join-back or
self-join. A latitude-weight join and grouped aggregation then emit:

- exact unweighted and cell-area-weighted sufficient sums and extrema;
- a fixed-bin absolute-difference histogram for annual P50/P95/P99.

Daily partitions are persisted atomically to one DuckDB cache. Interrupted runs
are resumable and reruns replace dates idempotently. The annual notebook reads
only these compact summaries; it never reconstructs observation-level annual
differences.

The runner supports two populations:

- `pairwise`: each pair uses rows valid for those two implementations;
- `all-implementations`: every pair uses the same rows valid across the fixed
  canonical seven implementations.

Both row- and area-weighted statistics are always produced together.

## Repository structure

- `notebooks/` — interactive daily execution and annual paper analysis.
- `scripts/` — repeatable headless daily processing.
- `src/liljegren_analysis/` — result scanning, SQL, distribution definitions,
  persistence, and pooling formulas.
- `tests/` — aggregation, persistence, distribution, and pooling regression
  tests.
- `docs/` — result layout and statistical method details.

## Inputs

Each supplied HeatStressBench campaign root must contain:

```text
results/<date>/<run>/<implementation>/results.parquet
```

Campaigns under `.hsb/` and generated files under `data/` are ignored by Git.

## Environment

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest -q
```

## Run the analysis

The default pair-specific run writes `data/daily_pairwise_v4.duckdb`:

```bash
uv run python scripts/daily_pairwise.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --run /path/to/campaign \
  --sample-scope pairwise
```

The paper complete-case cache writes
`data/daily_pairwise_all_implementations_v4.duckdb`:

```bash
uv run python scripts/daily_pairwise.py \
  --sample-scope all-implementations
```

Matching log paths are derived automatically. `--database` and `--log-file`
override the defaults. Each date uses a fresh worker process; the controller is
the only database writer.

For annual analysis:

```bash
uv run jupyter lab
```

Open `notebooks/yearly_pairwise.ipynb`. It pools exact sums, reconstructs
histogram quantile intervals, and builds row/area-weighted matrices including
parametric limits of agreement.

Extract the reusable country-specific Parquet cache before opening the country
classification notebook:

```bash
uv run python scripts/extract_country.py JPN
```

The default output is `data/country/<iso3>/observations.parquet`, accompanied by
`manifest.json`. Use `--start-date`, `--end-date`, repeatable `--run`, and
`--threads` to change extraction scope; existing caches require explicit
`--force` replacement.

For raw country classification-transition analysis with editable thresholds,
open `notebooks/country_classification_changes.ipynb`. It reads the cached
country observations rather than rescanning campaign Parquet. Edit the
parameters cell immediately after the imports: `COUNTRY_CODE` accepts ISO
alpha-2 or alpha-3
(default `SG`; use `JPN` for Japan), while the dates, thresholds, and DuckDB
thread count are also declared there. Setting `DUCKDB_THREADS` to zero keeps
DuckDB's default. DuckDB filters country cells from Parquet,
then the bounded observations and all subsequent analysis remain inspectable
pandas DataFrames. It does not compute global agreement statistics. For large
countries, narrow `ANALYSIS_START_DATE` and `ANALYSIS_END_DATE` to keep the
pandas working set manageable.

To select all 0.25-degree ERA5 cell footprints intersecting one or more ADM0
countries, pass ISO alpha-2 or alpha-3 codes:

```bash
uv run python scripts/country_cells.py SG JPN \
  --output data/country_cells_sg_jpn.csv
```

The script uses checksum-pinned Natural Earth 10m boundaries and emits both a
28-character `cell_id` and the matching `row_id_prefix`. Boundary-intersecting
cells are retained even when their centers fall outside the country, which is
required for countries smaller than one ERA5 cell.

## Method documentation

- [Campaign result scanning and DuckDB relations](docs/results.md)
- [Single-pass pairwise statistics](docs/pairwise-statistics.md)
- [ERA5 cell-area weighting](docs/area-weighted-statistics.md)
