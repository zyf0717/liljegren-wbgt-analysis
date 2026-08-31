# Liljegren WBGT implementation analysis

Reproducible analysis supporting a comparison of Liljegren wet-bulb globe
temperature implementations over HeatStressBench ERA5 runs. Code under `src/`
contains testable computation; notebooks provide paper-facing analysis; scripts
provide hardened headless execution.

## Analysis design

For each date, WBGT, globe temperature (Tg), and natural wet-bulb temperature
(Tnwb) are matched by run, cycle, and row identity. Agreement uses joint
validity: all three quantities must have status `ok` and a non-null value for
each selected implementation. Signed differences are
`implementation_a - implementation_b`, with names ordered lexicographically so
each unordered pair appears once.

The v5 production query materializes one wide base relation per date, including
all three quantities and the ERA5 latitude weight. Pairwise scope uses one
implementation self-join. Complete-case scope aggregates all 21 values for the
seven implementations once and expands the 21 fixed pairs only after the 7:1
reduction. The grouped aggregation then emits:

- exact unweighted and cell-area-weighted sufficient sums and extrema;
- a fixed-bin absolute-difference histogram for annual P50/P95/P99;
- independent per-quantity status and value-presence coverage.

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
- `src/liljegren_analysis/country/` — country boundaries, grid cells, and
  observation extraction.
- `src/liljegren_analysis/comparison/` — pairwise SQL, persistence,
  distributions, weighting, and pooling formulas.
- `src/liljegren_analysis/results.py` — shared HSB run result scanning.
- `tests/` — aggregation, persistence, distribution, and pooling regression
  tests.
- `docs/` — result layout and statistical method details.

## Inputs

Each supplied HeatStressBench run root must contain:

```text
results/<date>/<run>/<implementation>/results.parquet
```

HSB runs under `.hsb/` and generated files under `data/` are ignored by Git.

## Environment

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest -q
```

## Run the analysis

The default pair-specific run reads
`.hsb/runs/wbgt-comparison-20260823-181813` and writes
`data/daily_pairwise_v5.duckdb`:

```bash
uv run python scripts/daily_pairwise.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --sample-scope pairwise
```

The paper complete-case cache writes
`data/daily_pairwise_all_implementations_v5.duckdb`:

```bash
uv run python scripts/daily_pairwise.py \
  --sample-scope all-implementations
```

### Cellwise WBGT temporal quantiles

Compute one implementation at a time; repeat the command for each result
implementation. The pipeline assigns deterministic integer `cell_id` values
from validated ERA5 row-ID spatial prefixes, streams each daily shard into a
single implementation-local fixed-bin memmap, and writes only the final
Parquet product:

```bash
uv run python scripts/compute_cellwise_wbgt_quantiles.py \
  --hsb-run /path/to/hsb-run \
  --implementation liljegren-c \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --output-root data/cellwise_wbgt_quantiles
```

`grid.parquet` holds the integer/source-cell identity and coordinates;
`implementations/<implementation>.parquet` holds per-cell valid counts,
P50/P90/P95/P99 midpoint estimates, their containing-bin bounds, and explicit
underflow/overflow counts. `metadata.json` records the grid fingerprint,
input fingerprint, bin specification, run scope, and throughput. The default
raw-WBGT bins are -70 to 70 °C at 0.02 °C, with explicit edge bins. Work files
live in `.work/`, resume only from a completed daily checkpoint with matching
input metadata, and are removed after successful output persistence. A
deterministic 100-cell exact-order-statistic validation sample runs by default;
use `--exact-validation-cells 0` only when intentionally deferring it.

The joint-support temperature product is the preferred input when comparing
implementations. It reads all seven implementations and WBGT, Tg, and Tnwb
together once per date to construct a strict common-validity mask. Replay then
scans all three quantities once per implementation/date and updates three
concurrent disk-backed histograms:

```bash
uv run python scripts/compute_joint_cellwise_temperature_quantiles.py \
  --hsb-run /path/to/hsb-run \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --output-root data/cellwise_temperature_quantiles/2025-joint-v1 \
  --threads 8 \
  --memory-limit 16GB
```

An observation enters any joint histogram only when all 21
implementation/quantity results have status `ok` and a finite value. Each wide
`implementations/<implementation>.parquet` contains the shared `n_joint`,
WBGT/Tg/Tnwb quantiles and bounds, and quantity-specific native-valid counts.
Daily masks are bit-packed and content-hashed under `.work/joint_mask/` while a
run is incomplete. Three dense in-memory NumPy histograms exist during replay:
about 4.93 GB each, or 14.79 GB total, for 351,848 cells, 7,002 bins, and
annual `uint16` counters. DuckDB is opened and closed for each daily mask or
implementation replay scan, uses the configurable 16 GB default limit and
`.work/duckdb/` spill directory, and never retains an accumulating multi-date
relation. Histograms are discarded after their implementation's final Parquet
file is written; no histogram work files are persisted. Completed outputs are
idempotent; compatible masks and finalized implementation files resume after
interruption, and `.work/` is
removed after all seven wide files and metadata are committed atomically.

Matching log paths are derived automatically. `--database` and `--log-file`
override the defaults. Each date uses a fresh worker process; the controller is
the only database writer.

### Cellwise upper-tail implementation spread

The reference-free analysis joins the persisted implementation quantiles by
`cell_id`; it does not read daily HSB result shards. It reports P50/P90/P95/P99
cross-implementation spreads using the retained-grid area weights, the
Liljegren-C diagnostic, extrema frequencies, a comparison chart, and one map
per quantile:

```bash
uv run python scripts/analyze_cellwise_quantile_spread.py \
  --input-root data/cellwise_temperature_quantiles/2025-joint-v1 \
  --output-root data/cellwise_temperature_quantiles/analysis/wbgt \
  --quantity wbgt
```

Use `--quantity tg` or `--quantity tnwb` with a distinct output directory to
analyze the other wide-product temperatures. Legacy WBGT-only products remain
readable and the original extraction API is unchanged.

Open `notebooks/11-cellwise-temperature-quantiles.ipynb` to configure and run
the persisted quantile generation, then
`notebooks/31-cellwise-quantile-spread.ipynb` for the paper-facing
summary. The generation notebook invokes the same public API as the script;
the analysis notebook reads persisted outputs and does not duplicate the
calculations.

For annual analysis:

```bash
uv run jupyter lab
```

Open `notebooks/30-yearly-pairwise.ipynb`. It pools exact sums, reconstructs
histogram quantile intervals, and builds row/area-weighted matrices including
parametric limits of agreement.

Extract the reusable country-specific Parquet cache before opening the country
classification notebook:

```bash
uv run python scripts/extract_country.py JPN
```

The v2 output is `data/country/<iso3>/observations.parquet`, accompanied by
`manifest.json`. It retains all matched rows and all six qualified component
columns, including failed results. Use `--start-date`, `--end-date`, repeatable
`--run`, and `--threads` to change extraction scope; existing caches require
explicit `--force` replacement.

For raw country classification-transition analysis with editable thresholds,
open `notebooks/20-era5-country-classification-changes.ipynb`. It reads the cached
country observations rather than rescanning HSB run Parquet. Edit the
parameters cell immediately after the imports: `COUNTRY_CODE` accepts ISO
alpha-2 or alpha-3
(default `JPN`), while the dates, thresholds, and DuckDB
thread count are also declared there. Setting `DUCKDB_THREADS` to zero keeps
DuckDB's default. DuckDB filters country cells from Parquet,
then the bounded observations and all subsequent analysis remain inspectable
pandas DataFrames. It does not compute global agreement statistics. For large
countries, narrow `ANALYSIS_START_DATE` and `ANALYSIS_END_DATE` to keep the
pandas working set manageable. The notebook reports Tg/Tnwb availability but
applies heat-risk thresholds only to rows with valid WBGT.

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

- [HSB run result scanning and DuckDB relations](docs/results.md)
- [Single-pass pairwise statistics](docs/pairwise-statistics.md)
- [ERA5 cell-area weighting](docs/area-weighted-statistics.md)
