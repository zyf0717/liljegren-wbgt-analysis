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

The production Polars plan builds one wide base relation per date, including
all three quantities and the ERA5 latitude weight. Pairwise scope uses one
implementation self-join. Complete-case scope selects the configured six
implementations, aggregates their 18 values once, and expands the 15 pairs only
after the 6:1 reduction. The grouped aggregation then emits:

- exact unweighted and cell-area-weighted sufficient sums and extrema;
- a fixed-bin absolute-difference histogram for annual P50/P95/P99;
- independent per-quantity status and value-presence coverage.

Daily summaries are atomically published as independent Parquet date
partitions. Interrupted runs leave no complete manifest, and reruns replace a
date idempotently. The annual notebook reads only these compact summaries; it
never reconstructs observation-level annual differences.

The runner supports two populations:

- `pairwise`: each pair uses rows valid for those two implementations;
- `all-implementations`: every pair among `argonne-liljegren-c`, `heatstress`,
  `kong-huber-pywbgt`, `kwodzicki-pywbgt`, `meteor`, and `thermofeel` uses the
  same rows valid across all six.

Both row- and area-weighted statistics are always produced together.

## Repository structure

- `notebooks/` — interactive daily execution and annual paper analysis.
- `scripts/` — repeatable headless daily processing.
- `src/liljegren_analysis/country/` — country boundaries, grid cells, and
  observation extraction.
- `src/liljegren_analysis/comparison/` — Polars pairwise aggregation, Parquet persistence,
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

The default complete-case run reads the sibling HeatStressBench repository's
`.hsb/runs/wbgt-comparison-20260902-150355`
and resumes
`data/pairwise/6-targets/all-implementations/schema-v7/dates/<date>/`:

```bash
uv run python scripts/daily_pairwise.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
```

Use pair-specific support explicitly when required:

```bash
uv run python scripts/daily_pairwise.py \
  --sample-scope pairwise
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
  --implementation argonne-liljegren-c \
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
implementations. By default it reads the six report targets and WBGT, Tg, and
Tnwb together once per date to construct a strict common-validity mask. Repeat
`--implementation` to override that target set. Replay then scans all three
quantities once per implementation/date and updates three concurrent
disk-backed histograms:

```bash
uv run python scripts/compute_joint_cellwise_temperature_quantiles.py \
  --hsb-run /path/to/hsb-run \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --output-root data/cellwise_temperature_quantiles/2025-joint-v1 \
  --date-workers 4 \
  --replay-workers 2
```

An observation enters any joint histogram only when all 18 default
implementation/quantity results have status `ok` and a finite value. Each wide
`implementations/<implementation>.parquet` contains the shared `n_joint`,
WBGT/Tg/Tnwb quantiles and bounds, and quantity-specific native-valid counts.
Daily masks are generated by independent date workers, bit-packed, and
content-hashed under `.work/joint_mask/` while a run is incomplete. Replay
workers process implementations into disjoint sets of three disk-backed
histograms. The recommended 4 date workers and 2 replay workers bound active
histogram storage to roughly 30 GiB while using this workstation's otherwise
idle cores. Only the parent process commits manifest checkpoints. Completed
outputs are idempotent; compatible masks and finalized implementation files
resume after interruption, and `.work/` is removed after all six wide files
and metadata are committed.

### Global ERA5 population weights

Aggregate LandScan Global 2024 population counts onto every 0.25-degree ERA5
cell and normalize them to global population shares:

```bash
uv run python scripts/compute_era5_population_weights.py
```

The output is `data/cellwise_population_weight/population.parquet`, with stable
`source_cell_id`, coordinates, integer `population`, and `population_weight`.
The LandScan 30-arc-second grid aligns exactly with each regular 0.25-degree
ERA5 cell; the two polar rows use their physically clipped half-height bands.
Use `--force` to replace an existing output.

Matching log paths are derived automatically. `--output-root` and `--log-file`
override the defaults. Workers compute dates independently; publication is one
atomic Parquet directory per date.

### Cellwise upper-tail implementation spread

The reference-free analysis joins the persisted implementation quantiles by
`cell_id`; it does not read daily HSB result shards. It reports P50/P90/P95/P99
cross-implementation spreads, the Argonne Liljegren-C diagnostic, extrema frequencies,
a comparison chart, and one map per quantile. Weight global statistics by equal
cell (`none`), ERA5 cell geometry (`geometry`, the default), or LandScan
population (`population`):

```bash
uv run python scripts/analyze_cellwise_quantile_spread.py \
  --input-root data/cellwise_temperature_quantiles/2025-joint-v1 \
  --output-root data/cellwise_temperature_quantiles/analysis/wbgt \
  --quantity wbgt
```

For population weighting, generate the population table first and then add
`--weighting population`; use `--population-weights-path` to override the
default `data/cellwise_population_weight/population.parquet`.

Use `--quantity tg` or `--quantity tnwb` with a distinct output directory to
analyze the other wide-product temperatures. Legacy WBGT-only products remain
readable and the original extraction API is unchanged.

### Global ISO 7243 cell-pair classification changes

Classify four annual joint-support WBGT summary values and emit every unordered
implementation pair for every complete-case cell (15 for the default six
targets). The product is
deliberately unweighted: join `cell_lookup.parquet` downstream to geometric or
LandScan population weights when calculating aggregate shares.

```bash
uv run python scripts/analyze_cellwise_iso7243_classification.py \
  --input-root data/cellwise_temperature_quantiles/2025-joint-v1 \
  --output-root data/cellwise_iso7243_classification_changes \
  --thresholds-c 25 26 28 30 33
```

`cell_pair_differences.parquet` stores only `n_higher` and `n_lower`; derive
`n_changed = n_higher + n_lower` and
`n_same = n_compared - n_changed` when needed. It is written one implementation
pair at a time, without a cross join or materializing the output.

### Cellwise ISO 7243 non-consensus percentage

The separate non-consensus ETL scans the raw daily result rows. For each cell,
it records the percentage of joint-valid annual rows where the nine ISO 7243
WBGT classes are not unanimous:

```bash
uv run python scripts/analyze_cellwise_iso7243_non_consensus.py \
  --thresholds-c 25 26 28 30 33
uv run python scripts/plot_cellwise_iso7243_non_consensus.py
```

The default HSB input is the same run used by daily pairwise analysis. The
output is `data/cellwise_iso7243_non_consensus/cellwise_non_consensus.parquet`,
with `n_joint`, `n_non_consensus`, and `non_consensus_percent`.

Render 36 implementation-pair maps (each `n_changed / 4`, using a shared
dynamic P99 color scale) plus the supplementary aggregate map and unweighted,
geometric-weighted, and population-weighted `n_changed` distributions:

```bash
uv run python scripts/plot_cellwise_iso7243_classification.py
```

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
columns, including failed results. Use `--start-date`, `--end-date`, and
repeatable `--run` to change extraction scope; existing caches require explicit
`--force` replacement.

Classify the cached ERA5 country observations with any ordered WBGT threshold
profile; this step never selects cells or rescans source results:

```bash
uv run python scripts/analyze_era5_country_classification.py \
  --cache data/country/jpn/observations.parquet \
  --output-root data/country_classification/jpn \
  --start-date 2025-01-01 --end-date 2025-12-31 \
  --thresholds-c 21 25 28 31
```

For NASA POWER workloads, use the equivalent run-oriented command with optional
repeatable `--country` and `--location` filters:

```bash
uv run python scripts/analyze_nasa_power_classification.py \
  --run .hsb/runs/<run> \
  --output-root data/country_classification/sg \
  --start-date 2001-01-01 --end-date 2025-12-31 \
  --country SG --location singapore \
  --thresholds-c 31 32 33
```

Both commands write class definitions, pair and monthly transition summaries,
transition counts, and the 100 largest changed observations. The corresponding
notebooks are thin editable frontends over these same APIs; the ERA5 notebook
consumes only the pre-extracted cache.

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

- [HSB result discovery and Arrow batches](docs/results.md)
- [Single-pass pairwise statistics](docs/pairwise-statistics.md)
- [ERA5 cell-area weighting](docs/area-weighted-statistics.md)
