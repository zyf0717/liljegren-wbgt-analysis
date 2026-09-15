# Liljegren WBGT implementation analysis

Research workflows for comparing Liljegren wet-bulb globe temperature
implementations over HeatStressBench results. Computation lives under `src/`,
scripts provide reproducible CLI entry points, and notebooks expose the same
APIs through editable configuration cells.

## Setup

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest -q
```

HeatStressBench runs must contain:

```text
results/<date>/<run>/<implementation>/results.parquet
```

Generated data and linked HSB runs are ignored by Git.

## Pairwise analysis

Daily summaries match WBGT, globe temperature, and natural wet-bulb
temperature by run, cycle, and row identity. Both row- and ERA5-area-weighted
statistics are written atomically as resumable date partitions.

```bash
uv run python scripts/daily_pairwise.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
```

The default `all-implementations` scope uses common validity across the six
configured implementations and all three quantities. Use `--sample-scope
pairwise` for pair-specific support. The default uses one process worker;
Polars selects its thread-pool size automatically.

Pool daily partitions into annual Parquet tables and the reference-comparison
workbook:

```bash
uv run python scripts/analyze_yearly_pairwise.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
```

See [pairwise-statistics.md](docs/pairwise-statistics.md) for definitions and
[area-weighted-statistics.md](docs/area-weighted-statistics.md) for weighting.

## Cellwise analysis

The only supported temporal-quantile product uses strict joint support across
WBGT, Tg, and Tnwb:

```bash
uv run python scripts/compute_joint_cellwise_temperature_quantiles.py \
  --hsb-run /path/to/hsb-run \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --date-workers 8 \
  --replay-workers 2
```

It stores a canonical grid, one wide Parquet table per implementation, and
metadata under `data/cellwise/6-targets/temperature-quantiles/`. Work masks and histogram
checkpoints remain resumable until all artifacts have been atomically
published.

On the 351,848-cell global grid, eight-worker mask construction measured about
4.6 GiB aggregate RSS. During two-worker replay, RSS rises as random histogram
updates fault disk-backed pages into each worker: midway through the first
implementation pair, each worker measured about 14.2 GiB RSS and the coordinator
plus workers measured about 29 GiB aggregate RSS. Most of that replay RSS is
file-backed and reclaimable, but reserve 35 GiB of available RAM for replay and
finalization transients. Increasing `--replay-workers` multiplies both the
approximately 13.8 GiB of mapped histograms per worker and storage pressure.

Create global population weights for spread coverage and population-weighted
summaries (once):

```bash
uv run python scripts/compute_era5_population_weights.py
```

Analyze cross-implementation spread:

```bash
uv run python scripts/analyze_cellwise_quantile_spread.py --quantity wbgt
```

After analyzing WBGT, Tnwb, and Tg, render separate P50 and P95 vertical map
stacks with a shared 0–10 °C scale in each figure:

```bash
uv run python scripts/plot_cellwise_quantile_spread.py
```

Each spread analysis also writes `implementation_extrema_frequency.csv`: one
row per implementation and extremum (`min`/`max`), with `p50` and `p95`
columns containing fractions (0–1). Shares use the selected
weighting (`geometry` by default; `--weighting none` counts cells equally)
over complete-case cells. Exact ties go to the first target in implementation
order, matching the existing Parquet extrema frequencies; each percentile
sums to 1 separately for minima and maxima.

Each analysis also writes a transposed `spread_coverage.xlsx` workbook with
`tropical`, `non_tropical`, and `global` sheets. Columns identify percentile
(`p50`, `p95`) and inclusive spread threshold (≥1°C through ≥8°C); metrics are
rows. Tropical cells have centers between 23.44°S and 23.44°N inclusive;
non-tropical cells are outside that band, and global includes the entire
canonical land grid. `affected_land_area_weight` reports the summed geometric
weight of affected cells, while `land_area_percent` divides that value by the
geometric weight of all canonical cells in the region. Thus it is already
area-weighted rather than a percentage of cell count.
`affected_population` sums raw population counts, and `population_percent`
divides by `total_population` and multiplies by 100. Percentages use **all canonical cells
in each region** as the denominator, including cells without complete
quantiles. The two `complete_case_*_percent` columns expose that support;
empty regions or zero population produce blank percentages. Percentage cells
are numeric and display at least one decimal place, with additional precision
for small nonzero values so they never display as zero. Counts remain exact
integers. Country inputs use
their own grid for all regions (`global` means the entire country grid).
Land-area weights are normalized ERA5-cell equivalents, not km², and display
to two decimal places while retaining full numeric precision. Threshold equality allows
1e-12°C of subtraction roundoff.
Both coverage measures are independent of `--weighting`. Population counts
are therefore required for every spread run; override the default file with
`--population-weights-path`.

The population pipeline reads LandScan Global 2024 through `rioxarray`, using
Rasterio windows to stream one ERA5 latitude band at a time. Integer counts
are summed into cells centered on ERA5 grid points, extending 0.125° in each
direction, wrapping longitude at the dateline and clipping latitude at the
poles. Outputs under `data/cellwise_population_weight/` contain raw population
counts keyed by `source_cell_id` and source provenance in `metadata.json`
(schema 3). Consumers calculate normalized weights for their analysis grid.
Use `--force` to replace existing outputs.

Generate hourly WBGT ISO 7243 classification artifacts and figures:

```bash
uv run python scripts/analyze_cellwise_iso7243_classification.py \
  --hsb-run ../HeatStressBench/.hsb/runs/wbgt-comparison-20260902-150355 \
  --start-date 2025-01-01 --end-date 2025-12-31 \
  --thresholds-c 25 26 28 30 33
uv run python scripts/plot_cellwise_iso7243_classification.py
uv run python scripts/plot_cellwise_iso7243_non_consensus.py
```

The analysis classifies raw **hourly WBGT only**. By default (`--sample-scope
all-implementations`), a cell-hour must have an `ok`
status and finite WBGT in all six selected implementations; Tg/Tnwb do not
affect its validity. Both pairwise differences and non-consensus use this same
denominator. Threshold equality enters the higher class. Pairwise `n_higher`
and `n_lower` describe class(A) relative to class(B), while maps show their sum
as a percentage of valid hours. Unsupported cells have zero counts and NaN
rates. Weighted histograms describe cell-pair rates, in five-percentage-point
bins, rather than weighting individual hours.

The classification plotting workflow also writes
`figures/population_affected.csv` and
`figures/population_affected.png`. For every implementation pair and
for all-target non-consensus, these report the LandScan population residing in
cells where the annual rate is at least 5%, 10%, 20%, or 30% of valid hours.
Each qualifying cell's population is counted once per comparison and threshold;
the statistic is not a population-hour total. The CSV has one row per comparison
and `5%`, `10%`, `20%`, and `30%` columns. Each value combines an affected
population estimate with the percentage of the complete configured population
dataset in parentheses. The unitless estimate is in billions for global products
and millions for country products. Values use two significant figures. The
grouped-bar figure displays that percentage of total population.
The workflow additionally writes
`figures/non_consensus_population_affected.png`, a line chart of total
population percentage against the four annual non-consensus rate thresholds.

Use `--sample-scope pairwise` (Python: `sample_scope="pairwise"`) to require valid
WBGT only in A and B for each pair. Each pair then has its own `n_compared`;
missing or invalid WBGT in another target does not exclude that comparison.
Non-consensus always requires valid WBGT across all selected targets, in either
mode. The overview averages the supported pair rates equally within each cell;
it does not pool counts from different pair denominators. Plot readers detect
the scope from metadata and support both modes.

Select a separate `--output-root` for pairwise results, for example
`data/cellwise/6-targets/iso7243-classification/2025-hourly-wbgt-v2-pairwise/`,
and pass that directory as `--input-root` to the plotting scripts. Switching
scope in an existing output root is rejected unless `--force` is used.
Checkpoints cannot be shared between scopes. Custom target sets must use
separate output roots from the default six-target products.

Outputs are under
`data/cellwise/6-targets/iso7243-classification/2025-hourly-wbgt-v2/`.
The previous quantile-based output is preserved and rejected by hourly plot
readers. `analyze_cellwise_iso7243_non_consensus.py` accepts the same arguments
and computes/reuses the same shared product; it is unnecessary to scan twice.

Use repeated `--implementation` options to override the six targets, and
`--date-workers` / `--batch-size` to override defaults of eight processes and
250,000 rows. Each worker decodes one implementation at a time with Arrow
threads disabled. Atomic daily checkpoints in `.work/` are retained for resume;
configuration and input fingerprints must match. Completed compatible products
are reused after integrity checks. `--force` rebuilds final artifacts and permits
a new checkpoint configuration, preserving old checkpoint generations.
Completion metadata records hourly support, thresholds, input fingerprints,
counts, timings, and worker peak RSS. A writer lock prevents concurrent runs
from publishing to the same directory.

For the 2025 six-target run, expect approximately 111 GiB of compressed selected
column reads. Reserve 10 GiB of disk for outputs/checkpoints and approximately
5–6 GiB of RAM for eight workers; actual elapsed time and peak RSS are recorded
in metadata. Annual per-cell counts are uint32, aggregated using uint64. No
individual hourly classifications or observation-pair expansion is persisted.
Pairwise mode additionally holds approximately 48 MiB of validity masks per
worker on this grid and stores one extra daily counter per cell-pair. It still
uses a single raw-data scan for both products.

## Country analysis

Build a reusable country observation cache, then run the ERA5 classifier:

```bash
uv run python scripts/extract_country.py JPN
uv run python scripts/analyze_era5_country_classification.py \
  --cache data/country/jpn/observations.parquet \
  --output-root data/country_classification/jpn \
  --start-date 2025-01-01 --end-date 2025-12-31 \
  --thresholds-c 21 25 28 31
```

Country extraction also accepts an explicit `WEST SOUTH EAST NORTH` rectangle
that clips the selected country polygon.
For the contiguous United States, excluding Alaska, Hawaii, and other
territories, build only the observation cache with:

```bash
uv run python scripts/extract_country.py USA \
  --bounds -124.848974 24.396308 -66.885444 49.384358
```

This writes `data/country/usa/observations.parquet`. The USA Natural Earth
polygon is intersected with the bounds, removing Alaska, Hawaii, territories,
and neighbouring Canadian or Mexican cells. The selected bounds are persisted
in `manifest.json`.

Country caches can feed the same cellwise pipelines as raw HSB results. Use the
same six default implementations and keep each country in a separate output
tree:

```bash
# CONUS OSHA workload limits: very heavy 25 C, heavy 26 C, moderate 28 C, light 30 C
uv run python scripts/analyze_cellwise_iso7243_classification.py \
  --country-cache data/country/usa/observations.parquet \
  --start-date 2025-01-01 --end-date 2025-12-31 \
  --thresholds-c 25 26 28 30 \
  --output-root data/country/usa/cellwise/6-targets/iso7243/2025

# JPN MoE: <21 C, [21, 25) C, [25, 28) C, [28, 31) C, >=31 C
uv run python scripts/analyze_cellwise_iso7243_classification.py \
  --country-cache data/country/jpn/observations.parquet \
  --start-date 2025-01-01 --end-date 2025-12-31 \
  --thresholds-c 21 25 28 31 \
  --output-root data/country/jpn/cellwise/6-targets/iso7243/2025

# Thailand workload limits: heavy 30 C, moderate 32 C, light 34 C
uv run python scripts/extract_country.py THA
uv run python scripts/analyze_cellwise_iso7243_classification.py \
  --country-cache data/country/tha/observations.parquet \
  --start-date 2025-01-01 --end-date 2025-12-31 \
  --thresholds-c 30 32 34 \
  --output-root data/cellwise/tha
uv run python scripts/plot_cellwise_iso7243_classification.py \
  --input-root data/cellwise/tha
uv run python scripts/plot_cellwise_iso7243_non_consensus.py \
  --input-root data/cellwise/tha

uv run python scripts/compute_joint_cellwise_temperature_quantiles.py \
  --country-cache data/country/usa/observations.parquet \
  --start-date 2025-01-01 --end-date 2025-12-31 \
  --output-root data/country/usa/cellwise/6-targets/temperature-quantiles/2025
uv run python scripts/analyze_cellwise_quantile_spread.py \
  --input-root data/country/usa/cellwise/6-targets/temperature-quantiles/2025 \
  --output-root data/country/usa/cellwise/6-targets/quantile-spread/2025
```

The non-consensus and plotting scripts accept these country cellwise roots
without a separate country implementation. Country population percentages use
the population within the cache grid; global products retain the global
population denominator. Country maps use Mercator, are fitted to the cache
extent, and overlay the pinned Natural Earth country boundary. Global maps
retain Equal Earth.

ERA5-country and NASA POWER classification default to the same six targets as
the ISO 7243 cellwise analysis. Repeat `--implementation` to replace that set.
Alongside pairwise change and transition CSVs, both workflows write
`non_consensus_summary.csv` and `monthly_non_consensus_summary.csv` on complete
all-target WBGT support. NASA POWER also writes location and, when applicable,
country non-consensus summaries.
The ERA5-country classifier streams the cache in bounded-memory daily slices;
row-level classified and paired DataFrames are therefore not materialized.

NASA POWER classification uses
`scripts/analyze_nasa_power_classification.py` with an HSB run root and explicit
date, threshold, and location scope.

## Notebooks

Launch `uv run jupyter lab`. Every workflow notebook declares its editable
configuration as uppercase globals and calls the same canonical subsystem API
as its CLI counterpart:

- `00-test-imports.ipynb` checks the installed analysis environment.
- `10` and `11` run pairwise and joint-cellwise generation.
- `20` and `21` run ERA5-country and NASA POWER classification.
- `30` renders interactive annual pairwise matrices from API-produced outputs.
- `31` renders an interactive Equal Earth implementation-pair map.

Import APIs from their owning modules, for example:

```python
from liljegren_analysis.comparison import run_daily_pairwise
from liljegren_analysis.cellwise import compute_joint_cellwise_temperature_quantiles
from liljegren_analysis.results import HSBRunResults
```

The package root intentionally contains no compatibility re-exports.

## Layout

- `src/liljegren_analysis/` — analysis APIs and shared computation.
- `scripts/` — thin CLI adapters.
- `notebooks/` — configurable non-CLI front ends and visualization.
- `tests/` — calculation, persistence, workflow, and notebook checks.
- `docs/` — data contracts and statistical methods.
