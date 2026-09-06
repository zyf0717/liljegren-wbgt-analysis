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
  --end-date 2025-12-31 \
  --workers 4
```

The default `all-implementations` scope uses common validity across the six
configured implementations and all three quantities. Use `--sample-scope
pairwise` for pair-specific support.

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
  --date-workers 4 \
  --replay-workers 2
```

It stores a canonical grid, one wide Parquet table per implementation, and
metadata under `data/cellwise/temperature-quantiles/`. Work masks and histogram
checkpoints remain resumable until all artifacts have been atomically
published.

Analyze cross-implementation spread:

```bash
uv run python scripts/analyze_cellwise_quantile_spread.py --quantity wbgt
```

Create global population weights when population-weighted summaries are
needed:

```bash
uv run python scripts/compute_era5_population_weights.py
```

Generate ISO 7243 classification artifacts and figures:

```bash
uv run python scripts/analyze_cellwise_iso7243_classification.py \
  --thresholds-c 25 26 28 30 33
uv run python scripts/plot_cellwise_iso7243_classification.py

uv run python scripts/analyze_cellwise_iso7243_non_consensus.py \
  --thresholds-c 25 26 28 30 33
uv run python scripts/plot_cellwise_iso7243_non_consensus.py
```

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
