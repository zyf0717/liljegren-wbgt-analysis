# Liljegren WBGT implementation analysis

This repository contains the reproducible analysis supporting a paper that
compares implementations of the Liljegren wet-bulb globe temperature (WBGT)
model over HeatStressBench ERA5 runs. It is an analysis repository, not a
general-purpose Python library. Code under `src/` keeps the computational core
importable, testable, and safe to execute in child processes. Notebooks support
first-cut and interactive analysis, while scripts provide hardened headless
execution of established workflows.

## Analysis design

The analysis matches valid WBGT results by date, run, cycle, and row, then
computes signed differences for every implementation pair. Differences are
defined as `implementation_a - implementation_b`, with implementation names
ordered lexicographically so each unordered pair appears once.

Pairwise statistics are computed daily in DuckDB and persisted to a local
database. Each date runs in a fresh child process because reused DuckDB
processes exhibited progressive slowdown during long runs. A single controller
writes results synchronously, and date-scoped transactions make interrupted
runs resumable and reruns idempotent.

The yearly notebook pools only mergeable daily statistics: counts, means,
sample variances, MAE, RMSE, and extrema. Daily approximate quantiles are not
pooled because scalar quantiles cannot reconstruct observation-level yearly
quantiles.

## Repository structure

- `notebooks/` — first-cut analysis and interactive work that benefits from a
  GUI, including viewing tables and charts. The daily and yearly pairwise
  notebooks expose analysis choices and intermediate results directly.
- `scripts/` — hardened command-line entry points for repeatable headless runs,
  including logging, validation, concurrency, and failure handling.
- `src/liljegren_analysis/` — testable shared functions and APIs used by both
  notebooks and scripts, including result scanning, pairwise SQL,
  transactional persistence, and pooling calculations.
- `tests/` — regression tests for result discovery, pairwise aggregation,
  persistence, and pooled statistics.
- `docs/` — details of the result layout and statistical formulas.

## Inputs and generated data

The HeatStressBench campaign outputs are external inputs and are not committed.
Each campaign root supplied to the analysis must contain:

```text
results/<date>/<run>/<implementation>/results.parquet
```

Local campaign paths under `.hsb/` and generated files under `data/` are
ignored by Git. The default paths in the script and notebook identify the runs
used by this analysis and may need to be changed when reproducing it elsewhere.

## Reproducible environment

Python 3.12 or newer and
[uv](https://docs.astral.sh/uv/) are required. Create or synchronize the locked
project environment with:

```bash
uv sync
```

Run the test suite before analysis:

```bash
uv run pytest -q
```

## Run the analysis

For an interactive run, start Jupyter and execute the daily notebook before the
yearly notebook:

```bash
uv run jupyter lab
```

For command-line daily processing, provide one or more HeatStressBench campaign
roots as needed:

```bash
uv run python scripts/daily_pairwise.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --run /path/to/campaign \
  --database data/daily_pairwise.duckdb
```

The yearly notebook reads `data/daily_pairwise.duckdb`, requires complete daily
coverage for each implementation pair, and produces the pooled table used by
downstream paper analysis.

## Method documentation

- [Campaign result scanning and DuckDB relations](docs/results.md)
- [Daily-to-overall pairwise statistics](docs/pairwise-statistics.md)
