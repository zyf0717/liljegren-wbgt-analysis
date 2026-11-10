# Liljegren WBGT implementation analysis

Compare Liljegren wet-bulb globe temperature (WBGT) implementations using
HeatStressBench ERA5 results. The workflows produce pairwise statistics,
cellwise temperature quantiles and spread maps, and global and country
classification summaries.

## Quick start

Use Python 3.12 or newer and uv. From the repository root, install the
dependencies and copy the example configuration.

```bash
uv sync
cp config.example.yaml config.yaml
```

In `config.yaml`, set `run.hsb_run_path` and `run.population_raster`, then
choose the date range, implementations, countries, and `pipeline.steps`.
Run the pipeline from the repository root.

```bash
uv run python -m scripts.run_pipeline --config config.yaml
```

Outputs default to `results/<hsb-run>/`. Use `uv run jupyter lab` for the
notebooks or `uv run python -m pytest -q` to run the tests.

## Documentation

- Configuration, scripts, notebooks, and checkpoint reuse. [Workflow guide](docs/workflows.md)
- Input and output paths. [Data and result files](docs/results.md)
- Daily and annual comparisons. [Pairwise statistics](docs/pairwise-statistics.md), [area weighting](docs/area-weighted-statistics.md)
- Quantiles, spread, population weighting, and hourly classification. [Cellwise analysis](docs/cellwise-analysis.md)
- 2025 ERA5 diagnostic. [mdljts-wbgt invalid rows](docs/mdljts-era5-2025-invalid-rows.md)
