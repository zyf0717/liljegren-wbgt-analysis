# Liljegren WBGT implementation analysis

ERA5 research workflows for comparing Liljegren wet-bulb globe temperature
implementations in HeatStressBench results. Reusable computation lives in
`analysis/`; scripts and notebooks call those functions directly.

## Setup

Use Python 3.12 or newer and uv. From the repository root:

```bash
uv sync
cp config.example.yaml config.yaml
# Edit config.yaml to point to your ERA5 run and population raster.
uv run python -m pytest -q
```

The repository is not installed as a Python package. `pyproject.toml` and
`uv.lock` manage the analysis environment. Local configuration, datasets,
external HSB runs, and generated results are ignored by Git.

Input shards follow this layout:

```text
<hsb-run>/results/<date>/<run>/<implementation>/results.parquet
```

## Configuration

`config.yaml` contains one section per workflow. The sample uses YAML anchors
to share input paths, dates, and implementation selections. Edit the relevant
section when switching between global and country analyses. Relative paths
resolve against the repository root. Date ranges include both endpoints.
Input paths, implementation selections, and quantile probabilities are explicit
settings; annual pairwise aggregation inherits its scope from the daily product.

Scripts take no command-line arguments. Run them as modules from the repository
root. They read their section, call the core function, and save effective
settings alongside the outputs. The sample writes new products under
`data/analysis/`; existing datasets and older products are not migrated or removed.

## Workflows

| Analysis | Run with `uv run python -m` | Configuration section |
| --- | --- | --- |
| Daily pairwise statistics | `scripts.daily_pairwise` | `daily_pairwise` |
| Annual statistics and workbook | `scripts.analyze_yearly_pairwise` | `yearly_pairwise` |
| Joint cellwise temperature quantiles | `scripts.compute_joint_cellwise_temperature_quantiles` | `quantiles` |
| Population on the ERA5 grid | `scripts.compute_era5_population_weights` | `population` |
| Quantile spread and coverage | `scripts.analyze_cellwise_quantile_spread` | `quantile_spread` |
| Stacked quantile-spread maps | `scripts.plot_cellwise_quantile_spread` | `plot_quantile_spread` |
| Hourly classification and non-consensus | `scripts.analyze_cellwise_iso7243_classification` | `iso7243` |
| Classification figures | `scripts.plot_cellwise_iso7243_classification` | `plot_iso7243` |
| Non-consensus map | `scripts.plot_cellwise_iso7243_non_consensus` | `plot_non_consensus` |
| Global non-consensus map stack | `scripts.plot_global_iso7243_non_consensus_stack` | `plot_non_consensus_stack` |
| Country grid-cell CSV | `scripts.country_cells` | `country_cells` |
| Country observation caches | `scripts.extract_country` | `extract_country` |
| Country classification summaries | `scripts.analyze_era5_country_classification` | `country_classification` |

Run daily pairwise analysis before annual aggregation. Compute quantiles before
spread analysis; population counts are needed for population weighting and
coverage exports. The spread script processes all configured quantities; the
plotting script renders their maps separately. Run hourly classification once
to produce both pairwise class differences and non-consensus counts, then render
the desired figures.

Country extraction writes `<output_directory>/<iso3>/observations.parquet` and
its selection manifest, including row-group routing. Country caches from earlier
revisions need a fresh extraction to provide this routing metadata. Optional
`bounds: [west, south, east, north]` clips the
country polygon and supports selections such as CONUS. To use a country cache
in either cellwise workflow, set `hsb_run_paths: null`, set `country_cache`, and
choose a separate output directory. Country maps and population denominators
follow the selected country grid.

## Notebooks and reusable functions

```bash
uv run jupyter lab
```

Each notebook loads the same YAML section as its script. The first cell permits
explicit changes to `SETTINGS` for exploration; the effective settings are saved
when the analysis succeeds. Notebook setup supports kernels started in either
the repository root or `notebooks/`.

- `00`: environment imports.
- `10`, `11`: daily pairwise statistics and joint quantiles.
- `20`: country classification summaries.
- `30`: interactive annual comparison matrices and CSV export.
- `31`: interactive implementation-pair quantile maps.

Core functions take explicit paths and do not load configuration. Workflows
that publish files return their output directory; filenames are stable and
documented alongside the analysis. Calculation functions return named tables.

```python
from analysis.comparison.workflows import run_daily_pairwise
from analysis.classification.era5 import analyze_era5_country_cache
from analysis.results import discover_results, iter_result_batches
```

Shared grid identities and area weights live in `analysis/grid.py`; input discovery
and batch reading live in `analysis/results.py`. The topic directories contain
comparison, cellwise, classification, and country calculations. HSB schemas and
identifier formats are trusted. Workflow boundaries check settings and support;
internal functions operate on the selected data directly.

For interactive Matplotlib work, load an hourly summary and build a live figure:

```python
from analysis.config import load_settings
from analysis.cellwise.iso7243_summary import summarize_cellwise_iso7243_classification
from analysis.cellwise.iso7243_plots import classification_figure

settings = load_settings("plot_iso7243")
summary = summarize_cellwise_iso7243_classification(
    settings["input_root"], population_weights_path=settings["population_weights_path"]
)
figure = classification_figure(summary)
```

Figure builders respect the active notebook backend; batch plotting entry
points save and close figures.

## Resume and recomputation

Daily comparisons and hourly classification reuse completed dates. Quantiles
reuse daily validity masks and completed implementations; an interrupted
implementation replay restarts from its beginning. Temporary files are renamed
only after successful writes. Completion metadata is written after the final
outputs. Input paths, sizes, timestamps, and analysis settings guard reuse.

Use one writer per output directory. Changing scientific settings or inputs
requires a fresh output directory or `overwrite: true`, which discards only the
workflow's named outputs and checkpoints. Set it back to `false` to resume.
Cheap summaries and plots are regenerated when invoked. Input files should stay
unchanged during a run; deliberate code changes to an algorithm require explicit
recomputation.

Global quantile replay uses disk-backed histograms and can require roughly
35 GiB of available RAM with two replay workers. Keep the configurable batch
sizes and worker limits modest; streaming and memory mapping remain necessary
for annual global data.

Statistical definitions and data interpretation:
[pairwise statistics](docs/pairwise-statistics.md),
[area weighting](docs/area-weighted-statistics.md),
[cellwise analysis](docs/cellwise-analysis.md), and
[HSB result files](docs/results.md).
