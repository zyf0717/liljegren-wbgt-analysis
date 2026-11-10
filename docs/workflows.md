# Running the ERA5 analysis

The repository uses `pyproject.toml` and `uv.lock` to manage the environment.
Scripts and notebooks import `analysis/` directly from the checkout. See
[setup](../README.md#quick-start) and the [example configuration](../config.example.yaml).

## Configuration

The `run` section sets the source HeatStressBench run, inclusive dates,
implementations, population raster, and country thresholds. Relative paths
resolve against the repository root. Outputs default to
`results/<source-run-folder>/`, and boundary downloads default to
`data/administrative_boundaries/`. Set `run.output_root` and
`run.boundary_cache` to change those locations. See [result files](results.md).

Workflow sections override derived paths and settings such as worker counts,
quantile probabilities, and global classification thresholds. Downstream
defaults follow upstream output overrides. Configurations without a `run`
section specify each workflow explicitly. Use
`analysis.config.load_settings("iso7243")` to inspect a resolved section.

The example sets four workers each for daily pairwise statistics and global
hourly classification, with two Polars threads per process. OpenBLAS, OpenMP,
and MKL default to one thread. Environment settings take precedence.

## Pipeline

```bash
uv run python -m scripts.run_pipeline --config config.yaml
```

With a `run` section, list workflow names in `pipeline.steps` to include all
their generated sections, including country variants. Omit a workflow to skip
it. For explicit selection, use a mapping such as
`steps: {iso7243: [iso7243_jpn]}` under `pipeline`. An empty section list
skips that workflow. Select any prerequisites whose outputs still need to be
generated.

The runner processes the selected workflows in the order below, each section in
a separate subprocess. Timestamped directories under `pipeline.output_root`
(default `results/<hsb-run>/pipeline/`) hold stdout, stderr, and event logs.
The runner stops at the first failed step. Each workflow controls its own
checkpoint and overwrite behavior.

## Individual workflows

Run these modules from the repository root with `uv run python -m` followed
by the module name. They take their default section from `config.yaml` and save
effective settings alongside their outputs.

| Analysis | Module | Default section |
| --- | --- | --- |
| Population on the ERA5 grid | `scripts.compute_era5_population_weights` | `population` |
| Country grid-cell CSV | `scripts.country_cells` | `country_cells` |
| Country observation caches | `scripts.extract_country` | `extract_country` |
| Daily pairwise statistics | `scripts.daily_pairwise` | `daily_pairwise` |
| Annual statistics and comparison workbooks | `scripts.analyze_yearly_pairwise` | `yearly_pairwise` |
| Hourly classification and non-consensus | `scripts.analyze_cellwise_iso7243_classification` | `iso7243` |
| Classification figures | `scripts.plot_cellwise_iso7243_classification` | `plot_iso7243` |
| Non-consensus map | `scripts.plot_cellwise_iso7243_non_consensus` | `plot_non_consensus` |
| Global non-consensus map stack | `scripts.plot_global_iso7243_non_consensus_stack` | `plot_non_consensus_stack` |
| Country classification summaries | `scripts.analyze_era5_country_classification` | `country_classification` |
| Joint cellwise temperature quantiles | `scripts.compute_joint_cellwise_temperature_quantiles` | `quantiles` |
| Quantile spread and coverage | `scripts.analyze_cellwise_quantile_spread` | `quantile_spread` |
| Stacked quantile-spread maps | `scripts.plot_cellwise_quantile_spread` | `plot_quantile_spread` |

Run daily pairwise statistics before annual aggregation. Extract country
caches before country classification. Hourly classification computes both
pairwise class differences and non-consensus counts for the plotting modules.
Compute quantiles before spread analysis, and generate population counts before
using population weighting or exporting population coverage. Spread analysis
processes all configured quantities, and its plotting module renders the
saved outputs.

## Country analysis

`run.countries` maps uppercase ISO3 codes to local thresholds in degrees
Celsius. The example selects JPN and THA. Use `countries: {}` for global
workflows only. Hourly analysis and plot variants such as `iso7243_jpn` and
`plot_iso7243_tha` accept their own overrides and use
`country/<iso3>/hourly_classification/` under the output root. Country hourly
analysis defaults to one date worker. Override `iso7243_jpn.date_workers`,
for example, to change it.

The first country uses the `country_classification` section, which notebook
`20` also reads. Subsequent countries use `country_classification_<iso3>`.
Generated section suffixes and output directory names use lowercase codes.

Extraction writes `<output_directory>/<iso3>/observations.parquet` and its
selection manifest. The run configuration supplies the selected implementations
by default. Set `extract_country.implementations: null` to extract every
available target, or omit `implementations` in configurations without a `run`
section. For single-country extraction, set
`bounds: [west, south, east, north]` to clip the polygon, for example to select
CONUS. Re-extract older caches that lack row-group routing in the manifest.

To use a country cache in a cellwise workflow, set `hsb_run_paths: null`, set
`country_cache` to the observations file, and choose a separate output
directory. Country maps and population denominators use the selected grid.

To regenerate Japan's horizontal non-consensus composite from saved results,
run the following command with the actual run folder name.

```bash
run_root=results/your-hsb-run
uv run python -m scripts.plot_jpn_hourly_classification \
  --input-root "$run_root/country/jpn/hourly_classification" \
  --population-weights-path "$run_root/population/population.parquet"
```

For Thailand, use `scripts.plot_tha_hourly_classification` and the `country/tha/`
path. Both scripts verify the country and restore the original layout, including
margins, title spacing, colorbars, and whitespace crop. They write
`figures/non_consensus_stacked.png` beneath the input root by default. Use
`--output-directory` to change the destination.

## Notebooks and Python entry points

Launch with `uv run jupyter lab`. Notebook kernels can start in the repository
root or `notebooks/`. Each analysis notebook loads the same YAML section as
its corresponding script. Edit `SETTINGS` in the setup cell to override
values for an exploratory run. The notebook saves the effective settings after
successful analysis.

| Notebook prefix | Analysis |
| --- | --- |
| `00` | Environment imports |
| `10` | Daily pairwise statistics |
| `11` | Joint cellwise temperature quantiles |
| `20` | Country classification summaries |
| `30` | Interactive annual comparison matrices and CSV export |
| `31` | Interactive implementation-pair quantile maps |

Core functions take explicit paths and settings. Functions that publish
workflow outputs return the output directory. Calculation functions such as
`analyze_era5_country_cache` return named tables. Shared grid identities and
area weights live in `analysis/grid.py`, and input discovery and batch reading
live in `analysis/results.py`. Workflows validate settings and sample support
at their boundaries. See [result files](results.md) for an input-reading example.

In a notebook, load an existing hourly summary to build a live Matplotlib figure.

```python
from analysis.cellwise.iso7243_plots import classification_figure
from analysis.cellwise.iso7243_summary import summarize_cellwise_iso7243_classification
from analysis.config import load_settings

settings = load_settings("plot_iso7243")
summary = summarize_cellwise_iso7243_classification(
    settings["input_root"], population_weights_path=settings["population_weights_path"]
)
figure = classification_figure(summary)
```

Figure builders use the active notebook backend. Batch plotting entry points
save and close figures.

## Resume and recomputation

Daily comparisons and hourly classification reuse completed dates. Quantiles
reuse daily validity masks and completed implementations. An interrupted
implementation replay starts that implementation again. Workflows fingerprint
inputs using paths, sizes, and timestamps and check analysis settings before
reusing checkpoints. They publish files by renaming completed temporary files
and write completion metadata after the final outputs.

Use one writer per output directory and keep inputs stable during the run.
For changed inputs, scientific settings, or algorithms, use a fresh output
directory or set `overwrite: true` to discard that workflow's outputs and
checkpoints. Set it back to `false` to resume. Summaries and plots regenerate
when invoked. Skip completed population and extraction steps on subsequent
pipeline runs unless you intend to replace their outputs with `overwrite: true`.

Global quantile replay uses disk-backed histograms and can require roughly
35 GiB of available RAM with two replay workers. Keep batch sizes and worker
limits modest when processing annual global data.
