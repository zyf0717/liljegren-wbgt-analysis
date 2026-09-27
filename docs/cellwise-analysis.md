# Cellwise analysis definitions

## Quantiles and spread

Temporal quantiles use strict joint support across the selected implementations
and WBGT, Tg, and Tnwb. A sample must have successful statuses and finite values
in every selected implementation and quantity. The histogram-based order statistic
is reported with its containing bin bounds; sampled exact values verify the error
against the configured bin width.

Each spread analysis also writes `implementation_extrema_frequency.csv`: one
row per implementation and extremum (`min`/`max`), with `p50` and `p95`
columns containing fractions (0–1). Shares use the selected
weighting (`geometry` by default; `weighting: none` counts cells equally)
over complete-case cells. Exact ties go to the first target in implementation
order, matching the existing Parquet extrema frequencies; each percentile
sums to 1 separately for minima and maxima.

The spread script calls `export_spread_coverage` to write a transposed `spread_coverage.xlsx` workbook with
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
Both coverage measures are independent of `weighting`. Population counts
are needed for coverage and population weighting only. Set
`population_weights_path: null` to compute geometric or unweighted spreads alone.
`summarize_quantile_spread` returns tables without writing files; the analysis
entry point publishes them, and the plotting script renders figures separately.

The population pipeline reads LandScan Global 2024 with Rasterio windows and
NumPy reductions, streaming one ERA5 latitude band at a time. Integer counts
are summed into cells centered on ERA5 grid points, extending 0.125° in each
direction, wrapping longitude at the dateline and clipping latitude at the
poles. Outputs under the configured population output directory contain raw population
counts keyed by `source_cell_id` and source provenance in `metadata.json`
(schema 3). Consumers calculate normalized weights for their analysis grid.
Use `overwrite: true` to replace existing outputs.

## Hourly classification

The analysis classifies raw **hourly WBGT only**. By default (`sample_scope: all-implementations`), a cell-hour must have an `ok`
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
`figures/non_consensus_population_affected.png`, a map of cells with an annual
all-target non-consensus rate of at least 20% of valid hours. Cell colors show
LandScan population counts on a continuous linear `rocket_r` scale, capped at the
95th percentile of population across affected cells (including zeros).
Higher populations use the darkest color, with a triangular colorbar extension
indicating saturation. Colorbar values are thousands of people per cell.
Cells below 20% are grey; zero-population affected cells use the lightest color.
Unsupported cells are omitted.
Map colorbars share narrow width settings and indicate upper saturation with a
triangular extension instead of a `≥` tick label. Percentage scales reaching
100% have no upper extension.
The global stacked-map script writes `figures/non_consensus_stacked.png` at the
same 1,635-pixel width, 220 DPI, and 12-point panel-title size as the stacked
quantile-spread figures; its height follows its two-panel content.

Use `sample_scope: pairwise` to require valid
WBGT only in A and B for each pair. Each pair then has its own `n_compared`;
missing or invalid WBGT in another target does not exclude that comparison.
Non-consensus always requires valid WBGT across all selected targets, in either
mode. The overview averages the supported pair rates equally within each cell;
it does not pool counts from different pair denominators. Plot readers detect
the scope from metadata and support both modes.
