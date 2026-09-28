# Single-pass daily-to-annual pairwise statistics

`compute_daily_pairwise(date, files, ...)` accepts preselected result records
and performs a date-scoped Polars lazy scan for WBGT,
Tg, and Tnwb. A shared lazy base relation joins ERA5 latitude weights before any
fan-out and supplies both agreement and coverage summaries. Pairwise scope uses
one self-join and expands quantities afterward. Complete-case scope selects the
configured implementation inventory and aggregates all three quantities before
expanding the implementation pairs. For six implementations, this produces 15
pairs after a 6:1 reduction. `persist_daily_pairwise()` writes each daily table
atomically, then publishes the partition completion record.

## Population selection

An implementation row is agreement-eligible only when all three statuses are
`ok` and all three values are non-null. Pairwise scope joins these joint-valid
rows per implementation pair. Complete-case scope retains only identities
present exactly once for every configured implementation. Therefore each pair
has identical `n` and `area_weight_sum` across WBGT, Tg, and Tnwb. Coverage
summaries retain the independent status and value-presence counts for each
quantity.

Input roots are enumerated by `(run_id, implementation)` before scanning.
Overlapping scopes are rejected to prevent many-to-many joins.

## Parquet date tables

`pairwise.parquet`, keyed by date, quantity, and implementation pair, contains
authoritative sufficient statistics:

- `n`, `sum_difference_c`, `sum_squared_difference_c2`, and
  `sum_absolute_difference_c`;
- signed and absolute extrema;
- corresponding ERA5 area-weight totals and weighted sums;
- exact count and area mass above absolute-difference thresholds 0.1, 0.2,
  and 0.5 °C.

Inputs are promoted to float64 before Polars differences and reductions. Daily
derived means, variances, and scalar
quantiles are intentionally absent.

`abs_histogram.parquet` stores non-empty fixed bins keyed by quantity,
with count and area mass. Bounds are lower-inclusive and upper-exclusive:

- `[0,2)` by 0.01 °C;
- `[2,5)` by 0.05 °C;
- `[5,20)` by 0.25 °C;
- `[20,100)` by 1 °C;
- overflow `[100,+∞)`.

Annual P50/P95/P99 are the midpoints of their containing bins; lower and upper
bounds are retained. An overflow percentile is an explicit error. Exact strict
threshold fractions come from the main sufficient-statistics table rather than
from histogram approximation.

`quantity_status.parquet` is keyed by date, quantity, implementation, status, and
`value_present`. It stores row count and area-weight mass independently of the
joint-validity agreement filter.

## Annual pooling

`analyze_yearly_pairwise()` reads a requested range of completed date
partitions, checks pair and implementation coverage, and writes the pooled
pairwise table, quantity coverage, metadata, and two workbooks:

- `pairwise_metrics_vs_<reference>.xlsx`: metrics relative to the reference implementation.
- `pairwise_area_weighted_matrices.xlsx`: nine pairwise matrices covering area-weighted
  mean difference, mean absolute difference, and RMSE for WBGT, Tnwb, and Tg.

The matrix workbook follows the notebook's reference-then-export implementation
order. Values are in degrees Celsius; signed differences are row minus column.
Diagonals and unavailable comparisons remain blank. Excel displays two decimal
places while retaining the underlying numeric precision.
The yearly script and notebook call this function with settings from `config.yaml`.

For count (N), signed sum (S_1), squared sum (S_2), and absolute sum
(S_A):

\[
\bar d=S_1/N,\quad
s_d^2=\frac{S_2-S_1^2/N}{N-1},\quad
\operatorname{MAE}=S_A/N,\quad
\operatorname{RMSE}=\sqrt{S_2/N}.
\]

Parametric agreement limits are:

\[
\operatorname{LoA}_{lower}=\bar d-1.96s_d,\qquad
\operatorname{LoA}_{upper}=\bar d+1.96s_d.
\]

They are model-based limits, not empirical quantiles. ERA5 observations are
spatially and temporally correlated, so the limits are descriptive unless that
dependence is modeled.

`metadata.json` records sample and quantity scope, the configured implementation
cohort, and histogram definition version. Annual aggregation inherits these
settings from the daily product. The workflow resume
manifest records the selected dates and input identity.
Each completed date contains three Parquet files and a manifest written last;
incomplete directories are not valid inputs.
