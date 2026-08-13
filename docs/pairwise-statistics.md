# Single-pass daily-to-annual pairwise statistics

`compute_daily_pairwise()` performs one date-scoped scan. Pairwise scope uses a
self-join; complete-case scope aggregates the seven implementation values into
one wide row per observation and directly expands the 21 fixed pairs. Its one
`GROUPING SETS` aggregate returns pair totals and absolute-difference histogram
bins. `persist_daily_pairwise()` validates and replaces both daily partitions
in one transaction.

## Population selection

Pairwise scope uses every row valid for an individual pair. Complete-case scope
restricts input to `heatstress`, `heatstress-jl`, `heatstress-r`, `liljegren-c`,
`meteor`, `pywbgt`, and `thermofeel`, then retains only identities present once
and valid for all seven. The fixed pair expansion occurs only after this check,
so all 21 pairs have identical support without an implementation join-back.

Input roots are enumerated by `(run_id, implementation)` before scanning.
Overlapping scopes are rejected to prevent many-to-many joins.

## v4 daily tables

`daily_pairwise` contains authoritative sufficient statistics:

- `n`, `sum_difference_c`, `sum_squared_difference_c2`, and
  `sum_absolute_difference_c`;
- signed and absolute extrema;
- corresponding ERA5 area-weight totals and weighted sums;
- exact count and area mass above absolute-difference thresholds 0.1, 0.2,
  and 0.5 °C.

Floating sums use DuckDB's compensated `fsum`. Daily derived means, variances,
and scalar quantiles are intentionally absent.

`daily_pairwise_abs_histogram` stores non-empty fixed bins with count and area
mass. Bounds are lower-inclusive and upper-exclusive:

- `[0,2)` by 0.01 °C;
- `[2,5)` by 0.05 °C;
- `[5,20)` by 0.25 °C;
- `[20,100)` by 1 °C;
- overflow `[100,+∞)`.

Annual P50/P95/P99 are the midpoints of their containing bins; lower and upper
bounds are retained. An overflow percentile is an explicit error. Exact strict
threshold fractions come from the main sufficient-statistics table rather than
from histogram approximation.

## Annual pooling

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

Schema metadata records version 4, sample scope, canonical implementation-set
version, and histogram version. Old databases are never migrated or mixed with
v4 partitions.
