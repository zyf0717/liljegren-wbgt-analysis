# Daily-to-overall pairwise statistics

Compute row-level pairs and daily sufficient statistics in DuckDB, then pool
the small daily table in the notebook. The library exposes the reusable daily
work unit, transactional persistence, and moment calculations; grouping,
column selection, and output construction remain notebook-level decisions.

The reusable daily work unit and transactional persistence API are exported as
`compute_daily_pairwise()` and `persist_daily_pairwise()`. The CLI and daily
notebook share these functions; process orchestration and filesystem defaults
remain in their respective entry points.

The row-weighted database is `data/daily_pairwise_v2.duckdb`. Its explicit
19-column schema is recorded as version 2 in `pairwise_schema`; persistence
validates that contract before deleting any rows and inserts columns by name.
The legacy 21-column database is intentionally not migrated in place.

Before scanning a date, every input root is enumerated by `(run_id,
implementation)`. Duplicate scopes are rejected because unioning overlapping
results would turn the pairwise self-join into a many-to-many join. Different
roots may still supply complementary implementations for the same run IDs.

## Daily sufficient-statistics query

```python
daily_pairwise = all_results.query(
    "results",
    """
    WITH valid_results AS (
        SELECT
            result_date,
            run_id,
            implementation,
            cycle_ordinal,
            row_id,
            value_c
        FROM results
        WHERE status = 'ok'
          AND value_c IS NOT NULL
    ),
    paired_values AS (
        SELECT
            left_result.result_date,
            left_result.implementation AS implementation_a,
            right_result.implementation AS implementation_b,
            left_result.value_c - right_result.value_c AS difference_c
        FROM valid_results AS left_result
        INNER JOIN valid_results AS right_result
            ON left_result.result_date = right_result.result_date
           AND left_result.run_id = right_result.run_id
           AND left_result.cycle_ordinal = right_result.cycle_ordinal
           AND left_result.row_id = right_result.row_id
           AND left_result.implementation < right_result.implementation
    )
    SELECT
        implementation_a,
        implementation_b,
        result_date,
        count(*) AS n,
        avg(difference_c) AS mean_difference_c,
        var_samp(difference_c) AS variance_difference,
        avg(abs(difference_c)) AS mean_absolute_difference_c,
        sqrt(avg(difference_c * difference_c)) AS rmse_c,
        min(difference_c) AS min_difference_c,
        max(difference_c) AS max_difference_c,
        max(abs(difference_c)) AS max_absolute_difference_c
    FROM paired_values
    GROUP BY implementation_a, implementation_b, result_date
    ORDER BY implementation_a, implementation_b, result_date
    """,
).df()
```

The `<` comparison emits every unordered implementation pair once. Signed
differences are always `implementation_a - implementation_b`.

## Calculation API

Pool every mergeable output metric from the persisted difference-only daily
table with the project helper:

```python
from liljegren_analysis import pooled_difference_statistics

statistics = pooled_difference_statistics(
    pair["n"],
    mean_differences=pair["mean_difference_c"],
    sample_variances=pair["variance_difference"],
    mean_absolute_differences=pair["mean_absolute_difference_c"],
    root_mean_squared_differences=pair["rmse_c"],
    min_differences=pair["min_difference_c"],
    max_differences=pair["max_difference_c"],
    max_absolute_differences=pair["max_absolute_difference_c"],
)
```

This helper returns the pooled count, difference mean/variance/standard
deviation, MAE, RMSE, and extrema. Grouping and construction of
the implementation-pair output table remain notebook responsibilities.

The notebook can apply these calculations inside its own `groupby` loop and
construct only the required output columns. Additive and extrema metrics do
not require library helpers:

```python
total_n = pair["n"].sum()
mean_absolute_difference_c = (
    pair["n"] * pair["mean_absolute_difference_c"]
).sum() / total_n
rmse_c = (
    (pair["n"] * pair["rmse_c"] ** 2).sum() / total_n
) ** 0.5
min_difference_c = pair["min_difference_c"].min()
max_difference_c = pair["max_difference_c"].max()
max_absolute_difference_c = pair["max_absolute_difference_c"].max()
```

For hierarchical rollups, retain the pooled counts, difference means, sample
variances, MAE, RMSE, and extrema listed below in the notebook's output table.

## Required columns

| Column | Daily statistic |
|---|---|
| `n` | Matched row count |
| `mean_difference_c` | Mean signed difference |
| `variance_difference` | Sample variance of signed differences |
| `mean_absolute_difference_c` | Mean absolute difference |
| `rmse_c` | Root mean squared difference |
| `min_difference_c`, `max_difference_c` | Signed extrema |
| `max_absolute_difference_c` | Maximum absolute difference |

All statistics for a row must describe the same matched observations.

## Pooling formulas

For group (i), let (n_i) be its count, \(\bar z_i\) its mean, and
\(s_{z,i}^2\) its sample variance. Let \(N=\sum_i n_i\).

### Mean

\[
\bar z = \frac{\sum_i n_i\bar z_i}{N}
\]

This is used for `mean_difference_c`.

### Sample variance and SD

\[
M2_z = \sum_i\left[(n_i-1)s_{z,i}^2+n_i(\bar z_i-\bar z)^2\right]
\]

\[
s_z^2=\frac{M2_z}{N-1},\qquad s_z=\sqrt{s_z^2}
\]

The first term is within-group variation; the second is between-group
variation. Averaging daily SDs or variances is incorrect.

### MAE and RMSE

For daily MAE \(a_i\) and RMSE \(r_i\):

\[
\operatorname{MAE}=\frac{\sum_i n_i a_i}{N}
\]

\[
\operatorname{RMSE}=\sqrt{\frac{\sum_i n_i r_i^2}{N}}
\]

### Extrema

Overall minima are the minimum of daily minima; overall maxima are the maximum
of daily maxima. The same rule applies to maximum absolute difference.

### Bias standard error and limits of agreement

For pooled signed-difference mean \(\bar d\) and SD \(s_d\):

\[
SE_{bias}=\frac{s_d}{\sqrt N}
\]

\[
LoA_{lower}=\bar d-1.96s_d,\qquad
LoA_{upper}=\bar d+1.96s_d
\]

These formulas conventionally assume independent observations and, for limits
of agreement, an approximately normal stable difference distribution. ERA5
rows are spatially and temporally correlated. Treat the reported standard
error and limits as naive descriptive quantities unless dependence is modeled.

## Quantiles are not scalar-mergeable

Daily medians, `quantile_cont` values, and scalar `approx_quantile` results
cannot be combined into observation-level overall quantiles. Do not average
daily quantiles. Exact overall quantiles require the underlying differences;
approximate overall quantiles require a mergeable sketch or common-bin
histogram.

## Exactness

The pooled moment formulas are mathematically identical to computing the
metrics from all matched rows at once. Floating-point evaluation order may
cause negligible last-bit differences. The implementation uses pooled means,
sample moments, and between-group corrections rather than subtracting large raw
sums, reducing numerical cancellation.
