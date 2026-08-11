# Daily-to-overall pairwise statistics

Compute row-level pairs and daily sufficient statistics in DuckDB, then pool
the small daily table in the notebook. The library exposes only the moment
calculations; grouping, column selection, and output construction remain
notebook-level decisions.

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
            left_result.value_c AS value_a_c,
            right_result.value_c AS value_b_c,
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
        avg(value_a_c) AS mean_a_c,
        avg(value_b_c) AS mean_b_c,
        avg(difference_c) AS mean_difference_c,
        var_samp(value_a_c) AS variance_a,
        var_samp(value_b_c) AS variance_b,
        var_samp(difference_c) AS variance_difference,
        covar_samp(value_a_c, value_b_c) AS covariance_ab,
        sum(abs(difference_c)) AS absolute_difference_sum_c,
        sum(difference_c * difference_c) AS squared_difference_sum_c,
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

```python
from liljegren_analysis import (
    pooled_mean,
    pooled_pearson_correlation,
    pooled_sample_covariance,
    pooled_sample_variance,
)
```

Each function accepts one-dimensional array-like inputs, including Pandas
Series. For one implementation pair selected by notebook code:

```python
pair = daily_pairwise.loc[
    (daily_pairwise["implementation_a"] == implementation_a)
    & (daily_pairwise["implementation_b"] == implementation_b)
]
n = pair["n"]

mean_difference_c = pooled_mean(n, pair["mean_difference_c"])
variance_difference = pooled_sample_variance(
    n,
    pair["mean_difference_c"],
    pair["variance_difference"],
)
covariance_ab = pooled_sample_covariance(
    n,
    pair["mean_a_c"],
    pair["mean_b_c"],
    pair["covariance_ab"],
)
pearson_r = pooled_pearson_correlation(
    n,
    pair["mean_a_c"],
    pair["mean_b_c"],
    pair["variance_a"],
    pair["variance_b"],
    pair["covariance_ab"],
)
```

The notebook can apply these calculations inside its own `groupby` loop and
construct only the required output columns. Additive and extrema metrics do
not require library helpers:

```python
total_n = pair["n"].sum()
mean_absolute_difference_c = pair["absolute_difference_sum_c"].sum() / total_n
rmse_c = (pair["squared_difference_sum_c"].sum() / total_n) ** 0.5
min_difference_c = pair["min_difference_c"].min()
max_difference_c = pair["max_difference_c"].max()
max_absolute_difference_c = pair["max_absolute_difference_c"].max()
```

For hierarchical rollups, retain the pooled counts, means, sample variances,
sample covariance, additive sums, and extrema listed below in the notebook's
output table.

## Required columns

| Column | Daily statistic |
|---|---|
| `n` | Matched row count |
| `mean_a_c`, `mean_b_c` | Matched implementation means |
| `mean_difference_c` | Mean signed difference |
| `variance_a`, `variance_b` | Sample variances of matched values |
| `variance_difference` | Sample variance of signed differences |
| `covariance_ab` | Sample covariance of matched values |
| `absolute_difference_sum_c` | Sum of absolute differences |
| `squared_difference_sum_c` | Sum of squared differences |
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

This is used for `mean_a_c`, `mean_b_c`, and `mean_difference_c`.

### Sample variance and SD

\[
M2_z = \sum_i\left[(n_i-1)s_{z,i}^2+n_i(\bar z_i-\bar z)^2\right]
\]

\[
s_z^2=\frac{M2_z}{N-1},\qquad s_z=\sqrt{s_z^2}
\]

The first term is within-group variation; the second is between-group
variation. Averaging daily SDs or variances is incorrect.

### Sample covariance and Pearson correlation

For daily sample covariance \(c_{xy,i}\):

\[
C_{xy}=\sum_i\left[(n_i-1)c_{xy,i}
+n_i(\bar x_i-\bar x)(\bar y_i-\bar y)\right]
\]

\[
\operatorname{Cov}(x,y)=\frac{C_{xy}}{N-1},\qquad
r=\frac{C_{xy}}{\sqrt{M2_xM2_y}}
\]

Daily correlations alone are insufficient for exact pooling. `r_squared` is
reported as \(r^2\), which measures squared linear association, not agreement.

### MAE and RMSE

For daily absolute-difference sum \(A_i=\sum |d|\) and squared-difference sum
\(Q_i=\sum d^2\):

\[
\operatorname{MAE}=\frac{\sum_i A_i}{N}
\]

\[
\operatorname{RMSE}=\sqrt{\frac{\sum_i Q_i}{N}}
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
