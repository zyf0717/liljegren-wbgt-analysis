# ERA5 cell-area-weighted pairwise statistics

This analysis is independent of the row-weighted pipeline. It uses the ERA5
0.25-degree grid underlying the HeatStressBench results and covers the 351,848
cells selected during materialization by `lsm > 0.5`. It does not apply the
fractional land-sea-mask value or population weights.

## Cell weights

ERA5 latitude is decoded from row identities of the form
`era5-global-[ns][0-9]{6}-...`. Any other identity fails the weighted analysis.
For cell-center latitude \(\phi\) and spacing \(\delta=0.25^\circ\), define

\[
w = \frac{
    \sin(\min(90^\circ, \phi + \delta/2))
    - \sin(\max(-90^\circ, \phi - \delta/2))
}{2\sin(\delta/2)}.
\]

This is exact spherical latitude-band area normalized so an interior
equatorial cell has weight one. The common longitude-width and Earth-radius
terms cancel from every reported moment. Each hourly observation receives its
cell weight.

## Daily sufficient statistics

For difference \(d_j\), cell weight \(w_j\), and total weight
\(W=\sum_j w_j\), the daily table stores raw matched count, `area_weight_sum`,
and:

\[
\mu = \frac{\sum_j w_jd_j}{W}, \qquad
\sigma^2 = \frac{\sum_j w_j(d_j-\mu)^2}{W},
\]

\[
\operatorname{MAE}_w = \frac{\sum_jw_j|d_j|}{W}, \qquad
\operatorname{RMSE}_w =
\sqrt{\frac{\sum_jw_jd_j^2}{W}}.
\]

The variance is a descriptive weighted population variance. Weighted
quantiles and extrema are excluded.

## Yearly pooling

For daily weight \(W_i\), mean \(\mu_i\), and population variance
\(\sigma_i^2\):

\[
\mu = \frac{\sum_i W_i\mu_i}{\sum_iW_i}, \qquad
\sigma^2 = \frac{
    \sum_i W_i\left(\sigma_i^2 + (\mu_i-\mu)^2\right)
}{\sum_iW_i}.
\]

Daily weighted MAE values are averaged by \(W_i\); daily weighted RMSE values
are pooled by their squared values and \(W_i\).

Run the daily pipeline with `scripts/daily_pairwise_area_weighted.py`. It
writes `data/daily_pairwise_area_weighted_v1.duckdb`. The yearly aggregation
and matrix visualization are in `notebooks/yearly_pairwise_area_weighted.ipynb`.
