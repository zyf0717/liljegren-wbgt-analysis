# ERA5 cell-area-weighted pairwise statistics

Area-weighted statistics are emitted by the same v4 query and over the same
matched observations as unweighted statistics. There is no second full-data
pass.

## Cell weights

ERA5 latitude is decoded from row identities of the form
`era5-global-[ns][0-9]{6}-...`; invalid identities fail the combined pipeline.
For cell-center latitude \(\phi\) and spacing \(\delta=0.25^\circ\):

\[
w=\frac{
\sin(\min(90^\circ,\phi+\delta/2))-
\sin(\max(-90^\circ,\phi-\delta/2))
}{2\sin(\delta/2)}.
\]

This is exact spherical latitude-band area normalized to one at an interior
equatorial cell. Earth-radius and longitude-width factors cancel. The fixed 721
latitude weights are precomputed in memory and joined once after pair creation.

## Sufficient sums and annual metrics

The daily cache retains total weight \(W\), weighted signed sum \(WS_1\),
weighted squared sum \(WS_2\), and weighted absolute sum \(WS_A\). Annual
statistics are derived directly:

\[
\bar d_w=WS_1/W,\qquad
\sigma_w^2=WS_2/W-\bar d_w^2,
\]

\[
\operatorname{MAE}_w=WS_A/W,\qquad
\operatorname{RMSE}_w=\sqrt{WS_2/W}.
\]

Weighted limits of agreement are \(\bar d_w\pm1.96\sigma_w\). The variance is
a descriptive weighted population variance. Histogram bins also retain area
mass, enabling area-weighted P50/P95/P99 and threshold fractions without
reading ERA5 observations again.
