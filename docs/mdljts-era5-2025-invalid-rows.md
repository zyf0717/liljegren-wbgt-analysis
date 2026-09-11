# mdljts-wbgt: ERA5 global 2025 invalid-row diagnosis

Investigated 2026-09-09 against HeatStressBench run
`wbgt-comparison-20260902-150355`. The dominant failure is an upstream R
list-indexing bug that invalidates entire batches, not native solver failure.

## Annual accounting

The persisted six- and seven-target annual products both use
`sample_scope="all-implementations"`. They require non-null WBGT, Tg, and Tnwb
with status `ok` for every selected implementation at each observation identity.
Adding mdljts therefore reduces every pair's N, including pairs not involving it.

| Population | N |
| --- | ---: |
| Input observations per implementation | 3,078,453,651 |
| Six-target complete cases | 2,861,963,129 |
| Seven-target complete cases | 2,260,786,490 |
| Additional exclusions | 601,176,639 (21.0057% of six-target N) |

Aggregating all 365 seven-target `quantity_status.parquet` files gives the same
counts for each of mdljts's three quantities:

| Status | N | Percentage of inputs |
| --- | ---: | ---: |
| ok | 2,307,001,386 | 74.9403% |
| target-error | 770,545,504 | 25.0303% |
| non-convergent | 906,761 | 0.02946% |

An independent scan of all 2,920 mdljts result manifests reproduces these totals.
Every `target-error` is in shard `0007` or `0008`: both fail in full on all 365
days, with no partially `target-error` shards. At least 99.849% of the incremental
601,176,639 exclusions are attributable to these batch errors (even assigning
every mdljts non-convergent row to the incremental loss). The annual counts alone
do not identify the exact overlap with the other six implementations' failures.

## Root cause

The target's `R/wrapper.R:22` contains:

```r
out[pres > 2000 | pres < 800] <- NA
```

`out` is the named list returned by `.C`, not a data frame of observations.
The pressure predicate indexes list elements instead of rows inside vectors.
The list has 23 original elements; Tg, Tnwb, Tpsy, Twbg, and status are elements
19 through 23. A low-pressure input at position 23 therefore replaces the
**entire status vector** with a scalar logical `NA`. Low pressures at positions
19, 20, and 22 similarly destroy the component vectors. Mask entries beyond
position 23 can extend the list, but do not mask the corresponding output rows.

HeatStressBench's `mdljts_component()` rejects nonnumeric or incorrectly sized
status/value vectors and emits `target-error` for every row. This follows from
the malformed upstream result; it is not an ERA5 missing-data or pairwise-join
problem. The adapter also shares the one native status across all components,
explaining the identical coverage totals.

ERA5 rows are ordered by cell, then hour. The first 24 observations of each
affected shard are one Antarctic cell's hourly pressures. Reading the original
monthly `instantaneous.nc` surface pressure at those cells gives:

| Shard | First cell | Jan 1 pressure, hPa | Full-year pressure, hPa |
| --- | --- | --- | --- |
| 0007 | 74° S, 33.25° E | 664.504–666.530 | 637.719–681.453 |
| 0008 | 82.5° S, 15.25° W | 769.430–777.000 | 744.894–799.655 |

The first 23 pressures are all below 800 hPa on every day for both cells.
Consequently, all five output list elements are replaced with scalar `NA`,
discarding roughly 1.055 million observations per shard per day, including
subsequent cells irrespective of their own pressure or convergence.

## Recorded-image reproduction

The run's result manifests identify image
`sha256:9a1af3d6a181f66ec726b96126c0ca33712807da5c97a110b2209cbb989392e5`
(local tag `heatstressbench/mdljts-wbgt:20fab95f45cf`), target commit
`cd672a886880b67f3f27bdbf75038d8f7ff0bac2`, package version 1.2.
Inspection of `wbgt::wbgt` inside that exact image confirms the erroneous line.

Run this using `docker run --rm --network none -i --entrypoint Rscript
heatstressbench/mdljts-wbgt:20fab95f45cf -` from a shell with Docker access:

```r
source("/opt/mdljts-adapter-common.R")
check <- function(p, fn = wbgt::wbgt) {
  n <- length(p)
  r <- function(x) rep(x, n)
  x <- fn(r(2025L), r(1L), r(1L), r(12L), r(30L), r(0L), r(0L),
          r(-74), r(33.25), r(400), p, r(-15), r(75), r(4), r(10),
          r(-0.052), r(0L))
  print(table(mdljts_component(x$Twbg, x$status, n)$status))
}
check(rep(1000, 24))                    # 24 ok
check(rep(700, 24))                     # 24 target-error
p <- rep(1000, 24); p[23] <- 700
check(p)                               # 24 target-error
p <- rep(1000, 24); p[24] <- 700
check(p)                               # 24 ok: low pressure is not masked

# Diagnostic only: remove the erroneous assignment from a local function copy.
f <- wbgt::wbgt
b <- as.list(body(f))
b <- b[!vapply(b, function(e)
  identical(e, quote(out[pres > 2000 | pres < 800] <- NA)), logical(1))]
body(f) <- as.call(b)
check(rep(700, 24), f)                  # 24 ok: native C converges
```

All five outcomes above were reproduced. Removing the mask in memory yields
WBGT −11.7801 °C for the synthetic 700 hPa case; no installed code was changed.

## Implications and remediation

The existing seven-target cohort has systematic Antarctic exclusions and depends
on batch boundaries and row order. Low-pressure rows later in otherwise intact
batches can also incorrectly bypass the wrapper's intended pressure restriction.
Changing batch sizes or retrying scalar calls is therefore not a reliable fix.

Correct the upstream mask to operate on rows within output vectors, with explicit
per-row invalid status, then rebuild under a distinct recorded target/adapter
identity and rerun mdljts and its downstream summaries. Retaining the wrapper's
800–2000 hPa policy would still exclude low-pressure observations individually;
removing that policy or calling C directly is a separate target-behavior change.
Do not assume all 601 million rows would be recovered by a policy-preserving fix.

For current comparisons, retain the six-target cohort as the unaffected product.
Using pairwise scope would protect comparisons between the other implementations,
but mdljts comparisons would still carry the corrupted geographic coverage.

No target code, benchmark outputs, or existing analysis products were modified.
