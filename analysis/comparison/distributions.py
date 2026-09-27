"""Versioned fixed-bin summaries for pairwise and raw WBGT distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ABS_HISTOGRAM_VERSION = 1

WBGT_HISTOGRAM_VERSION = 5


@dataclass(frozen=True)
class WBGTHistogramSpec:
    """Fixed raw-WBGT bins with explicit open-ended edge bins.

    Bin ``0`` is underflow, regular bins are ``1`` through
    ``regular_bin_count``, and the final bin is overflow.  Regular bins are
    lower-inclusive and upper-exclusive, including a value exactly equal to
    ``maximum_c`` in overflow rather than silently clipping it.
    """

    minimum_c: float = -70.0
    maximum_c: float = 70.0
    width_c: float = 0.02
    version: int = WBGT_HISTOGRAM_VERSION

    def __post_init__(self) -> None:
        if not self.minimum_c < self.maximum_c:
            raise ValueError("minimum_c must be less than maximum_c")
        if self.width_c <= 0.0:
            raise ValueError("width_c must be positive")
        regular_count = (self.maximum_c - self.minimum_c) / self.width_c
        if not np.isclose(regular_count, round(regular_count), atol=1e-10):
            raise ValueError("WBGT histogram range must be divisible by width")

    @property
    def regular_bin_count(self) -> int:
        return round((self.maximum_c - self.minimum_c) / self.width_c)

    @property
    def bin_count(self) -> int:
        """Total bins, including one underflow and one overflow bin."""

        return self.regular_bin_count + 2

    @property
    def underflow_bin_id(self) -> int:
        return 0

    @property
    def overflow_bin_id(self) -> int:
        return self.bin_count - 1

    def bin_ids(self, values: np.ndarray) -> np.ndarray:
        """Map finite raw WBGT values to fixed-bin IDs without clipping."""

        numeric = np.asarray(values, dtype=np.float64)
        scaled = (numeric - self.minimum_c) / self.width_c
        # Decimal bin edges are not generally binary-representable.  Snap only
        # numerical roundoff at an otherwise exact edge before flooring, so a
        # value such as -19.98 belongs to [-19.98, -19.96), not its predecessor.
        nearest = np.rint(scaled)
        edge_roundoff = np.isclose(scaled, nearest, rtol=0.0, atol=1e-10)
        scaled[edge_roundoff] = nearest[edge_roundoff]
        result = np.floor(scaled).astype(np.int64) + 1
        result[numeric < self.minimum_c] = self.underflow_bin_id
        result[numeric >= self.maximum_c] = self.overflow_bin_id
        return result


DEFAULT_WBGT_HISTOGRAM = WBGTHistogramSpec()

# Each tuple is (lower temperature, upper temperature, width, first bin ID).
ABS_BIN_SEGMENTS = (
    (0.0, 2.0, 0.01, 0),
    (2.0, 5.0, 0.05, 200),
    (5.0, 20.0, 0.25, 260),
    (20.0, 100.0, 1.0, 320),
)


def absolute_difference_bins(values):
    """Polars expressions for bin ID and bounds, using the same intervals."""
    import polars as pl

    absolute = values.abs()
    bin_id, lower_bound, upper_bound = (
        pl.lit(400),
        pl.lit(100.0),
        pl.lit(None, dtype=pl.Float64),
    )
    index = pl.col("abs_bin_id")
    for lower, upper, width, first in reversed(ABS_BIN_SEGMENTS):
        stop = first + round((upper - lower) / width)
        bin_id = (
            pl.when(absolute < upper)
            .then(first + ((absolute - lower) / width).floor())
            .otherwise(bin_id)
        )
        lower_bound = (
            pl.when(index < stop)
            .then(lower + (index - first) * width)
            .otherwise(lower_bound)
        )
        upper_bound = (
            pl.when(index < stop)
            .then(lower + (index - first + 1) * width)
            .otherwise(upper_bound)
        )
    return bin_id.cast(pl.Int32), lower_bound, upper_bound
