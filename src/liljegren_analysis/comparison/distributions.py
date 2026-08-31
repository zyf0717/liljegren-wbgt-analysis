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
        return int(round((self.maximum_c - self.minimum_c) / self.width_c))

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
        if not np.isfinite(numeric).all():
            raise ValueError("WBGT values must be finite before binning")
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

    def bin_bounds(self, bin_id: int) -> tuple[float | None, float | None]:
        """Return lower/upper bounds, with ``None`` for an open edge."""

        if bin_id == self.underflow_bin_id:
            return None, self.minimum_c
        if 1 <= bin_id <= self.regular_bin_count:
            lower = self.minimum_c + (bin_id - 1) * self.width_c
            return lower, lower + self.width_c
        if bin_id == self.overflow_bin_id:
            return self.maximum_c, None
        raise ValueError(f"Invalid WBGT bin ID: {bin_id}")


DEFAULT_WBGT_HISTOGRAM = WBGTHistogramSpec()

ABS_DIFFERENCE_BIN_SQL = """
CASE
    WHEN abs_difference_c < 2.0
        THEN floor(abs_difference_c / 0.01)::INTEGER
    WHEN abs_difference_c < 5.0
        THEN 200 + floor((abs_difference_c - 2.0) / 0.05)::INTEGER
    WHEN abs_difference_c < 20.0
        THEN 260 + floor((abs_difference_c - 5.0) / 0.25)::INTEGER
    WHEN abs_difference_c < 100.0
        THEN 320 + floor((abs_difference_c - 20.0) / 1.0)::INTEGER
    ELSE 400
END
""".strip()

def absolute_difference_bin_bounds(bin_id: int) -> tuple[float, float | None]:
    """Return lower-inclusive and upper-exclusive absolute-difference bounds."""

    if 0 <= bin_id < 200:
        return bin_id * 0.01, (bin_id + 1) * 0.01
    if 200 <= bin_id < 260:
        offset = bin_id - 200
        return 2.0 + offset * 0.05, 2.0 + (offset + 1) * 0.05
    if 260 <= bin_id < 320:
        offset = bin_id - 260
        return 5.0 + offset * 0.25, 5.0 + (offset + 1) * 0.25
    if 320 <= bin_id < 400:
        offset = bin_id - 320
        return 20.0 + offset, 21.0 + offset
    if bin_id == 400:
        return 100.0, None
    raise ValueError(f"Invalid absolute-difference bin ID: {bin_id}")


def absolute_difference_bounds_sql(bin_expression: str) -> tuple[str, str]:
    """Return SQL expressions for absolute-difference bin bounds."""

    lower = f"""
    CASE
        WHEN {bin_expression} < 200 THEN {bin_expression} * 0.01
        WHEN {bin_expression} < 260
            THEN 2.0 + ({bin_expression} - 200) * 0.05
        WHEN {bin_expression} < 320
            THEN 5.0 + ({bin_expression} - 260) * 0.25
        WHEN {bin_expression} < 400
            THEN 20.0 + ({bin_expression} - 320) * 1.0
        ELSE 100.0
    END
    """.strip()
    upper = f"""
    CASE
        WHEN {bin_expression} < 200 THEN ({bin_expression} + 1) * 0.01
        WHEN {bin_expression} < 260
            THEN 2.0 + ({bin_expression} - 199) * 0.05
        WHEN {bin_expression} < 320
            THEN 5.0 + ({bin_expression} - 259) * 0.25
        WHEN {bin_expression} < 400
            THEN 20.0 + ({bin_expression} - 319) * 1.0
        ELSE NULL
    END
    """.strip()
    return f"CAST(({lower}) AS DOUBLE)", f"CAST(({upper}) AS DOUBLE)"
