"""Chunked quantile extraction from cellwise fixed-bin histograms."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from ..comparison.distributions import WBGTHistogramSpec
from ..comparison.statistics import histogram_quantile_bin_indices


def quantile_table(
    *,
    cell_ids: np.ndarray,
    histogram: np.ndarray,
    valid: np.ndarray,
    underflow: np.ndarray,
    overflow: np.ndarray,
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    chunk_rows: int = 8_192,
) -> pa.Table:
    """Derive midpoint estimates and containing bounds without a global cumsum."""

    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    count = len(cell_ids)
    indices = np.full((count, len(probabilities)), -1, dtype=np.int64)
    for start in range(0, count, chunk_rows):
        stop = min(start + chunk_rows, count)
        indices[start:stop] = histogram_quantile_bin_indices(
            histogram[start:stop], probabilities
        )

    columns: dict[str, pa.Array] = {
        "cell_id": pa.array(cell_ids, type=pa.int64()),
        "n_valid": pa.array(valid),
        "underflow_n": pa.array(underflow),
        "overflow_n": pa.array(overflow),
    }
    for column, probability in enumerate(probabilities):
        label = f"p{int(probability * 100):02d}"
        estimates, lowers, uppers = _bounds_for_indices(
            indices[:, column], histogram_spec
        )
        columns[f"{label}_c"] = _nullable_float_array(estimates)
        columns[f"{label}_lower_c"] = _nullable_float_array(lowers)
        columns[f"{label}_upper_c"] = _nullable_float_array(uppers)
    return pa.table(columns)


def _bounds_for_indices(
    indices: np.ndarray,
    spec: WBGTHistogramSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.full(len(indices), np.nan, dtype=np.float64)
    upper = np.full(len(indices), np.nan, dtype=np.float64)
    regular = (indices >= 1) & (indices <= spec.regular_bin_count)
    lower[regular] = spec.minimum_c + (indices[regular] - 1) * spec.width_c
    upper[regular] = lower[regular] + spec.width_c
    underflow = indices == spec.underflow_bin_id
    overflow = indices == spec.overflow_bin_id
    upper[underflow] = spec.minimum_c
    lower[overflow] = spec.maximum_c
    estimate = (lower + upper) / 2.0
    estimate[~regular] = np.nan
    return estimate, lower, upper


def _nullable_float_array(values: np.ndarray) -> pa.Array:
    return pa.array(values, type=pa.float64(), mask=~np.isfinite(values))
