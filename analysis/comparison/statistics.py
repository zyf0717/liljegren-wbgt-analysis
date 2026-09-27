"""Calculations for pooling grouped summary statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def histogram_quantile_bin_indices(
    counts: ArrayLike,
    probabilities: tuple[float, ...] = (0.50, 0.95, 0.99),
) -> np.ndarray:
    """Return containing-bin indices for 1-D or row-wise histogram counts.

    The rank convention is the existing one: the first bin whose cumulative
    mass is at least ``probability * total_mass``.  A ``-1`` result represents
    an all-zero histogram row.  Callers processing large dense histograms can
    invoke this helper in row chunks to bound the temporary cumulative array.
    """

    masses = np.asarray(counts)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if masses.ndim == 1:
        total = math.fsum(masses)
        if total == 0:
            return np.full(len(probabilities_array), -1, dtype=np.int64)
        cumulative = np.cumsum(masses, dtype=np.float64)
        indices = np.searchsorted(cumulative, probabilities_array * total, side="left")
        # Fractional cumulative sums may finish a few ULPs below fsum(total).
        return np.minimum(indices, np.flatnonzero(masses)[-1])

    rows = masses
    cumulative_dtype = (
        np.uint64 if np.issubdtype(rows.dtype, np.integer) else np.float64
    )
    cumulative = np.cumsum(rows, axis=1, dtype=cumulative_dtype)
    totals = cumulative[:, -1]
    result = np.full((len(rows), len(probabilities_array)), -1, dtype=np.int64)
    nonzero = totals > 0
    for index, probability in enumerate(probabilities_array):
        reached = cumulative >= probability * totals[:, np.newaxis]
        result[nonzero, index] = reached[nonzero].argmax(axis=1)
    return result


def histogram_quantiles(
    *,
    bin_lowers: ArrayLike,
    bin_uppers: ArrayLike,
    counts: ArrayLike,
    probabilities: tuple[float, ...] = (0.50, 0.95, 0.99),
) -> dict[float, tuple[float | None, float | None, float | None]]:
    """Return midpoint and containing bounds for generic fixed-bin counts."""

    lowers = np.asarray(bin_lowers, dtype=np.float64)
    uppers = np.asarray(bin_uppers, dtype=np.float64)
    masses = np.asarray(counts)
    if not masses.sum() > 0:
        raise ValueError("counts must contain positive total mass")
    indices = histogram_quantile_bin_indices(masses, probabilities)
    result: dict[float, tuple[float | None, float | None, float | None]] = {}
    for probability, index in zip(probabilities, indices, strict=True):
        lower = None if np.isnan(lowers[index]) else float(lowers[index])
        upper = None if np.isnan(uppers[index]) else float(uppers[index])
        estimate = None if lower is None or upper is None else (lower + upper) / 2.0
        result[probability] = estimate, lower, upper
    return result


def pooled_sufficient_statistics(daily: pd.DataFrame) -> dict[str, float | int]:
    """Pool one quantity/pair's daily sufficient sums into annual statistics."""
    total_n = sum(map(int, daily["n"]))
    s1 = math.fsum(daily["sum_difference_c"])
    s2 = math.fsum(daily["sum_squared_difference_c2"])
    sa = math.fsum(daily["sum_absolute_difference_c"])
    mean = s1 / total_n
    sample_variance = float("nan")
    if total_n > 1:
        sample_numerator = _nonnegative_roundoff(
            s2 - s1 * s1 / total_n,
            s2,
            s1 * s1 / total_n,
            "sample variance numerator",
        )
        sample_variance = sample_numerator / (total_n - 1)
    stddev = math.sqrt(sample_variance)

    total_weight = math.fsum(daily["area_weight_sum"])
    ws1 = math.fsum(daily["area_weighted_difference_sum_c"])
    ws2 = math.fsum(daily["area_weighted_squared_difference_sum_c2"])
    wsa = math.fsum(daily["area_weighted_absolute_difference_sum_c"])
    weighted_mean = ws1 / total_weight
    weighted_variance = _nonnegative_roundoff(
        ws2 / total_weight - weighted_mean * weighted_mean,
        ws2 / total_weight,
        weighted_mean * weighted_mean,
        "area-weighted variance",
    )
    weighted_stddev = math.sqrt(weighted_variance)

    result: dict[str, float | int] = {
        "n": total_n,
        "mean_difference_c": mean,
        "variance_difference": sample_variance,
        "stddev_difference_c": stddev,
        "mean_absolute_difference_c": sa / total_n,
        "rmse_c": math.sqrt(s2 / total_n),
        "min_difference_c": float(daily["min_difference_c"].min()),
        "max_difference_c": float(daily["max_difference_c"].max()),
        "max_absolute_difference_c": float(daily["max_absolute_difference_c"].max()),
        "loa_lower_c": mean - 1.96 * stddev,
        "loa_upper_c": mean + 1.96 * stddev,
        "area_weight_sum": total_weight,
        "area_weighted_mean_difference_c": weighted_mean,
        "area_weighted_variance_difference": weighted_variance,
        "area_weighted_stddev_difference_c": weighted_stddev,
        "area_weighted_mean_absolute_difference_c": wsa / total_weight,
        "area_weighted_rmse_c": math.sqrt(ws2 / total_weight),
        "area_weighted_loa_lower_c": weighted_mean - 1.96 * weighted_stddev,
        "area_weighted_loa_upper_c": weighted_mean + 1.96 * weighted_stddev,
    }
    for label in ("0_1", "0_2", "0_5"):
        count = sum(map(int, daily[f"n_absolute_difference_gt_{label}_c"]))
        weight = math.fsum(daily[f"area_weight_absolute_difference_gt_{label}_c"])
        result[f"fraction_absolute_difference_gt_{label}_c"] = count / total_n
        result[f"area_weighted_fraction_absolute_difference_gt_{label}_c"] = (
            weight / total_weight
        )
    return result


def histogram_quantile_statistics(bins: pd.DataFrame) -> dict[str, float]:
    """P50/P95/P99 for one quantity/pair's already pooled absolute histogram."""
    bins = bins.sort_values("abs_bin_id")
    result = {}
    for column, prefix in (("n", ""), ("area_weight_sum", "area_weighted_")):
        quantiles = histogram_quantiles(
            bin_lowers=bins["abs_bin_lower_c"],
            bin_uppers=bins["abs_bin_upper_c"],
            counts=bins[column],
            probabilities=(0.50, 0.95, 0.99),
        )
        for label, (estimate, lower, upper) in zip(
            ("median", "p95", "p99"), quantiles.values(), strict=True
        ):
            if estimate is None:
                raise ValueError(
                    f"{prefix}{label} absolute difference falls in overflow bin"
                )
            stem = f"{prefix}{label}_absolute_difference"
            result.update(
                {
                    f"{stem}_c": estimate,
                    f"{stem}_lower_c": lower,
                    f"{stem}_upper_c": upper,
                }
            )
    return result


def _nonnegative_roundoff(
    value: float,
    left: float,
    right: float,
    name: str,
) -> float:
    tolerance = 64 * np.finfo(float).eps * max(abs(left), abs(right), 1.0)
    if value < -tolerance:
        raise ValueError(f"{name} is materially negative: {value}")
    return max(value, 0.0)
