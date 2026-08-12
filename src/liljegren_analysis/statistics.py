"""Calculations for pooling grouped summary statistics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def pooled_difference_statistics(
    counts: ArrayLike,
    *,
    mean_differences: ArrayLike,
    sample_variances: ArrayLike,
    mean_absolute_differences: ArrayLike,
    root_mean_squared_differences: ArrayLike,
    min_differences: ArrayLike,
    max_differences: ArrayLike,
    max_absolute_differences: ArrayLike,
) -> dict[str, float | int]:
    """Pool mergeable difference statistics from grouped summaries.

    Inputs after ``counts`` are keyword-only to keep the similarly shaped
    daily summary arrays unambiguous. Observation-level quantiles cannot be
    reconstructed from grouped scalar quantiles and are not returned.
    """

    n = _count_array(counts)
    means = _finite_array(mean_differences, "mean_differences", length=len(n))
    absolute_means = _nonnegative_array(
        mean_absolute_differences,
        "mean_absolute_differences",
        length=len(n),
    )
    root_mean_squares = _nonnegative_array(
        root_mean_squared_differences,
        "root_mean_squared_differences",
        length=len(n),
    )
    minima = _finite_array(min_differences, "min_differences", length=len(n))
    maxima = _finite_array(max_differences, "max_differences", length=len(n))
    absolute_maxima = _nonnegative_array(
        max_absolute_differences,
        "max_absolute_differences",
        length=len(n),
    )

    total = int(n.sum())
    mean = _pooled_mean(n, means)
    variance = _pooled_sample_variance(n, means, sample_variances)
    mean_absolute_difference = float(np.dot(n, absolute_means) / total)
    root_mean_squared_difference = float(
        np.sqrt(np.dot(n, np.square(root_mean_squares)) / total)
    )

    return {
        "n": total,
        "mean_difference_c": mean,
        "variance_difference": variance,
        "stddev_difference_c": float(np.sqrt(variance)),
        "mean_absolute_difference_c": mean_absolute_difference,
        "rmse_c": root_mean_squared_difference,
        "min_difference_c": float(minima.min()),
        "max_difference_c": float(maxima.max()),
        "max_absolute_difference_c": float(absolute_maxima.max()),
    }


def _pooled_mean(counts: ArrayLike, means: ArrayLike) -> float:
    """Return ``sum(n_i * mean_i) / sum(n_i)``."""

    n = _count_array(counts)
    mean = _finite_array(means, "means", length=len(n))
    return float(np.dot(n, mean) / n.sum())


def _pooled_sample_variance(
    counts: ArrayLike,
    means: ArrayLike,
    sample_variances: ArrayLike,
) -> float:
    """Pool sample variances, including within- and between-group variation.

    Given total count ``N`` and pooled mean ``mean``, this computes::

        M2 = sum((n_i - 1) * variance_i + n_i * (mean_i - mean)**2)
        variance = M2 / (N - 1)

    ``NaN`` is returned when the total count is less than two.
    """

    n = _count_array(counts)
    mean = _finite_array(means, "means", length=len(n))
    variance = _sample_moment_array(
        sample_variances, n, "sample_variances", nonnegative=True
    )
    total = n.sum()
    if total < 2:
        return float("nan")

    overall_mean = float(np.dot(n, mean) / total)
    within = np.zeros_like(n)
    multiple = n > 1
    within[multiple] = (n[multiple] - 1) * variance[multiple]
    between = n * np.square(mean - overall_mean)
    return float((within.sum() + between.sum()) / (total - 1))

def _array(values: ArrayLike, name: str, *, length: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} length does not match counts")
    return result


def _count_array(values: ArrayLike) -> np.ndarray:
    result = _array(values, "counts")
    if not np.isfinite(result).all() or (result < 1).any() or (result != np.floor(result)).any():
        raise ValueError("counts must contain positive finite integers")
    return result


def _finite_array(values: ArrayLike, name: str, *, length: int | None = None) -> np.ndarray:
    result = _array(values, name, length=length)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _nonnegative_array(
    values: ArrayLike,
    name: str,
    *,
    length: int | None = None,
) -> np.ndarray:
    result = _finite_array(values, name, length=length)
    if (result < 0).any():
        raise ValueError(f"{name} must not be negative")
    return result


def _sample_moment_array(
    values: ArrayLike,
    counts: np.ndarray,
    name: str,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    result = _array(values, name, length=len(counts))
    multiple = counts > 1
    if not np.isfinite(result[multiple]).all():
        raise ValueError(f"{name} must be finite where count is greater than one")
    if nonnegative and (result[multiple] < 0).any():
        raise ValueError(f"{name} must not be negative")
    return result
