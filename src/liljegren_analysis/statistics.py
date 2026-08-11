"""Calculations for pooling grouped summary statistics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def pooled_mean(counts: ArrayLike, means: ArrayLike) -> float:
    """Return ``sum(n_i * mean_i) / sum(n_i)``."""

    n = _count_array(counts)
    mean = _finite_array(means, "means", length=len(n))
    return float(np.dot(n, mean) / n.sum())


def pooled_sample_variance(
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


def pooled_sample_covariance(
    counts: ArrayLike,
    means_x: ArrayLike,
    means_y: ArrayLike,
    sample_covariances: ArrayLike,
) -> float:
    """Pool sample covariances with the between-group covariance correction.

    With pooled means ``mean_x`` and ``mean_y``, this computes::

        C = sum(
            (n_i - 1) * covariance_i
            + n_i * (mean_x_i - mean_x) * (mean_y_i - mean_y)
        )
        covariance = C / (N - 1)
    """

    n = _count_array(counts)
    mean_x = _finite_array(means_x, "means_x", length=len(n))
    mean_y = _finite_array(means_y, "means_y", length=len(n))
    covariance = _sample_moment_array(sample_covariances, n, "sample_covariances")
    total = n.sum()
    if total < 2:
        return float("nan")

    overall_x = float(np.dot(n, mean_x) / total)
    overall_y = float(np.dot(n, mean_y) / total)
    within = np.zeros_like(n)
    multiple = n > 1
    within[multiple] = (n[multiple] - 1) * covariance[multiple]
    between = n * (mean_x - overall_x) * (mean_y - overall_y)
    return float((within.sum() + between.sum()) / (total - 1))


def pooled_pearson_correlation(
    counts: ArrayLike,
    means_x: ArrayLike,
    means_y: ArrayLike,
    sample_variances_x: ArrayLike,
    sample_variances_y: ArrayLike,
    sample_covariances: ArrayLike,
) -> float:
    """Return Pearson correlation reconstructed from grouped moments."""

    variance_x = pooled_sample_variance(counts, means_x, sample_variances_x)
    variance_y = pooled_sample_variance(counts, means_y, sample_variances_y)
    covariance = pooled_sample_covariance(counts, means_x, means_y, sample_covariances)
    denominator = np.sqrt(variance_x * variance_y)
    return float(covariance / denominator) if denominator > 0 else float("nan")


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
