"""Calculations for pooling grouped summary statistics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def pooled_sufficient_statistics(
    counts: ArrayLike,
    *,
    sum_differences: ArrayLike,
    sum_squared_differences: ArrayLike,
    sum_absolute_differences: ArrayLike,
    min_differences: ArrayLike,
    max_differences: ArrayLike,
    max_absolute_differences: ArrayLike,
    area_weight_sums: ArrayLike,
    area_weighted_sum_differences: ArrayLike,
    area_weighted_sum_squared_differences: ArrayLike,
    area_weighted_sum_absolute_differences: ArrayLike,
    exceedance_counts_0_1: ArrayLike,
    exceedance_counts_0_2: ArrayLike,
    exceedance_counts_0_5: ArrayLike,
    exceedance_area_weights_0_1: ArrayLike,
    exceedance_area_weights_0_2: ArrayLike,
    exceedance_area_weights_0_5: ArrayLike,
) -> dict[str, float | int]:
    """Pool authoritative daily sufficient sums into annual statistics."""

    n = _count_array(counts)
    length = len(n)
    s1_values = _finite_array(sum_differences, "sum_differences", length=length)
    s2_values = _nonnegative_array(
        sum_squared_differences,
        "sum_squared_differences",
        length=length,
    )
    sa_values = _nonnegative_array(
        sum_absolute_differences,
        "sum_absolute_differences",
        length=length,
    )
    minima = _finite_array(min_differences, "min_differences", length=length)
    maxima = _finite_array(max_differences, "max_differences", length=length)
    absolute_maxima = _nonnegative_array(
        max_absolute_differences,
        "max_absolute_differences",
        length=length,
    )
    weights = _positive_array(area_weight_sums, "area_weight_sums", length=length)
    ws1_values = _finite_array(
        area_weighted_sum_differences,
        "area_weighted_sum_differences",
        length=length,
    )
    ws2_values = _nonnegative_array(
        area_weighted_sum_squared_differences,
        "area_weighted_sum_squared_differences",
        length=length,
    )
    wsa_values = _nonnegative_array(
        area_weighted_sum_absolute_differences,
        "area_weighted_sum_absolute_differences",
        length=length,
    )

    total_n = int(n.sum())
    s1 = math.fsum(s1_values)
    s2 = math.fsum(s2_values)
    sa = math.fsum(sa_values)
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

    total_weight = math.fsum(weights)
    ws1 = math.fsum(ws1_values)
    ws2 = math.fsum(ws2_values)
    wsa = math.fsum(wsa_values)
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
        "min_difference_c": float(minima.min()),
        "max_difference_c": float(maxima.max()),
        "max_absolute_difference_c": float(absolute_maxima.max()),
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
    thresholds = (
        ("0_1", exceedance_counts_0_1, exceedance_area_weights_0_1),
        ("0_2", exceedance_counts_0_2, exceedance_area_weights_0_2),
        ("0_5", exceedance_counts_0_5, exceedance_area_weights_0_5),
    )
    for label, count_values, weight_values in thresholds:
        exceedance_count = _nonnegative_integer_sum(
            count_values,
            f"exceedance_counts_{label}",
            length,
        )
        exceedance_weight = math.fsum(
            _nonnegative_array(
                weight_values,
                f"exceedance_area_weights_{label}",
                length=length,
            )
        )
        if exceedance_count > total_n or exceedance_weight > total_weight * (
            1.0 + 1e-12
        ):
            raise ValueError(f"Exceedance mass at {label} exceeds total mass")
        result[f"fraction_absolute_difference_gt_{label}_c"] = (
            exceedance_count / total_n
        )
        result[f"area_weighted_fraction_absolute_difference_gt_{label}_c"] = (
            exceedance_weight / total_weight
        )
    return result


def histogram_quantile_statistics(
    bin_ids: ArrayLike,
    *,
    bin_lowers: ArrayLike,
    bin_uppers: ArrayLike,
    counts: ArrayLike,
    area_weight_sums: ArrayLike,
    expected_count: int,
    expected_area_weight_sum: float,
) -> dict[str, float]:
    """Return midpoint estimates and containing intervals for P50/P95/P99."""

    ids = _integer_array(bin_ids, "bin_ids")
    length = len(ids)
    lowers = _finite_array(bin_lowers, "bin_lowers", length=length)
    uppers = _array(bin_uppers, "bin_uppers", length=length)
    n = _count_array(counts)
    if len(n) != length:
        raise ValueError("counts length does not match bin_ids")
    weights = _positive_array(
        area_weight_sums,
        "area_weight_sums",
        length=length,
    )

    bins: dict[int, tuple[float, float | None, int, list[float]]] = {}
    for bin_id, lower, upper, count, weight in zip(
        ids.astype(int),
        lowers,
        uppers,
        n.astype(int),
        weights,
        strict=True,
    ):
        upper_value = None if np.isnan(upper) else float(upper)
        existing = bins.get(bin_id)
        if existing is None:
            bins[bin_id] = (
                float(lower),
                upper_value,
                int(count),
                [float(weight)],
            )
        else:
            if existing[:2] != (float(lower), upper_value):
                raise ValueError(f"Inconsistent bounds for bin {bin_id}")
            bins[bin_id] = (
                existing[0],
                existing[1],
                existing[2] + int(count),
                [*existing[3], float(weight)],
            )

    ordered = [
        (bin_id, lower, upper, count, math.fsum(weight_parts))
        for bin_id, (lower, upper, count, weight_parts) in sorted(bins.items())
    ]
    total_count = sum(row[3] for row in ordered)
    total_weight = math.fsum(row[4] for row in ordered)
    if total_count != expected_count:
        raise ValueError(
            f"Histogram count {total_count} differs from expected {expected_count}"
        )
    if not math.isclose(
        total_weight,
        expected_area_weight_sum,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise ValueError("Histogram area weight differs from expected total")

    result: dict[str, float] = {}
    for label, probability in (("median", 0.50), ("p95", 0.95), ("p99", 0.99)):
        _add_histogram_quantile(
            result,
            label,
            probability,
            ordered,
            mass_index=3,
            total_mass=float(total_count),
            prefix="",
        )
        _add_histogram_quantile(
            result,
            label,
            probability,
            ordered,
            mass_index=4,
            total_mass=total_weight,
            prefix="area_weighted_",
        )
    return result


def _add_histogram_quantile(
    result: dict[str, float],
    label: str,
    probability: float,
    bins: list[tuple[int, float, float | None, int, float]],
    *,
    mass_index: int,
    total_mass: float,
    prefix: str,
) -> None:
    cumulative = 0.0
    for row in bins:
        cumulative += float(row[mass_index])
        if cumulative >= probability * total_mass:
            lower, upper = row[1], row[2]
            if upper is None:
                raise ValueError(
                    f"{prefix}{label} absolute difference falls in overflow bin"
                )
            stem = f"{prefix}{label}_absolute_difference"
            result[f"{stem}_c"] = (lower + upper) / 2.0
            result[f"{stem}_lower_c"] = lower
            result[f"{stem}_upper_c"] = upper
            return
    raise ValueError(f"Could not locate {prefix}{label} histogram quantile")


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


def _integer_array(values: ArrayLike, name: str) -> np.ndarray:
    result = _array(values, name)
    if not np.isfinite(result).all() or (result != np.floor(result)).any():
        raise ValueError(f"{name} must contain finite integers")
    return result


def _nonnegative_integer_sum(
    values: ArrayLike,
    name: str,
    length: int,
) -> int:
    result = _integer_array(values, name)
    if len(result) != length:
        raise ValueError(f"{name} length does not match counts")
    if (result < 0).any():
        raise ValueError(f"{name} must not be negative")
    return int(result.sum())


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


def pooled_area_weighted_difference_statistics(
    counts: ArrayLike,
    *,
    area_weight_sums: ArrayLike,
    mean_differences: ArrayLike,
    population_variances: ArrayLike,
    mean_absolute_differences: ArrayLike,
    root_mean_squared_differences: ArrayLike,
) -> dict[str, float | int]:
    """Pool mergeable cell-area-weighted difference statistics."""

    n = _count_array(counts)
    weights = _positive_array(
        area_weight_sums,
        "area_weight_sums",
        length=len(n),
    )
    means = _finite_array(mean_differences, "mean_differences", length=len(n))
    variances = _nonnegative_array(
        population_variances,
        "population_variances",
        length=len(n),
    )
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

    total_weight = float(weights.sum())
    mean = float(np.dot(weights, means) / total_weight)
    variance = float(
        np.dot(
            weights,
            variances + np.square(means - mean),
        )
        / total_weight
    )
    mean_absolute_difference = float(
        np.dot(weights, absolute_means) / total_weight
    )
    root_mean_squared_difference = float(
        np.sqrt(
            np.dot(weights, np.square(root_mean_squares))
            / total_weight
        )
    )

    return {
        "n": int(n.sum()),
        "area_weight_sum": total_weight,
        "area_weighted_mean_difference_c": mean,
        "area_weighted_variance_difference": variance,
        "area_weighted_stddev_difference_c": float(np.sqrt(variance)),
        "area_weighted_mean_absolute_difference_c": mean_absolute_difference,
        "area_weighted_rmse_c": root_mean_squared_difference,
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


def _positive_array(
    values: ArrayLike,
    name: str,
    *,
    length: int | None = None,
) -> np.ndarray:
    result = _finite_array(values, name, length=length)
    if (result <= 0).any():
        raise ValueError(f"{name} must be positive")
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
