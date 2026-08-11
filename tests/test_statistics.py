from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from liljegren_analysis import (  # noqa: E402
    pooled_mean,
    pooled_pearson_correlation,
    pooled_sample_covariance,
    pooled_sample_variance,
)


def _sample_covariance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1))


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    return _sample_covariance(x, y) / (x.std(ddof=1) * y.std(ddof=1))


def _daily_row(result_date: str, x: list[float], y: list[float]) -> dict[str, object]:
    values_x = np.asarray(x, dtype=float)
    values_y = np.asarray(y, dtype=float)
    difference = values_x - values_y
    n = len(difference)
    return {
        "implementation_a": "a",
        "implementation_b": "b",
        "result_date": result_date,
        "n": n,
        "mean_a_c": values_x.mean(),
        "mean_b_c": values_y.mean(),
        "mean_difference_c": difference.mean(),
        "variance_a": values_x.var(ddof=1) if n > 1 else np.nan,
        "variance_b": values_y.var(ddof=1) if n > 1 else np.nan,
        "variance_difference": difference.var(ddof=1) if n > 1 else np.nan,
        "covariance_ab": _sample_covariance(values_x, values_y) if n > 1 else np.nan,
        "absolute_difference_sum_c": np.abs(difference).sum(),
        "squared_difference_sum_c": np.square(difference).sum(),
        "min_difference_c": difference.min(),
        "max_difference_c": difference.max(),
        "max_absolute_difference_c": np.abs(difference).max(),
    }


class StatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x_groups = [np.asarray([1.0, 2.0, 4.0]), np.asarray([5.0, 7.0]), np.asarray([8.0])]
        self.y_groups = [np.asarray([0.0, 2.0, 3.0]), np.asarray([6.0, 5.0]), np.asarray([7.0])]
        self.daily = pd.DataFrame(
            [
                _daily_row("2025-01-01", self.x_groups[0].tolist(), self.y_groups[0].tolist()),
                _daily_row("2025-01-02", self.x_groups[1].tolist(), self.y_groups[1].tolist()),
                _daily_row("2025-01-03", self.x_groups[2].tolist(), self.y_groups[2].tolist()),
            ]
        )
        self.x = np.concatenate(self.x_groups)
        self.y = np.concatenate(self.y_groups)
        self.difference = self.x - self.y

    def test_pooling_primitives_match_raw_values(self) -> None:
        n = self.daily["n"]
        self.assertAlmostEqual(pooled_mean(n, self.daily["mean_a_c"]), self.x.mean())
        self.assertAlmostEqual(
            pooled_sample_variance(n, self.daily["mean_a_c"], self.daily["variance_a"]),
            self.x.var(ddof=1),
        )
        self.assertAlmostEqual(
            pooled_sample_covariance(
                n,
                self.daily["mean_a_c"],
                self.daily["mean_b_c"],
                self.daily["covariance_ab"],
            ),
            _sample_covariance(self.x, self.y),
        )
        self.assertAlmostEqual(
            pooled_pearson_correlation(
                n,
                self.daily["mean_a_c"],
                self.daily["mean_b_c"],
                self.daily["variance_a"],
                self.daily["variance_b"],
                self.daily["covariance_ab"],
            ),
            _correlation(self.x, self.y),
        )

    def test_single_observation_returns_nan_sample_moments(self) -> None:
        self.assertTrue(np.isnan(pooled_sample_variance([1], [2.0], [np.nan])))
        self.assertTrue(
            np.isnan(pooled_sample_covariance([1], [2.0], [3.0], [np.nan]))
        )

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite integers"):
            pooled_mean([2, 0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
