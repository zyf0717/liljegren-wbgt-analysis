from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liljegren_analysis import (
    histogram_quantile_statistics,
    pooled_area_weighted_difference_statistics,
    pooled_difference_statistics,
    pooled_sufficient_statistics,
)


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
        "mean_difference_c": difference.mean(),
        "variance_difference": difference.var(ddof=1) if n > 1 else np.nan,
        "mean_absolute_difference_c": np.abs(difference).mean(),
        "rmse_c": np.sqrt(np.square(difference).mean()),
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

    def test_pooled_difference_statistics_match_raw_values(self) -> None:
        result = pooled_difference_statistics(
            self.daily["n"],
            mean_differences=self.daily["mean_difference_c"],
            sample_variances=self.daily["variance_difference"],
            mean_absolute_differences=self.daily["mean_absolute_difference_c"],
            root_mean_squared_differences=self.daily["rmse_c"],
            min_differences=self.daily["min_difference_c"],
            max_differences=self.daily["max_difference_c"],
            max_absolute_differences=self.daily["max_absolute_difference_c"],
        )

        self.assertEqual(result["n"], len(self.difference))
        self.assertAlmostEqual(result["mean_difference_c"], self.difference.mean())
        self.assertAlmostEqual(
            result["variance_difference"], self.difference.var(ddof=1)
        )
        self.assertAlmostEqual(
            result["stddev_difference_c"], self.difference.std(ddof=1)
        )
        self.assertAlmostEqual(
            result["mean_absolute_difference_c"], np.abs(self.difference).mean()
        )
        self.assertAlmostEqual(
            result["rmse_c"], np.sqrt(np.square(self.difference).mean())
        )
        self.assertEqual(result["min_difference_c"], self.difference.min())
        self.assertEqual(result["max_difference_c"], self.difference.max())
        self.assertEqual(
            result["max_absolute_difference_c"], np.abs(self.difference).max()
        )

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite integers"):
            pooled_difference_statistics(
                [2, 0],
                mean_differences=[1.0, 2.0],
                sample_variances=[1.0, 1.0],
                mean_absolute_differences=[1.0, 2.0],
                root_mean_squared_differences=[1.0, 2.0],
                min_differences=[1.0, 2.0],
                max_differences=[1.0, 2.0],
                max_absolute_differences=[1.0, 2.0],
            )

    def test_pooled_difference_statistics_validate_inputs(self) -> None:
        arguments = {
            "mean_differences": [0.5],
            "sample_variances": [0.5],
            "mean_absolute_differences": [0.5],
            "root_mean_squared_differences": [0.5],
            "min_differences": [0.0],
            "max_differences": [1.0],
            "max_absolute_differences": [1.0],
        }

        with self.assertRaisesRegex(ValueError, "length does not match counts"):
            pooled_difference_statistics(
                [2],
                **{**arguments, "max_absolute_differences": [1.0, 2.0]},
            )
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            pooled_difference_statistics(
                [2],
                **{**arguments, "mean_absolute_differences": [-1.0]},
            )

    def test_pooled_area_weighted_statistics_match_direct_values(self) -> None:
        groups = [
            (np.asarray([1.0, 3.0]), np.asarray([1.0, 0.5])),
            (np.asarray([-1.0, 2.0]), np.asarray([0.25, 0.75])),
        ]
        counts = [len(values) for values, _ in groups]
        weight_sums = [weights.sum() for _, weights in groups]
        means = [
            np.average(values, weights=weights)
            for values, weights in groups
        ]
        variances = [
            np.average(np.square(values - mean), weights=weights)
            for (values, weights), mean in zip(groups, means, strict=True)
        ]
        absolute_means = [
            np.average(np.abs(values), weights=weights)
            for values, weights in groups
        ]
        root_mean_squares = [
            np.sqrt(np.average(np.square(values), weights=weights))
            for values, weights in groups
        ]

        result = pooled_area_weighted_difference_statistics(
            counts,
            area_weight_sums=weight_sums,
            mean_differences=means,
            population_variances=variances,
            mean_absolute_differences=absolute_means,
            root_mean_squared_differences=root_mean_squares,
        )

        values = np.concatenate([values for values, _ in groups])
        weights = np.concatenate([weights for _, weights in groups])
        mean = np.average(values, weights=weights)
        self.assertEqual(result["n"], len(values))
        self.assertAlmostEqual(result["area_weight_sum"], weights.sum())
        self.assertAlmostEqual(
            result["area_weighted_mean_difference_c"],
            mean,
        )
        self.assertAlmostEqual(
            result["area_weighted_variance_difference"],
            np.average(np.square(values - mean), weights=weights),
        )
        self.assertAlmostEqual(
            result["area_weighted_mean_absolute_difference_c"],
            np.average(np.abs(values), weights=weights),
        )
        self.assertAlmostEqual(
            result["area_weighted_rmse_c"],
            np.sqrt(np.average(np.square(values), weights=weights)),
        )

    def test_sufficient_sums_match_direct_metrics_and_repartitioning(self) -> None:
        values = np.asarray([-1.0, 0.0, 0.15, 0.4, 2.0])
        weights = np.asarray([1.0, 0.5, 0.75, 0.25, 2.0])

        def pool(groups: list[np.ndarray]) -> dict[str, float | int]:
            return pooled_sufficient_statistics(
                [len(group) for group in groups],
                sum_differences=[values[group].sum() for group in groups],
                sum_squared_differences=[
                    np.square(values[group]).sum() for group in groups
                ],
                sum_absolute_differences=[
                    np.abs(values[group]).sum() for group in groups
                ],
                min_differences=[values[group].min() for group in groups],
                max_differences=[values[group].max() for group in groups],
                max_absolute_differences=[
                    np.abs(values[group]).max() for group in groups
                ],
                area_weight_sums=[weights[group].sum() for group in groups],
                area_weighted_sum_differences=[
                    np.dot(weights[group], values[group]) for group in groups
                ],
                area_weighted_sum_squared_differences=[
                    np.dot(weights[group], np.square(values[group]))
                    for group in groups
                ],
                area_weighted_sum_absolute_differences=[
                    np.dot(weights[group], np.abs(values[group]))
                    for group in groups
                ],
                exceedance_counts_0_1=[
                    np.count_nonzero(np.abs(values[group]) > 0.1)
                    for group in groups
                ],
                exceedance_counts_0_2=[
                    np.count_nonzero(np.abs(values[group]) > 0.2)
                    for group in groups
                ],
                exceedance_counts_0_5=[
                    np.count_nonzero(np.abs(values[group]) > 0.5)
                    for group in groups
                ],
                exceedance_area_weights_0_1=[
                    weights[group][np.abs(values[group]) > 0.1].sum()
                    for group in groups
                ],
                exceedance_area_weights_0_2=[
                    weights[group][np.abs(values[group]) > 0.2].sum()
                    for group in groups
                ],
                exceedance_area_weights_0_5=[
                    weights[group][np.abs(values[group]) > 0.5].sum()
                    for group in groups
                ],
            )

        first = pool([np.asarray([0, 1]), np.asarray([2, 3, 4])])
        second = pool(
            [np.asarray([0]), np.asarray([1, 2]), np.asarray([3, 4])]
        )
        for key in first:
            self.assertAlmostEqual(first[key], second[key])

        mean = values.mean()
        stddev = values.std(ddof=1)
        weighted_mean = np.average(values, weights=weights)
        weighted_stddev = np.sqrt(
            np.average(np.square(values - weighted_mean), weights=weights)
        )
        self.assertAlmostEqual(first["mean_difference_c"], mean)
        self.assertAlmostEqual(first["stddev_difference_c"], stddev)
        self.assertAlmostEqual(first["loa_lower_c"], mean - 1.96 * stddev)
        self.assertAlmostEqual(first["loa_upper_c"], mean + 1.96 * stddev)
        self.assertAlmostEqual(
            first["area_weighted_mean_difference_c"], weighted_mean
        )
        self.assertAlmostEqual(
            first["area_weighted_stddev_difference_c"], weighted_stddev
        )
        self.assertAlmostEqual(
            first["fraction_absolute_difference_gt_0_2_c"], 3 / 5
        )

    def test_histogram_quantiles_report_midpoint_and_interval(self) -> None:
        result = histogram_quantile_statistics(
            [0, 1, 2],
            bin_lowers=[0.0, 0.1, 0.2],
            bin_uppers=[0.1, 0.2, 0.3],
            counts=[2, 3, 5],
            area_weight_sums=[1.0, 1.0, 8.0],
            expected_count=10,
            expected_area_weight_sum=10.0,
        )
        self.assertAlmostEqual(result["median_absolute_difference_c"], 0.15)
        self.assertEqual(result["median_absolute_difference_lower_c"], 0.1)
        self.assertEqual(result["median_absolute_difference_upper_c"], 0.2)
        self.assertAlmostEqual(result["p95_absolute_difference_c"], 0.25)
        self.assertAlmostEqual(
            result["area_weighted_median_absolute_difference_c"],
            0.25,
        )
        exact_values = np.asarray([0.02, 0.08, 0.11, 0.14, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29])
        for probability, label in ((0.50, "median"), (0.95, "p95"), (0.99, "p99")):
            exact = np.quantile(exact_values, probability)
            self.assertLessEqual(
                result[f"{label}_absolute_difference_lower_c"],
                exact,
            )
            self.assertGreaterEqual(
                result[f"{label}_absolute_difference_upper_c"],
                exact,
            )

        with self.assertRaisesRegex(ValueError, "overflow bin"):
            histogram_quantile_statistics(
                [0, 400],
                bin_lowers=[0.0, 100.0],
                bin_uppers=[0.01, np.nan],
                counts=[1, 99],
                area_weight_sums=[1.0, 99.0],
                expected_count=100,
                expected_area_weight_sum=100.0,
            )


if __name__ == "__main__":
    unittest.main()
