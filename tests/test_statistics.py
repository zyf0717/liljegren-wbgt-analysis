from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.comparison.statistics import (
    histogram_quantile_statistics,
    histogram_quantiles,
    pooled_sufficient_statistics,
)


class StatisticsTests(unittest.TestCase):
    def test_sufficient_sums_match_direct_metrics_and_repartitioning(self) -> None:
        values = np.asarray([-1.0, 0.0, 0.15, 0.4, 2.0])
        weights = np.asarray([1.0, 0.5, 0.75, 0.25, 2.0])

        def pool(groups: list[np.ndarray]) -> dict[str, float | int]:
            return pooled_sufficient_statistics(
                pd.DataFrame(
                    {
                        "n": [len(group) for group in groups],
                        "sum_difference_c": [values[group].sum() for group in groups],
                        "sum_squared_difference_c2": [
                            np.square(values[group]).sum() for group in groups
                        ],
                        "sum_absolute_difference_c": [
                            np.abs(values[group]).sum() for group in groups
                        ],
                        "min_difference_c": [values[group].min() for group in groups],
                        "max_difference_c": [values[group].max() for group in groups],
                        "max_absolute_difference_c": [
                            np.abs(values[group]).max() for group in groups
                        ],
                        "area_weight_sum": [weights[group].sum() for group in groups],
                        "area_weighted_difference_sum_c": [
                            np.dot(weights[group], values[group]) for group in groups
                        ],
                        "area_weighted_squared_difference_sum_c2": [
                            np.dot(weights[group], np.square(values[group]))
                            for group in groups
                        ],
                        "area_weighted_absolute_difference_sum_c": [
                            np.dot(weights[group], np.abs(values[group]))
                            for group in groups
                        ],
                        "n_absolute_difference_gt_0_1_c": [
                            np.count_nonzero(np.abs(values[group]) > 0.1)
                            for group in groups
                        ],
                        "n_absolute_difference_gt_0_2_c": [
                            np.count_nonzero(np.abs(values[group]) > 0.2)
                            for group in groups
                        ],
                        "n_absolute_difference_gt_0_5_c": [
                            np.count_nonzero(np.abs(values[group]) > 0.5)
                            for group in groups
                        ],
                        "area_weight_absolute_difference_gt_0_1_c": [
                            weights[group][np.abs(values[group]) > 0.1].sum()
                            for group in groups
                        ],
                        "area_weight_absolute_difference_gt_0_2_c": [
                            weights[group][np.abs(values[group]) > 0.2].sum()
                            for group in groups
                        ],
                        "area_weight_absolute_difference_gt_0_5_c": [
                            weights[group][np.abs(values[group]) > 0.5].sum()
                            for group in groups
                        ],
                    }
                )
            )

        first = pool([np.asarray([0, 1]), np.asarray([2, 3, 4])])
        second = pool([np.asarray([0]), np.asarray([1, 2]), np.asarray([3, 4])])
        for key in first:
            self.assertAlmostEqual(first[key], second[key])
        self.assertEqual(first["n"], len(values))
        self.assertAlmostEqual(
            first["mean_absolute_difference_c"], np.abs(values).mean()
        )
        self.assertAlmostEqual(first["rmse_c"], np.sqrt(np.square(values).mean()))
        self.assertAlmostEqual(first["min_difference_c"], values.min())
        self.assertAlmostEqual(first["max_difference_c"], values.max())
        self.assertAlmostEqual(first["max_absolute_difference_c"], np.abs(values).max())
        self.assertAlmostEqual(
            first["area_weighted_mean_absolute_difference_c"],
            np.average(np.abs(values), weights=weights),
        )
        self.assertAlmostEqual(
            first["area_weighted_rmse_c"],
            np.sqrt(np.average(np.square(values), weights=weights)),
        )
        singleton = pool([np.asarray([0])])
        self.assertTrue(np.isnan(singleton["variance_difference"]))
        self.assertEqual(singleton["area_weighted_variance_difference"], 0)

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
        self.assertAlmostEqual(first["area_weighted_mean_difference_c"], weighted_mean)
        self.assertAlmostEqual(
            first["area_weighted_stddev_difference_c"], weighted_stddev
        )
        self.assertAlmostEqual(first["fraction_absolute_difference_gt_0_2_c"], 3 / 5)

    def test_histogram_quantiles_report_midpoint_and_interval(self) -> None:
        result = histogram_quantile_statistics(
            pd.DataFrame(
                {
                    "abs_bin_id": [0, 1, 2],
                    "abs_bin_lower_c": [0.0, 0.1, 0.2],
                    "abs_bin_upper_c": [0.1, 0.2, 0.3],
                    "n": [2, 3, 5],
                    "area_weight_sum": [1.0, 1.0, 8.0],
                }
            )
        )
        self.assertAlmostEqual(result["median_absolute_difference_c"], 0.15)
        self.assertEqual(result["median_absolute_difference_lower_c"], 0.1)
        self.assertEqual(result["median_absolute_difference_upper_c"], 0.2)
        self.assertAlmostEqual(result["p95_absolute_difference_c"], 0.25)
        self.assertAlmostEqual(
            result["area_weighted_median_absolute_difference_c"], 0.25
        )
        exact_values = np.asarray(
            [0.02, 0.08, 0.11, 0.14, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29]
        )
        for probability, label in ((0.5, "median"), (0.95, "p95"), (0.99, "p99")):
            exact = np.quantile(exact_values, probability)
            self.assertLessEqual(result[f"{label}_absolute_difference_lower_c"], exact)
            self.assertGreaterEqual(
                result[f"{label}_absolute_difference_upper_c"], exact
            )
        with self.assertRaisesRegex(ValueError, "overflow bin"):
            histogram_quantile_statistics(
                pd.DataFrame(
                    {
                        "abs_bin_id": [0, 400],
                        "abs_bin_lower_c": [0.0, 100.0],
                        "abs_bin_upper_c": [0.01, np.nan],
                        "n": [1, 99],
                        "area_weight_sum": [1.0, 99.0],
                    }
                )
            )

    def test_generic_histogram_quantiles_support_p90_and_fractional_mass(self) -> None:
        result = histogram_quantiles(
            bin_lowers=[0.0, 1.0, 2.0],
            bin_uppers=[1.0, 2.0, 3.0],
            counts=[0.25, 0.25, 1.5],
            probabilities=(0.5, 0.9),
        )
        self.assertEqual(result[0.5], (2.5, 2.0, 3.0))
        self.assertEqual(result[0.9], (2.5, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()


def test_fractional_histogram_p100_and_uint32_cumulative_counts():
    from analysis.comparison.statistics import histogram_quantile_bin_indices

    # Sequential addition of ten tenths can fall below math.fsum's total.
    np.testing.assert_array_equal(
        histogram_quantile_bin_indices(np.array([0.1] * 10 + [0.0]), (1.0,)), [9]
    )
    masses = np.array([[2**32 - 1, 2**32 - 1], [0, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(
        histogram_quantile_bin_indices(masses, (0.5, 1.0)), [[0, 1], [-1, -1]]
    )
