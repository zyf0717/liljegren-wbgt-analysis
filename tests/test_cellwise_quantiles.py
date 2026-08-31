from __future__ import annotations

import unittest

import numpy as np

from liljegren_analysis.cellwise.quantiles import quantile_table
from liljegren_analysis.comparison.distributions import DEFAULT_WBGT_HISTOGRAM
from liljegren_analysis.comparison.statistics import histogram_quantile_bin_indices


class CellwiseQuantileTests(unittest.TestCase):
    def test_rank_selection_and_zero_observation_cell(self) -> None:
        spec = DEFAULT_WBGT_HISTOGRAM
        histogram = np.zeros((2, spec.bin_count), dtype=np.uint16)
        histogram[0, spec.bin_ids(np.asarray([20.0]))[0]] = 1
        histogram[0, spec.bin_ids(np.asarray([21.0]))[0]] = 1
        histogram[0, spec.bin_ids(np.asarray([22.0]))[0]] = 1
        histogram[0, spec.bin_ids(np.asarray([23.0]))[0]] = 1
        result = quantile_table(
            cell_ids=np.asarray([0, 1]),
            histogram=histogram,
            valid=np.asarray([4, 0], dtype=np.uint16),
            underflow=np.asarray([0, 0], dtype=np.uint16),
            overflow=np.asarray([0, 0], dtype=np.uint16),
            histogram_spec=spec,
            probabilities=(0.50, 0.90, 0.95, 0.99),
            chunk_rows=1,
        )
        self.assertEqual(result["p50_lower_c"].to_pylist(), [21.0, None])
        self.assertEqual(result["p90_lower_c"].to_pylist(), [23.0, None])
        self.assertEqual(result["p95_upper_c"].to_pylist(), [23.02, None])
        np.testing.assert_array_equal(
            histogram_quantile_bin_indices(np.zeros(3, dtype=np.uint16)),
            np.asarray([-1, -1, -1]),
        )
