from __future__ import annotations

import unittest

import numpy as np

from analysis.cellwise.quantiles import quantile_table
from analysis.comparison.distributions import DEFAULT_WBGT_HISTOGRAM
from analysis.comparison.statistics import histogram_quantile_bin_indices
from analysis.config import DEFAULT_PROBABILITIES


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
            histogram_spec=spec,
            probabilities=DEFAULT_PROBABILITIES,
            chunk_rows=1,
        )
        self.assertEqual(result["p50_lower_c"].to_pylist(), [21.0, None])
        self.assertEqual(result["p95_upper_c"].to_pylist(), [23.02, None])
        np.testing.assert_array_equal(
            histogram_quantile_bin_indices(np.zeros(3, dtype=np.uint16)),
            np.asarray([-1, -1, -1]),
        )


def test_quantile_table_owns_edge_counts_after_memmap_closes(tmp_path):
    spec = DEFAULT_WBGT_HISTOGRAM
    histogram = np.memmap(
        tmp_path / "histogram", mode="w+", dtype=np.uint16, shape=(2, spec.bin_count)
    )
    histogram[0, 0] = 65535
    histogram[0, -1] = 65535
    result = quantile_table(
        cell_ids=np.arange(2),
        histogram=histogram,
        histogram_spec=spec,
        probabilities=(0.5, 0.95),
        chunk_rows=1,
    )
    histogram._mmap.close()
    assert result["n_valid"].to_pylist() == [131070, 0]
    assert result["underflow_n"].to_pylist() == [65535, 0]
    assert result["overflow_n"].to_pylist() == [65535, 0]
    assert result["p50_c"].to_pylist() == [None, None]
    assert result["p95_lower_c"].to_pylist() == [70.0, None]
