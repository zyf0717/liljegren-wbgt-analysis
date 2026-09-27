from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.cellwise.accumulator import (
    CellwiseHistogramAccumulator,
    counter_dtype,
)
from analysis.comparison.distributions import DEFAULT_WBGT_HISTOGRAM


class CellwiseHistogramTests(unittest.TestCase):
    def test_raw_wbgt_bin_boundaries_and_open_edges(self) -> None:
        spec = DEFAULT_WBGT_HISTOGRAM
        self.assertEqual(spec.regular_bin_count, 7000)
        self.assertEqual(spec.bin_count, 7002)
        np.testing.assert_array_equal(
            spec.bin_ids(np.asarray([-70.01, -70.0, -69.98, 69.98, 70.0])),
            np.asarray([0, 1, 2, 7000, 7001]),
        )
        self.assertEqual(spec.bin_bounds(0), (None, -70.0))
        self.assertEqual(spec.bin_bounds(1), (-70.0, -69.98))
        self.assertEqual(spec.bin_bounds(7001), (70.0, None))

    def test_daily_updates_equal_pooled_observation_histogram(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            accumulator = CellwiseHistogramAccumulator(
                Path(root),
                "example",
                cell_count=2,
                histogram_spec=DEFAULT_WBGT_HISTOGRAM,
                dtype=np.dtype(np.uint16),
                mode="w+",
            )
            try:
                accumulator.update(
                    np.asarray([0, 1, 0]), np.asarray([20.0, -80.0, 21.0])
                )
                accumulator.update(
                    np.asarray([0, 1, 1]), np.asarray([22.0, 80.0, 23.0])
                )
                self.assertEqual(accumulator.valid.tolist(), [3, 3])
                self.assertEqual(accumulator.underflow.tolist(), [0, 1])
                self.assertEqual(accumulator.overflow.tolist(), [0, 1])
                self.assertEqual(int(accumulator.histogram.sum()), 6)
                accumulator.flush()
                self.assertEqual(accumulator.valid.tolist(), [3, 3])
                self.assertEqual(int(accumulator.histogram.sum()), 6)
            finally:
                accumulator.close()

    def test_counter_type_and_overflow_guard(self) -> None:
        self.assertEqual(counter_dtype(8_760), np.dtype(np.uint16))
        self.assertEqual(counter_dtype(65_536), np.dtype(np.uint32))
        with tempfile.TemporaryDirectory() as root:
            accumulator = CellwiseHistogramAccumulator(
                Path(root),
                "example",
                cell_count=1,
                histogram_spec=DEFAULT_WBGT_HISTOGRAM,
                dtype=np.dtype(np.uint16),
                mode="w+",
            )
            try:
                accumulator.valid[0] = np.iinfo(np.uint16).max
                with self.assertRaisesRegex(OverflowError, "counter overflow"):
                    accumulator.update(np.asarray([0]), np.asarray([20.0]))
            finally:
                accumulator.close()
