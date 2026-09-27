from __future__ import annotations

import unittest

from analysis.comparison.distributions import (
    absolute_difference_bin_bounds,
)


class DistributionBinTests(unittest.TestCase):
    def test_absolute_difference_boundaries(self) -> None:
        self.assertEqual(absolute_difference_bin_bounds(0), (0.0, 0.01))
        self.assertEqual(absolute_difference_bin_bounds(199), (1.99, 2.0))
        self.assertEqual(absolute_difference_bin_bounds(200), (2.0, 2.05))
        self.assertEqual(absolute_difference_bin_bounds(259), (4.95, 5.0))
        self.assertEqual(absolute_difference_bin_bounds(260), (5.0, 5.25))
        self.assertEqual(absolute_difference_bin_bounds(319), (19.75, 20.0))
        self.assertEqual(absolute_difference_bin_bounds(320), (20.0, 21.0))
        self.assertEqual(absolute_difference_bin_bounds(399), (99.0, 100.0))
        self.assertEqual(absolute_difference_bin_bounds(400), (100.0, None))


if __name__ == "__main__":
    unittest.main()
