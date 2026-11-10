from __future__ import annotations

import unittest

from analysis.grid import (
    grid_from_source_cell_ids,
)


class CellwiseGridTests(unittest.TestCase):
    def test_grid_ids_are_deterministic_and_coordinates_are_decoded(self) -> None:
        grid = grid_from_source_cell_ids(
            [
                "era5-global-s012500-w0002500",
                "era5-global-n000000-e0000000",
                "era5-global-s012500-w0002500",
            ]
        )

        self.assertEqual(grid.table["cell_id"].to_pylist(), [0, 1])
        self.assertEqual(
            grid.table["source_cell_id"].to_pylist(),
            ["era5-global-n000000-e0000000", "era5-global-s012500-w0002500"],
        )
        self.assertEqual(grid.table["latitude"].to_pylist(), [0.0, -1.25])
        self.assertEqual(grid.table["longitude"].to_pylist(), [0.0, -0.25])
