from __future__ import annotations

import unittest

from liljegren_analysis.cellwise.grid import (
    grid_from_source_cell_ids,
    parse_era5_cell_id,
    source_cell_id_from_row_id,
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

    def test_invalid_source_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid ERA5"):
            parse_era5_cell_id("era5-global-n000100-e0000000")
        with self.assertRaisesRegex(ValueError, "Invalid ERA5 row"):
            source_cell_id_from_row_id("not-an-era5-row")

