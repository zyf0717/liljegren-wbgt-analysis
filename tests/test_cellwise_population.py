from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import rasterio
from affine import Affine

from liljegren_analysis.cellwise import population


class CellwisePopulationTests(unittest.TestCase):
    def test_band_sums_aligned_pixels_and_excludes_nodata(self) -> None:
        values = np.ones((30, 60), dtype=np.int32)
        values[0, 0] = population.LANDSCAN_NODATA

        totals = population._sum_latitude_band(values)

        self.assertEqual(totals.tolist(), [899, 900])

    def test_polar_and_regular_row_bounds_are_exact(self) -> None:
        self.assertEqual(population._landscan_row_bounds(900_000), (0, 15))
        self.assertEqual(population._landscan_row_bounds(897_500), (15, 45))
        self.assertEqual(population._landscan_row_bounds(0), (10_785, 10_815))
        self.assertEqual(population._landscan_row_bounds(-900_000), (21_585, 21_600))

    def test_source_cell_ids_follow_era5_convention(self) -> None:
        self.assertEqual(
            population._source_cell_id(0, 0), "era5-global-n000000-e0000000"
        )
        self.assertEqual(
            population._source_cell_id(-900_000, -1_800_000),
            "era5-global-s900000-w1800000",
        )

    def test_synthetic_geotiff_writes_parquet_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raster = root / "source.tif"
            with rasterio.open(
                raster,
                "w",
                driver="GTiff",
                width=60,
                height=30,
                count=1,
                dtype="int32",
                nodata=population.LANDSCAN_NODATA,
                crs="EPSG:4326",
                transform=Affine(1 / 120, 0.0, -180.0, 0.0, -1 / 120, 90.0),
            ) as destination:
                destination.write(np.ones((1, 30, 60), dtype=np.int32))

            with (
                patch.object(population, "_validate_landscan_raster"),
                patch.object(
                    population,
                    "_aggregate_population",
                    return_value=np.asarray([3, 1], dtype=np.int64),
                ),
                patch.object(
                    population,
                    "_era5_coordinates",
                    return_value=(
                        np.asarray([0.0, 0.0]),
                        np.asarray([0.0, 0.25]),
                        [
                            "era5-global-n000000-e0000000",
                            "era5-global-n000000-e0002500",
                        ],
                    ),
                ),
            ):
                outputs = population.compute_era5_population_weights(
                    raster, output_root=root / "output"
                )

            table = pq.read_table(outputs.population)
            self.assertEqual(table["population"].to_pylist(), [3, 1])
            self.assertEqual(table["population_weight"].to_pylist(), [0.75, 0.25])
            metadata = json.loads(outputs.metadata.read_text())
            self.assertEqual(metadata["population_total"], 4)
            self.assertEqual(metadata["era5_cell_count"], 2)


if __name__ == "__main__":
    unittest.main()
