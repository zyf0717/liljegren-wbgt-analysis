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

    def test_longitude_boundaries_and_dateline_wrap(self) -> None:
        values = np.zeros((1, 43_200), dtype=np.int32)
        # Pixel centers immediately either side of the first two cell boundaries,
        # Greenwich's boundaries, and the dateline cell's western boundary.
        columns = [0, 14, 15, 44, 45, 21_584, 21_585, 21_614, 21_615,
                   43_184, 43_185, 43_199]
        values[0, columns] = 2 ** np.arange(len(columns))
        expected = np.zeros(1_440, dtype=np.int64)
        expected[0] = 1 + 2 + 1024 + 2048
        expected[1] = 4 + 8
        expected[2] = 16
        expected[719] = 32
        expected[720] = 64 + 128
        expected[721] = 256
        expected[1439] = 512

        np.testing.assert_array_equal(population._sum_latitude_band(values), expected)

    def test_band_matches_geographic_pixel_center_assignment(self) -> None:
        rng = np.random.default_rng(7243)
        values = rng.integers(0, 100_000, size=(30, 43_200), dtype=np.int32)
        values[::3, ::17] = population.LANDSCAN_NODATA
        # Derive the reference from geographic coordinates, independently of
        # the production shift/reshape implementation.
        longitudes = -180.0 + (np.arange(values.shape[1]) + 0.5) / 120.0
        indices = np.floor((longitudes + 180.0 + 0.125) / 0.25).astype(int) % 1440
        valid = np.where(values == population.LANDSCAN_NODATA, 0, values)
        expected = np.zeros(1440, dtype=np.int64)
        np.add.at(expected, indices, valid.sum(axis=0, dtype=np.int64))

        actual = population._sum_latitude_band(values)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(int(actual.sum()), int(valid.sum(dtype=np.int64)))

    def test_band_accumulates_in_int64_and_rejects_negative_population(self) -> None:
        values = np.full((30, 60), 10_000_000, dtype=np.int32)
        np.testing.assert_array_equal(
            population._sum_latitude_band(values), [9_000_000_000, 9_000_000_000]
        )
        values[0, -1] = -1
        with self.assertRaisesRegex(ValueError, "negative population"):
            population._sum_latitude_band(values)

    def test_polar_and_regular_row_bounds_are_exact(self) -> None:
        self.assertEqual(population._landscan_row_bounds(900_000), (0, 15))
        self.assertEqual(population._landscan_row_bounds(897_500), (15, 45))
        self.assertEqual(population._landscan_row_bounds(0), (10_785, 10_815))
        self.assertEqual(population._landscan_row_bounds(-900_000), (21_585, 21_600))

    def test_latitude_bands_partition_the_raster(self) -> None:
        bounds = [population._landscan_row_bounds(lat)
                  for lat in range(900_000, -900_001, -2500)]
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 21_600)
        self.assertTrue(all(left[1] == right[0]
                            for left, right in zip(bounds, bounds[1:])))
        self.assertEqual([end - start for start, end in bounds],
                         [15] + [30] * 719 + [15])

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
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(
                metadata["population_cell_bounds"],
                "grid-point +/-0.125 degrees; longitude wraps, latitude clips",
            )
            self.assertEqual(metadata["population_total"], 4)
            self.assertEqual(metadata["era5_cell_count"], 2)


if __name__ == "__main__":
    unittest.main()
