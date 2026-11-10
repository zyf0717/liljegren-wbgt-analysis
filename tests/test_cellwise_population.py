from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from affine import Affine
from rasterio.windows import Window

from analysis.cellwise import population
from analysis.grid import source_cell_id


class CellwisePopulationTests(unittest.TestCase):
    def test_population_alignment_preserves_order_integer_counts_and_dataset_total(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "population.parquet"
            pq.write_table(
                pa.table(
                    {
                        "source_cell_id": ["b", "extra", "a"],
                        "population": [0, 7, 9_000_000_000],
                    }
                ),
                path,
            )
            weights, total = population.load_population_weights(["a", "b"], path)
            np.testing.assert_array_equal(weights, [9_000_000_000, 0])
            self.assertEqual(weights.dtype, np.int64)
            self.assertEqual(total, 9_000_000_007)
            zero, _ = population.load_population_weights(["b"], path)
            self.assertEqual(zero.tolist(), [0])
            empty, total = population.load_population_weights([], path)
            self.assertEqual(empty.size, 0)
            self.assertEqual(total, 9_000_000_007)
            with self.assertRaisesRegex(ValueError, "missing a cell"):
                population.load_population_weights(["missing"], path)

    def test_band_sums_aligned_pixels_and_excludes_nodata(self) -> None:
        values = np.ones((30, 60), dtype=np.int32)
        values[0, 0] = population.LANDSCAN_NODATA

        totals = population._sum_latitude_band(values)

        self.assertEqual(totals.tolist(), [899, 900])

    def test_longitude_boundaries_and_dateline_wrap(self) -> None:
        values = np.zeros((1, 43_200), dtype=np.int32)
        # Pixel centers immediately either side of the first two cell boundaries,
        # Greenwich's boundaries, and the dateline cell's western boundary.
        columns = [
            0,
            14,
            15,
            44,
            45,
            21_584,
            21_585,
            21_614,
            21_615,
            43_184,
            43_185,
            43_199,
        ]
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
        # the production coarsening implementation.
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
            population._sum_latitude_band(values),
            [9_000_000_000, 9_000_000_000],
        )
        values[0, -1] = -1
        with self.assertRaisesRegex(ValueError, "negative population"):
            population._sum_latitude_band(values)

    def test_raster_validation_rejects_invalid_grids_and_counts(self) -> None:
        profile = {
            "driver": "GTiff",
            "width": 43_200,
            "height": 21_600,
            "count": 1,
            "dtype": "int32",
            "crs": "EPSG:4326",
            "nodata": population.LANDSCAN_NODATA,
            "transform": Affine(1 / 120, 0, -180, 0, -1 / 120, 90),
            "sparse_ok": True,
        }
        for changes, message in [
            ({"count": 2}, "one band"),
            ({"width": 43_199}, "global grid"),
            ({"crs": None}, "EPSG:4326"),
            ({"crs": "EPSG:3857"}, "EPSG:4326"),
            ({"transform": Affine(1 / 120, 0, -179.875, 0, -1 / 120, 90)}, "transform"),
            ({"transform": Affine(1 / 120, 1e-8, -180, 0, -1 / 120, 90)}, "transform"),
            ({"nodata": -9999}, "nodata"),
            ({"dtype": "float64"}, "int32"),
        ]:
            with (
                self.subTest(changes=changes),
                rasterio.MemoryFile() as memory,
                memory.open(**(profile | changes)) as source,
                self.assertRaisesRegex(ValueError, message),
            ):
                population._validate_landscan_raster(source)

    def test_source_cell_ids_follow_era5_convention(self) -> None:
        self.assertEqual(source_cell_id(0, 0), "era5-global-n000000-e0000000")
        self.assertEqual(
            source_cell_id(-900_000, -1_800_000),
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
                width=43_200,
                height=21_600,
                count=1,
                dtype="int32",
                nodata=population.LANDSCAN_NODATA,
                crs="EPSG:4326",
                transform=Affine(1 / 120, 0.0, -180.0, 0.0, -1 / 120, 90.0),
                tiled=True,
                compress="deflate",
                sparse_ok=True,
            ) as destination:
                # Every latitude contributes a distinct count. Both dateline
                # halves and Greenwich exercise ID/coordinate alignment too.
                counts = np.arange(1, 21_601, dtype=np.int32)
                for column in (0, 43_199, 21_600):
                    destination.write(
                        counts[:, None], 1, window=Window(column, 0, 1, len(counts))
                    )

            outputs = population.compute_era5_population_weights(
                raster, output_root=root / "output"
            )
            table = pq.read_table(outputs / "population.parquet")
            self.assertEqual(
                table.column_names,
                ["source_cell_id", "latitude", "longitude", "population"],
            )
            actual = table["population"].to_numpy().reshape(721, 1440)
            # Independent geographic assignment covers all row boundaries,
            # including 15-pixel polar rows and 30-pixel regular rows.
            pixel_latitudes = 90 - (np.arange(21_600) + 0.5) / 120
            row_indices = np.floor((90 - pixel_latitudes + 0.125) / 0.25).astype(int)
            totals = np.zeros(721, dtype=np.int64)
            np.add.at(totals, row_indices, counts)
            expected = np.zeros((721, 1440), dtype=np.int64)
            expected[:, 0] = 2 * totals
            expected[:, 720] = totals
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(int(actual.sum()), 3 * int(counts.sum(dtype=np.int64)))
            for index, cell_id, latitude, longitude in [
                (0, "era5-global-n900000-w1800000", 90.0, -180.0),
                (360 * 1440 + 720, "era5-global-n000000-e0000000", 0.0, 0.0),
                (720 * 1440, "era5-global-s900000-w1800000", -90.0, -180.0),
            ]:
                self.assertEqual(table["source_cell_id"][index].as_py(), cell_id)
                self.assertEqual(table["latitude"][index].as_py(), latitude)
                self.assertEqual(table["longitude"][index].as_py(), longitude)
            metadata = json.loads((outputs / "metadata.json").read_text())
            self.assertEqual(metadata["schema_version"], 3)
            self.assertNotIn("population_weight_definition", metadata)
            self.assertEqual(metadata["population_total"], int(expected.sum()))
            self.assertEqual(metadata["era5_cell_count"], 721 * 1440)
            self.assertEqual(metadata["source_size_bytes"], raster.stat().st_size)
            self.assertEqual(metadata["source_mtime_ns"], raster.stat().st_mtime_ns)
            with self.assertRaises(FileExistsError):
                population.compute_era5_population_weights(
                    raster, output_root=root / "output"
                )


if __name__ == "__main__":
    unittest.main()
