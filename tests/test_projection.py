from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from liljegren_analysis.cellwise.projection import (
    EQUAL_EARTH_X_LIMIT,
    EQUAL_EARTH_Y_LIMIT,
    configure_equal_earth_axis,
    country_boundary_from_metadata,
    equal_earth_coordinates,
    mercator_coordinates,
    plot_country_boundary,
    plot_equal_earth_cells,
    regional_map_figure_size,
)
from liljegren_analysis.country.cells import NATURAL_EARTH_VERSION
from shapely.geometry import box


class EqualEarthProjectionTests(unittest.TestCase):
    def test_origin_bounds_and_symmetry(self) -> None:
        longitude = np.array([0.0, 180.0, -180.0, 45.0, -45.0])
        latitude = np.array([0.0, 0.0, 0.0, 30.0, -30.0])
        x, y = equal_earth_coordinates(longitude, latitude)

        np.testing.assert_allclose(x[:3], [0.0, EQUAL_EARTH_X_LIMIT, -EQUAL_EARTH_X_LIMIT])
        np.testing.assert_allclose(y[:3], 0.0)
        self.assertAlmostEqual(x[3], -x[4])
        self.assertAlmostEqual(y[3], -y[4])
        self.assertGreater(EQUAL_EARTH_Y_LIMIT, 0.0)

    def test_projection_is_equal_area(self) -> None:
        step = 1e-5
        for latitude in (-60.0, -30.0, 0.0, 30.0, 60.0):
            longitude_step = np.rad2deg(step)
            latitude_step = np.rad2deg(step)
            x1, _ = equal_earth_coordinates(
                np.array([longitude_step]), np.array([latitude])
            )
            x2, _ = equal_earth_coordinates(
                np.array([-longitude_step]), np.array([latitude])
            )
            _, y1 = equal_earth_coordinates(
                np.array([0.0]), np.array([latitude + latitude_step])
            )
            _, y2 = equal_earth_coordinates(
                np.array([0.0]), np.array([latitude - latitude_step])
            )
            jacobian = ((x1[0] - x2[0]) / (2 * step)) * (
                (y1[0] - y2[0]) / (2 * step)
            )
            self.assertAlmostEqual(jacobian / np.cos(np.deg2rad(latitude)), 1.0, places=5)

    def test_invalid_geographic_bounds_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "longitude/latitude bounds"):
            equal_earth_coordinates(np.array([181.0]), np.array([0.0]))
        with self.assertRaisesRegex(ValueError, "longitude/latitude bounds"):
            equal_earth_coordinates(np.array([0.0]), np.array([np.inf]))

    def test_country_extent_is_fitted_to_input_grid(self) -> None:
        figure, axis = plt.subplots()
        configure_equal_earth_axis(
            axis,
            longitude=np.array([-125.0, -66.5]),
            latitude=np.array([24.0, 49.5]),
        )
        try:
            x, y = mercator_coordinates(
                np.array([-125.0, -66.5]), np.array([24.0, 49.5])
            )
            self.assertLess(axis.get_xlim()[0], x.min())
            self.assertGreater(axis.get_xlim()[1], x.max())
            self.assertLess(axis.get_ylim()[0], y.min())
            self.assertGreater(axis.get_ylim()[1], y.max())
            self.assertLess(np.ptp(axis.get_xlim()), 0.5 * 2 * EQUAL_EARTH_X_LIMIT)
            self.assertLess(np.ptp(axis.get_ylim()), 0.5 * 2 * EQUAL_EARTH_Y_LIMIT)
        finally:
            plt.close(figure)

    def test_regional_canvas_follows_projected_aspect_ratio(self) -> None:
        tall = regional_map_figure_size(
            longitude=np.array([97.25, 105.75]),
            latitude=np.array([5.75, 20.5]),
        )
        wide = regional_map_figure_size(
            longitude=np.array([-124.75, -67.0]),
            latitude=np.array([24.5, 49.25]),
        )

        self.assertLess(tall[0], tall[1])
        self.assertGreater(wide[0], wide[1])

    def test_regional_cells_are_contiguous_projected_polygons(self) -> None:
        figure, axis = plt.subplots()
        collection = plot_equal_earth_cells(
            axis,
            longitude=np.array([0.0, 0.25]),
            latitude=np.array([30.0, 30.0]),
            values=np.array([1.0, 2.0]),
            regional=True,
            point_size=0.35,
            cmap="magma",
            vmin=0.0,
            vmax=2.0,
        )
        try:
            self.assertIsInstance(collection, PolyCollection)
            paths = collection.get_paths()
            self.assertEqual(len(paths), 2)
            np.testing.assert_allclose(
                paths[0].vertices[1:3], paths[1].vertices[[0, 3]]
            )
            expected_x, expected_y = mercator_coordinates(
                np.array([-0.125, 0.125]), np.array([29.875, 30.125])
            )
            np.testing.assert_allclose(
                paths[0].vertices[[0, 2]],
                np.column_stack((expected_x, expected_y)),
            )
        finally:
            plt.close(figure)

    def test_country_boundary_is_overlaid_without_changing_extent(self) -> None:
        figure, axis = plt.subplots()
        configure_equal_earth_axis(
            axis,
            longitude=np.array([100.0, 101.0]),
            latitude=np.array([10.0, 11.0]),
        )
        limits = axis.get_xlim(), axis.get_ylim()

        plot_country_boundary(axis, box(100.0, 10.0, 101.0, 11.0))

        try:
            self.assertEqual(len(axis.lines), 2)
            self.assertEqual((axis.get_xlim(), axis.get_ylim()), limits)
            self.assertEqual(axis.lines[0].get_color(), "white")
            self.assertEqual(axis.lines[1].get_color(), "0.1")
        finally:
            plt.close(figure)

    def test_country_boundary_is_loaded_and_clipped_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            boundaries = root / "boundaries" / (
                f"natural-earth-{NATURAL_EARTH_VERSION}-admin-0-countries-10m.geojson"
            )
            boundaries.parent.mkdir(parents=True)
            boundaries.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {
                        "ADMIN": "Example",
                        "ISO_A2": "XX",
                        "ISO_A2_EH": "XX",
                        "ISO_A3": "XXX",
                        "ISO_A3_EH": "XXX",
                        "ADM0_A3": "XXX",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
                    },
                }],
            }))
            metadata = {
                "country_cache": str(root / "country" / "xxx" / "observations.parquet"),
                "country_selection": {
                    "country_iso_a3": "XXX",
                    "selection_bounds": [1.0, 0.0, 3.0, 2.0],
                },
            }

            geometry = country_boundary_from_metadata(metadata)

            assert geometry is not None
            self.assertEqual(geometry.bounds, (1.0, 0.0, 2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
