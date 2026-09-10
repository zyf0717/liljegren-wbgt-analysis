from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from liljegren_analysis.cellwise.projection import (
    EQUAL_EARTH_X_LIMIT,
    EQUAL_EARTH_Y_LIMIT,
    configure_equal_earth_axis,
    equal_earth_coordinates,
    mercator_coordinates,
    plot_equal_earth_cells,
    regional_map_figure_size,
)


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


if __name__ == "__main__":
    unittest.main()
