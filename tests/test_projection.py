from __future__ import annotations

import unittest

import numpy as np

from liljegren_analysis.cellwise.projection import (
    EQUAL_EARTH_X_LIMIT,
    EQUAL_EARTH_Y_LIMIT,
    equal_earth_coordinates,
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


if __name__ == "__main__":
    unittest.main()
