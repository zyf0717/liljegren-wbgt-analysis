"""Dependency-free Equal Earth projection helpers for global map figures."""

from __future__ import annotations

from typing import Any

import numpy as np


_A1 = 1.340264
_A2 = -0.081106
_A3 = 0.000893
_A4 = 0.003796
_SQRT_3 = np.sqrt(3.0)


def equal_earth_coordinates(
    longitude: np.ndarray, latitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude degrees to unit-sphere Equal Earth coordinates."""

    longitude_array, latitude_array = np.broadcast_arrays(
        np.asarray(longitude, dtype=np.float64),
        np.asarray(latitude, dtype=np.float64),
    )
    if np.isinf(longitude_array).any() or np.isinf(latitude_array).any() or (
        np.isfinite(longitude_array).any()
        and np.nanmax(np.abs(longitude_array)) > 180.0
    ) or (
        np.isfinite(latitude_array).any()
        and np.nanmax(np.abs(latitude_array)) > 90.0
    ):
        raise ValueError("Equal Earth coordinates exceed longitude/latitude bounds")
    longitude_radians = np.deg2rad(longitude_array)
    latitude_radians = np.deg2rad(latitude_array)
    theta = np.arcsin((_SQRT_3 / 2.0) * np.sin(latitude_radians))
    theta_squared = theta * theta
    theta_sixth = theta_squared**3
    denominator = 3.0 * (
        9.0 * _A4 * theta_sixth * theta_squared
        + 7.0 * _A3 * theta_sixth
        + 3.0 * _A2 * theta_squared
        + _A1
    )
    x = 2.0 * _SQRT_3 * longitude_radians * np.cos(theta) / denominator
    y = theta * (
        _A4 * theta_sixth * theta_squared
        + _A3 * theta_sixth
        + _A2 * theta_squared
        + _A1
    )
    return x, y


EQUAL_EARTH_X_LIMIT = float(
    equal_earth_coordinates(np.array([180.0]), np.array([0.0]))[0][0]
)
EQUAL_EARTH_Y_LIMIT = float(
    equal_earth_coordinates(np.array([0.0]), np.array([90.0]))[1][0]
)


def configure_equal_earth_axis(axis: Any) -> None:
    """Apply an Equal Earth outline, graticule, limits, and aspect to an axis."""

    longitude_line = np.linspace(-180.0, 180.0, 361)
    latitude_line = np.linspace(-90.0, 90.0, 181)
    for latitude in (-60.0, -30.0, 0.0, 30.0, 60.0):
        x, y = equal_earth_coordinates(
            longitude_line, np.full_like(longitude_line, latitude)
        )
        axis.plot(x, y, color="0.7", linewidth=0.35, zorder=0)
    for longitude in (-120.0, -60.0, 0.0, 60.0, 120.0):
        x, y = equal_earth_coordinates(
            np.full_like(latitude_line, longitude), latitude_line
        )
        axis.plot(x, y, color="0.7", linewidth=0.35, zorder=0)
    for longitude in (-180.0, 180.0):
        x, y = equal_earth_coordinates(
            np.full_like(latitude_line, longitude), latitude_line
        )
        axis.plot(x, y, color="0.25", linewidth=0.6, zorder=0)
    for latitude in (-90.0, 90.0):
        x, y = equal_earth_coordinates(
            longitude_line, np.full_like(longitude_line, latitude)
        )
        axis.plot(x, y, color="0.25", linewidth=0.6, zorder=0)
    axis.set(
        xlim=(-1.02 * EQUAL_EARTH_X_LIMIT, 1.02 * EQUAL_EARTH_X_LIMIT),
        ylim=(-1.02 * EQUAL_EARTH_Y_LIMIT, 1.02 * EQUAL_EARTH_Y_LIMIT),
        aspect="equal",
    )
    axis.set_axis_off()
