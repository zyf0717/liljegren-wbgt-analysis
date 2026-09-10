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
ERA5_CELL_SIZE_DEGREES = 0.25


def regional_map_figure_size(
    longitude: np.ndarray, latitude: np.ndarray
) -> tuple[float, float]:
    """Fit a standalone regional map canvas to its projected aspect ratio."""

    x, y = equal_earth_coordinates(longitude, latitude)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        raise ValueError("Map extent contains no finite coordinates")
    y_span = float(np.ptp(y[valid]))
    if y_span <= 0.0:
        return (10.0, 4.8)
    aspect = float(np.ptp(x[valid])) / y_span
    height = 6.4 if aspect < 1.0 else 4.8
    return (float(np.clip(height * aspect + 1.5, 4.5, 10.0)), height)


def plot_equal_earth_cells(
    axis: Any,
    longitude: np.ndarray,
    latitude: np.ndarray,
    values: np.ndarray,
    *,
    regional: bool,
    point_size: float,
    cmap: str,
    vmin: float,
    vmax: float,
) -> Any:
    """Plot global centers or contiguous projected ERA5 cells for a region."""

    longitude_array, latitude_array, value_array = np.broadcast_arrays(
        np.asarray(longitude, dtype=np.float64),
        np.asarray(latitude, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )
    if longitude_array.ndim != 1:
        raise ValueError("Map cell arrays must be one-dimensional")
    if not regional:
        x, y = equal_earth_coordinates(longitude_array, latitude_array)
        return axis.scatter(
            x,
            y,
            c=value_array,
            s=point_size,
            marker="s",
            linewidths=0,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

    from matplotlib.collections import PolyCollection

    half_cell = ERA5_CELL_SIZE_DEGREES / 2.0
    corner_longitude = longitude_array[:, None] + np.array(
        [-half_cell, half_cell, half_cell, -half_cell]
    )
    corner_latitude = latitude_array[:, None] + np.array(
        [-half_cell, -half_cell, half_cell, half_cell]
    )
    x, y = equal_earth_coordinates(corner_longitude, corner_latitude)
    polygons = np.stack((x, y), axis=-1)
    collection = PolyCollection(
        polygons,
        array=np.ma.masked_invalid(value_array),
        cmap=cmap,
        edgecolors="none",
        linewidths=0,
        antialiaseds=False,
        rasterized=True,
    )
    collection.set_clim(vmin, vmax)
    axis.add_collection(collection)
    return collection


def configure_equal_earth_axis(
    axis: Any,
    longitude: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
) -> None:
    """Apply an Equal Earth outline, graticule, limits, and aspect to an axis."""

    longitude_line = np.linspace(-180.0, 180.0, 361)
    latitude_line = np.linspace(-90.0, 90.0, 181)
    for graticule_latitude in (-60.0, -30.0, 0.0, 30.0, 60.0):
        x, y = equal_earth_coordinates(
            longitude_line, np.full_like(longitude_line, graticule_latitude)
        )
        axis.plot(x, y, color="0.7", linewidth=0.35, zorder=0)
    for graticule_longitude in (-120.0, -60.0, 0.0, 60.0, 120.0):
        x, y = equal_earth_coordinates(
            np.full_like(latitude_line, graticule_longitude), latitude_line
        )
        axis.plot(x, y, color="0.7", linewidth=0.35, zorder=0)
    for boundary_longitude in (-180.0, 180.0):
        x, y = equal_earth_coordinates(
            np.full_like(latitude_line, boundary_longitude), latitude_line
        )
        axis.plot(x, y, color="0.25", linewidth=0.6, zorder=0)
    for boundary_latitude in (-90.0, 90.0):
        x, y = equal_earth_coordinates(
            longitude_line, np.full_like(longitude_line, boundary_latitude)
        )
        axis.plot(x, y, color="0.25", linewidth=0.6, zorder=0)
    axis.set(
        xlim=(-1.02 * EQUAL_EARTH_X_LIMIT, 1.02 * EQUAL_EARTH_X_LIMIT),
        ylim=(-1.02 * EQUAL_EARTH_Y_LIMIT, 1.02 * EQUAL_EARTH_Y_LIMIT),
        aspect="equal",
    )
    if longitude is not None or latitude is not None:
        if longitude is None or latitude is None:
            raise ValueError("longitude and latitude must be provided together")
        x, y = equal_earth_coordinates(longitude, latitude)
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            raise ValueError("Map extent contains no finite coordinates")
        x_span, y_span = np.ptp(x[valid]), np.ptp(y[valid])
        x_padding = max(0.05 * x_span, 0.01)
        y_padding = max(0.05 * y_span, 0.01)
        axis.set(
            xlim=(float(x[valid].min() - x_padding), float(x[valid].max() + x_padding)),
            ylim=(float(y[valid].min() - y_padding), float(y[valid].max() + y_padding)),
        )
    axis.set_axis_off()
