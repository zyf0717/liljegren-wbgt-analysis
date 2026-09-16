"""Dependency-free projections for global and regional map figures."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry.base import BaseGeometry

from ..country.cells import (
    NATURAL_EARTH_VERSION,
    GeographicBounds,
    country_geometry,
    load_countries,
)


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


def mercator_coordinates(
    longitude: np.ndarray, latitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude degrees to unit-sphere Mercator coordinates."""

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
        raise ValueError("Mercator coordinates exceed longitude/latitude bounds")
    longitude_radians = np.deg2rad(longitude_array)
    latitude_radians = np.deg2rad(
        np.clip(latitude_array, -89.999999, 89.999999)
    )
    return longitude_radians, np.arcsinh(np.tan(latitude_radians))


EQUAL_EARTH_X_LIMIT = float(
    equal_earth_coordinates(np.array([180.0]), np.array([0.0]))[0][0]
)
EQUAL_EARTH_Y_LIMIT = float(
    equal_earth_coordinates(np.array([0.0]), np.array([90.0]))[1][0]
)
ERA5_CELL_SIZE_DEGREES = 0.25
STACKED_MAP_DPI = 220
STACKED_MAP_WIDTH_PX = 1635
STACKED_MAP_WIDTH_IN = STACKED_MAP_WIDTH_PX / STACKED_MAP_DPI
PANEL_TITLE_FONT_SIZE_PT = 12.0
_MAJOR_CITIES = {
    "JPN": (
        ("Tokyo", 139.69171, 35.68950, (-4.0, 4.0), "right"),
        ("Osaka", 135.50107, 34.69379, (4.0, 4.0), "left"),
        ("Fukuoka", 130.41667, 33.60000, (4.0, 4.0), "left"),
    ),
    "THA": (
        ("Bangkok", 100.5020, 13.7573, (4.0, 4.0), "left"),
        ("Chiang Mai", 98.98468, 18.79038, (4.0, 4.0), "left"),
    ),
}


def add_map_colorbar(
    figure: Any,
    image: Any,
    *,
    ax: Any,
    label: str,
    capped: bool = True,
    shrink: float = 1.0,
) -> Any:
    """Use a narrow map colorbar with a triangle for upper saturation."""
    bounds = [axis.get_position() for axis in np.asarray(ax, dtype=object).ravel()]
    height = (
        (max(bound.y1 for bound in bounds) - min(bound.y0 for bound in bounds))
        * figure.get_figheight() * shrink
    )
    return figure.colorbar(
        image,
        ax=ax,
        label=label,
        extend="max" if capped else "neither",
        extendfrac=0.05,
        fraction=0.035,
        aspect=height / 0.16,
        pad=0.025,
        shrink=shrink,
    )


def country_boundary_from_metadata(
    metadata: Mapping[str, object],
) -> BaseGeometry | None:
    """Load the pinned country boundary colocated with a country cache."""

    selection = metadata.get("country_selection")
    cache = metadata.get("country_cache")
    if not isinstance(selection, Mapping) or not isinstance(cache, str):
        return None
    iso_a3 = selection.get("country_iso_a3")
    if not isinstance(iso_a3, str):
        return None
    cache_path = Path(cache).expanduser().resolve()
    if len(cache_path.parents) < 3:
        return None
    boundaries = cache_path.parents[2] / "boundaries" / (
        f"natural-earth-{NATURAL_EARTH_VERSION}-admin-0-countries-10m.geojson"
    )
    if not boundaries.is_file():
        return None
    country = load_countries(boundaries, (iso_a3,))[0]
    raw_bounds = selection.get("selection_bounds")
    bounds: GeographicBounds | None = None
    if raw_bounds is not None:
        if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 4:
            raise ValueError("Country selection_bounds must contain four values")
        bounds = tuple(float(value) for value in raw_bounds)  # type: ignore[assignment]
    return country_geometry(country, bounds=bounds)


def regional_map_figure_size(
    longitude: np.ndarray, latitude: np.ndarray
) -> tuple[float, float]:
    """Fit a standalone regional map canvas to its projected aspect ratio."""

    x, y = mercator_coordinates(longitude, latitude)
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
    """Plot Equal Earth global centers or contiguous Mercator regional cells."""

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
    x, y = mercator_coordinates(corner_longitude, corner_latitude)
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


def plot_country_boundary(axis: Any, geometry: BaseGeometry | None) -> None:
    """Overlay a legible Mercator country boundary without changing map limits."""

    if geometry is None or geometry.is_empty:
        return
    for longitude, latitude in _boundary_rings(geometry):
        x, y = mercator_coordinates(longitude, latitude)
        axis.plot(
            x,
            y,
            color="white",
            linewidth=1.5,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        axis.plot(
            x,
            y,
            color="0.1",
            linewidth=0.65,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )


def plot_major_cities(axis: Any, country_iso_a3: str | None) -> None:
    """Overlay selected city centres on a regional Mercator map."""

    cities = _MAJOR_CITIES.get(country_iso_a3 or "")
    if cities is None:
        return

    from matplotlib import patheffects

    longitude = np.asarray([city[1] for city in cities], dtype=np.float64)
    latitude = np.asarray([city[2] for city in cities], dtype=np.float64)
    x, y = mercator_coordinates(longitude, latitude)
    axis.scatter(
        x,
        y,
        s=56.0,
        marker="X",
        facecolors="0.1",
        edgecolors="white",
        linewidths=0.6,
        zorder=5,
    )
    for (name, _, _, offset, horizontal_alignment), city_x, city_y in zip(
        cities, x, y, strict=True
    ):
        axis.annotate(
            name,
            (city_x, city_y),
            xytext=offset,
            textcoords="offset points",
            color="0.1",
            fontsize=12.0,
            horizontalalignment=horizontal_alignment,
            verticalalignment="bottom",
            path_effects=[
                patheffects.Stroke(linewidth=2.0, foreground="white"),
                patheffects.Normal(),
            ],
            annotation_clip=True,
            zorder=6,
        )


def _boundary_rings(
    geometry: BaseGeometry,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if geometry.geom_type == "Polygon":
        rings = (geometry.exterior, *geometry.interiors)  # type: ignore[attr-defined]
        for ring in rings:
            coordinates = np.asarray(ring.coords, dtype=np.float64)
            yield coordinates[:, 0], coordinates[:, 1]
        return
    for part in getattr(geometry, "geoms", ()):
        yield from _boundary_rings(part)


def configure_equal_earth_axis(
    axis: Any,
    longitude: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
) -> None:
    """Apply a global Equal Earth or regional Mercator map configuration."""

    if longitude is not None or latitude is not None:
        if longitude is None or latitude is None:
            raise ValueError("longitude and latitude must be provided together")
        _configure_mercator_axis(axis, longitude, latitude)
        return

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
    axis.set_axis_off()


def _configure_mercator_axis(
    axis: Any, longitude: np.ndarray, latitude: np.ndarray
) -> None:
    x, y = mercator_coordinates(longitude, latitude)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        raise ValueError("Map extent contains no finite coordinates")
    x_span, y_span = np.ptp(x[valid]), np.ptp(y[valid])
    x_padding = max(0.05 * x_span, np.deg2rad(0.25))
    y_padding = max(0.05 * y_span, np.deg2rad(0.25))
    axis.set(
        xlim=(float(x[valid].min() - x_padding), float(x[valid].max() + x_padding)),
        ylim=(float(y[valid].min() - y_padding), float(y[valid].max() + y_padding)),
        aspect="equal",
    )
    axis.set_axis_off()
