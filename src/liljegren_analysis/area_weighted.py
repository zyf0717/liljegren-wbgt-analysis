"""ERA5 cell-area weights used by pairwise WBGT statistics."""

from __future__ import annotations

import math

import pyarrow as pa


ERA5_GRID_SPACING_DEGREES = 0.25
_ERA5_LATITUDE_SCALE = 10_000
_ERA5_LATITUDE_STEP = round(
    ERA5_GRID_SPACING_DEGREES * _ERA5_LATITUDE_SCALE
)
_ERA5_MAX_LATITUDE = 90 * _ERA5_LATITUDE_SCALE


def _normalized_cell_area_weight(latitude_degrees: float) -> float:
    half_spacing = ERA5_GRID_SPACING_DEGREES / 2.0
    lower = math.radians(max(-90.0, latitude_degrees - half_spacing))
    upper = math.radians(min(90.0, latitude_degrees + half_spacing))
    normalization = 2.0 * math.sin(math.radians(half_spacing))
    return (math.sin(upper) - math.sin(lower)) / normalization


def _precomputed_latitude_weights() -> pa.Table:
    latitude_units = range(
        -_ERA5_MAX_LATITUDE,
        _ERA5_MAX_LATITUDE + 1,
        _ERA5_LATITUDE_STEP,
    )
    codes: list[str] = []
    weights: list[float] = []
    for units in latitude_units:
        hemisphere = "s" if units < 0 else "n"
        codes.append(f"{hemisphere}{abs(units):06d}")
        weights.append(
            _normalized_cell_area_weight(units / _ERA5_LATITUDE_SCALE)
        )
    return pa.table({"latitude_code": codes, "area_weight": weights})


_ERA5_LATITUDE_WEIGHTS = _precomputed_latitude_weights()
