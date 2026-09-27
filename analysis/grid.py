"""Canonical ERA5 cell identities derived from HeatStressBench row IDs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ERA5_SPACING_DEGREES = 0.25
COORDINATE_SCALE = 10_000
ERA5_SPACING_UNITS = 2_500


@dataclass(frozen=True)
class CellGrid:
    """Integer cell IDs and their stable HeatStressBench source identities."""

    table: pa.Table

    @property
    def cell_count(self) -> int:
        return self.table.num_rows

    @property
    def fingerprint(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for value in self.table["source_cell_id"].combine_chunks().to_pylist():
            digest.update(value.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()


def grid_from_source_cell_ids(source_cell_ids: list[str]) -> CellGrid:
    """Create a deterministic integer grid from source cell IDs."""

    ordered = sorted(set(source_cell_ids))
    if not ordered:
        raise ValueError("Cannot construct a grid from no ERA5 cells")
    coordinates = [parse_era5_cell_id(value) for value in ordered]
    return CellGrid(
        pa.table(
            {
                "cell_id": pa.array(range(len(ordered)), type=pa.int64()),
                "source_cell_id": pa.array(ordered, type=pa.string()),
                "latitude": pa.array(
                    [latitude for latitude, _ in coordinates], type=pa.float64()
                ),
                "longitude": pa.array(
                    [longitude for _, longitude in coordinates], type=pa.float64()
                ),
            }
        )
    )


def parse_era5_cell_id(source_cell_id: str) -> tuple[float, float]:
    """Decode the coordinates in a trusted 28-character HSB cell key."""
    latitude = int(source_cell_id[13:19]) / COORDINATE_SCALE
    longitude = int(source_cell_id[21:28]) / COORDINATE_SCALE
    return (
        latitude if source_cell_id[12] == "n" else -latitude,
        longitude if source_cell_id[20] == "e" else -longitude,
    )


def source_cell_id(latitude_units: int, longitude_units: int) -> str:
    latitude_sign = "n" if latitude_units >= 0 else "s"
    longitude_sign = "e" if longitude_units >= 0 else "w"
    return (
        f"era5-global-{latitude_sign}{abs(latitude_units):06d}-"
        f"{longitude_sign}{abs(longitude_units):07d}"
    )


def load_grid(path: Path) -> CellGrid:
    return CellGrid(pq.read_table(path))


def normalized_cell_area(latitude: float) -> float:
    """Spherical cell area relative to an equatorial cell, clipped at the poles."""
    half = ERA5_SPACING_DEGREES / 2
    lower = math.radians(max(-90.0, latitude - half))
    upper = math.radians(min(90.0, latitude + half))
    return (math.sin(upper) - math.sin(lower)) / (2 * math.sin(math.radians(half)))


LATITUDE_WEIGHTS = {
    f"{'s' if units < 0 else 'n'}{abs(units):06d}": normalized_cell_area(
        units / COORDINATE_SCALE
    )
    for units in range(-900_000, 900_001, ERA5_SPACING_UNITS)
}
LATITUDE_WEIGHT_TABLE = pa.table(
    {
        "latitude_code": list(LATITUDE_WEIGHTS),
        "area_weight": list(LATITUDE_WEIGHTS.values()),
    }
)


def area_weights(source_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([LATITUDE_WEIGHTS[value[12:19]] for value in source_ids])
