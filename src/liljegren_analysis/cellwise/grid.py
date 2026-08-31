"""Canonical ERA5 cell identities derived from HeatStressBench row IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_ERA5_CELL_ID = re.compile(r"^era5-global-([ns])(\d{6})-([ew])(\d{7})$")
_SPACING_UNITS = 2_500
_COORDINATE_SCALE = 10_000


@dataclass(frozen=True)
class CellGrid:
    """Integer cell IDs and their stable HeatStressBench source identities."""

    table: pa.Table

    @property
    def cell_count(self) -> int:
        return self.table.num_rows

    @property
    def lookup_table(self) -> pa.Table:
        return self.table.select(["source_cell_id", "cell_id"])

    @property
    def fingerprint(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for value in self.table["source_cell_id"].combine_chunks().to_pylist():
            digest.update(value.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()


def grid_from_source_cell_ids(source_cell_ids: list[str]) -> CellGrid:
    """Create a deterministic integer grid from validated source cell IDs."""

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
    """Validate and decode a 28-character HeatStressBench ERA5 cell key."""

    match = _ERA5_CELL_ID.fullmatch(source_cell_id)
    if match is None:
        raise ValueError(f"Invalid ERA5 source cell ID: {source_cell_id!r}")
    latitude_units = int(match.group(2))
    longitude_units = int(match.group(4))
    if (
        latitude_units > 900_000
        or longitude_units > 1_800_000
        or latitude_units % _SPACING_UNITS
        or longitude_units % _SPACING_UNITS
        or (match.group(3) == "e" and longitude_units >= 1_800_000)
    ):
        raise ValueError(f"Invalid ERA5 source cell ID: {source_cell_id!r}")
    latitude = latitude_units / _COORDINATE_SCALE
    longitude = longitude_units / _COORDINATE_SCALE
    if match.group(1) == "s":
        latitude = -latitude
    if match.group(3) == "w":
        longitude = -longitude
    return latitude, longitude


def source_cell_id_from_row_id(row_id: str) -> str:
    """Extract and validate the spatial component of one observation row ID."""

    if len(row_id) < 29 or row_id[28] != "-":
        raise ValueError(f"Invalid ERA5 row ID: {row_id!r}")
    source_cell_id = row_id[:28]
    parse_era5_cell_id(source_cell_id)
    return source_cell_id


def load_grid(path: Path) -> CellGrid:
    """Load and validate a persisted canonical grid."""

    table = pq.read_table(path)
    required = ("cell_id", "source_cell_id", "latitude", "longitude")
    if any(column not in table.column_names for column in required):
        raise ValueError(f"Grid {path} lacks required columns {required}")
    table = table.select(required)
    ids = table["cell_id"].combine_chunks().to_numpy(zero_copy_only=False)
    source_ids = table["source_cell_id"].combine_chunks().to_pylist()
    latitudes = table["latitude"].combine_chunks().to_numpy(zero_copy_only=False)
    longitudes = table["longitude"].combine_chunks().to_numpy(zero_copy_only=False)
    if not np.array_equal(ids, np.arange(len(ids), dtype=ids.dtype)):
        raise ValueError("Grid cell_id values must be contiguous deterministic integers")
    if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError("Grid source_cell_id values must be unique and sorted")
    for source_id, latitude, longitude in zip(
        source_ids, latitudes, longitudes, strict=True
    ):
        expected_latitude, expected_longitude = parse_era5_cell_id(source_id)
        if not np.isclose(latitude, expected_latitude) or not np.isclose(
            longitude, expected_longitude
        ):
            raise ValueError(f"Grid coordinates do not match {source_id}")
    return CellGrid(table)
