"""Aggregate LandScan population counts onto the global ERA5 grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from ..grid import ERA5_SPACING_DEGREES, source_cell_id
from ..io.artifacts import atomic_json, atomic_parquet

LANDSCAN_RESOLUTION_DEGREES = 1.0 / 120.0
LANDSCAN_NODATA = -2_147_483_647
ERA5_LATITUDES = np.arange(90.0, -90.0 - ERA5_SPACING_DEGREES, -ERA5_SPACING_DEGREES)
ERA5_LONGITUDES = np.arange(-180.0, 180.0, ERA5_SPACING_DEGREES)


def load_population_weights(
    source_ids: list[str],
    path: str | Path,
) -> tuple[np.ndarray, int]:
    """Align persisted integer counts to source IDs and return the dataset total.

    Counts are already aggregated onto ERA5 cells. Zero-population grids are
    valid; callers choose their denominator and whether it must be positive.
    """
    population_path = Path(path).expanduser().resolve()
    if not population_path.is_file():
        raise FileNotFoundError(
            f"Population weights do not exist: {population_path}; run "
            "scripts/compute_era5_population_weights.py first"
        )
    table = pq.read_table(population_path, columns=["source_cell_id", "population"])
    population_ids = table["source_cell_id"].combine_chunks().to_pylist()
    counts = table["population"].combine_chunks()
    if (
        table["source_cell_id"].null_count
        or counts.null_count
        or not pa.types.is_integer(counts.type)
        or len(population_ids) != len(set(population_ids))
    ):
        raise ValueError(
            "Population weights contain invalid source_cell_id or population values"
        )
    population = np.asarray(counts.to_numpy(zero_copy_only=False), dtype=np.int64)
    if (population < 0).any():
        raise ValueError("Population weights contain negative population values")
    lookup = dict(zip(population_ids, population, strict=True))
    weights = np.asarray(
        [lookup.get(source_id, -1) for source_id in source_ids], dtype=np.int64
    )
    if (weights < 0).any():
        raise ValueError("Population weights are missing a cell from the analysis grid")
    return weights, int(population.sum(dtype=np.int64))


def compute_era5_population_weights(
    input_raster: str | Path,
    *,
    output_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Write global ERA5 population totals from aligned LandScan pixels.

    The 30-arc-second LandScan grid aligns exactly with 0.25-degree ERA5 cell
    boundaries at +/-0.125 degrees from each grid point. Longitude wraps at
    the dateline; latitude clips at the poles. Raster data are read one ERA5
    latitude band at a time.
    """

    raster_path = Path(input_raster).expanduser().resolve()
    if not raster_path.is_file():
        raise FileNotFoundError(f"LandScan raster does not exist: {raster_path}")
    root = Path(output_root).expanduser().resolve()
    population_path = root / "population.parquet"
    metadata_path = root / "metadata.json"
    if not overwrite and (population_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Population output already exists at {root}; pass overwrite=True to replace it"
        )

    with rasterio.open(raster_path) as raster:
        _validate_landscan_raster(raster)
        populations = _aggregate_population(raster)

    total_population = int(populations.sum(dtype=np.int64))
    if total_population <= 0:
        raise ValueError("LandScan raster has no positive valid population")
    latitudes = np.repeat(ERA5_LATITUDES, len(ERA5_LONGITUDES))
    longitudes = np.tile(ERA5_LONGITUDES, len(ERA5_LATITUDES))
    source_cell_ids = [
        source_cell_id(round(latitude * 10_000), round(longitude * 10_000))
        for latitude, longitude in zip(latitudes, longitudes, strict=True)
    ]
    table = pa.table(
        {
            "source_cell_id": pa.array(source_cell_ids, type=pa.string()),
            "latitude": pa.array(latitudes, type=pa.float64()),
            "longitude": pa.array(longitudes, type=pa.float64()),
            "population": pa.array(populations, type=pa.int64()),
        }
    )
    atomic_parquet(table, population_path)
    atomic_json(
        {
            "schema_version": 3,
            "source": "LandScan Global 2024",
            "source_raster": str(raster_path),
            "source_size_bytes": raster_path.stat().st_size,
            "source_mtime_ns": raster_path.stat().st_mtime_ns,
            "source_nodata": LANDSCAN_NODATA,
            "source_resolution_degrees": LANDSCAN_RESOLUTION_DEGREES,
            "era5_spacing_degrees": ERA5_SPACING_DEGREES,
            "era5_cell_count": table.num_rows,
            "grid_definition": "global-era5-0.25-degree-v1",
            "population_cell_bounds": "grid-point +/-0.125 degrees; longitude wraps, latitude clips",
            "population_total": total_population,
        },
        metadata_path,
    )
    return root


def _validate_landscan_raster(source: rasterio.io.DatasetReader) -> None:
    if source.count != 1:
        raise ValueError("LandScan raster must have exactly one band")
    if source.width != 43_200 or source.height != 21_600:
        raise ValueError("LandScan raster must be a 43200 by 21600 global grid")
    if source.crs is None or source.crs.to_epsg() != 4326:
        raise ValueError("LandScan raster must use EPSG:4326")
    expected = from_origin(
        -180.0,
        90.0,
        LANDSCAN_RESOLUTION_DEGREES,
        LANDSCAN_RESOLUTION_DEGREES,
    )
    if not source.transform.almost_equals(expected, precision=1e-10):
        raise ValueError(
            "LandScan raster transform is not the expected global 30-arc-second grid"
        )
    if source.nodata != LANDSCAN_NODATA:
        raise ValueError(f"LandScan raster nodata must be {LANDSCAN_NODATA}")
    if source.dtypes[0] != "int32":
        raise ValueError("LandScan raster must contain int32 population counts")


def _aggregate_population(source: rasterio.io.DatasetReader) -> np.ndarray:
    populations = np.empty((len(ERA5_LATITUDES), len(ERA5_LONGITUDES)), dtype=np.int64)
    half_spacing = ERA5_SPACING_DEGREES / 2
    transform = source.transform
    for row_index, latitude in enumerate(ERA5_LATITUDES):
        window = (
            from_bounds(
                -180.0,
                max(-90.0, latitude - half_spacing),
                180.0,
                min(90.0, latitude + half_spacing),
                transform,
            )
            .round_offsets()
            .round_lengths()
        )
        band = source.read(1, window=window)
        populations[row_index] = _sum_latitude_band(band)
    return populations.ravel()


def _sum_latitude_band(band: np.ndarray) -> np.ndarray:
    """Sum a latitude band into centered, dateline-wrapped ERA5 cells."""
    population = np.where(band == LANDSCAN_NODATA, 0, band)
    if (population < 0).any():
        raise ValueError("LandScan raster contains negative population values")
    pixels_per_cell = round(ERA5_SPACING_DEGREES / LANDSCAN_RESOLUTION_DEGREES)
    # The western half of the -180-degree cell is the raster's final columns.
    columns = population.sum(axis=0, dtype=np.int64)
    centered = np.roll(columns, pixels_per_cell // 2)
    return centered.reshape(-1, pixels_per_cell).sum(axis=1, dtype=np.int64)
