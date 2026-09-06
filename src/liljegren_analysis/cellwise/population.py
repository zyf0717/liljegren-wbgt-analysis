"""Aggregate LandScan population counts onto the global ERA5 grid."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rasterio
from rasterio.windows import Window

from ..io.artifacts import atomic_json as _atomic_json
from ..io.artifacts import atomic_parquet as _atomic_parquet

ERA5_SPACING_DEGREES = 0.25
LANDSCAN_RESOLUTION_DEGREES = 1.0 / 120.0
LANDSCAN_PIXELS_PER_ERA5_CELL = 30
LANDSCAN_NODATA = -2_147_483_647
POPULATION_SCHEMA_VERSION = 1
DEFAULT_LANDSCAN_PATH = Path(
    "data/population/landscan-global-2024/landscan-global-2024.tif"
)
DEFAULT_OUTPUT_ROOT = Path("data/cellwise_population_weight")


@dataclass(frozen=True)
class PopulationWeightOutputs:
    """Persisted global ERA5 population-weight artifacts."""

    population: Path
    metadata: Path


def compute_era5_population_weights(
    input_raster: str | Path = DEFAULT_LANDSCAN_PATH,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> PopulationWeightOutputs:
    """Write global ERA5 population totals and normalized LandScan weights.

    The 30-arc-second LandScan grid aligns exactly with 0.25-degree ERA5 cell
    boundaries. Raster data are read one ERA5 latitude band at a time.
    """

    raster_path = Path(input_raster).expanduser().resolve()
    if not raster_path.is_file():
        raise FileNotFoundError(f"LandScan raster does not exist: {raster_path}")
    root = Path(output_root).expanduser().resolve()
    population_path = root / "population.parquet"
    metadata_path = root / "metadata.json"
    if not force and (population_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Population output already exists at {root}; pass force=True to replace it"
        )

    with rasterio.open(raster_path) as source:
        _validate_landscan_raster(source)
        populations = _aggregate_population(source)

    total_population = int(populations.sum(dtype=np.int64))
    if total_population <= 0:
        raise ValueError("LandScan raster has no positive valid population")
    latitudes, longitudes, source_cell_ids = _era5_coordinates()
    weights = populations.astype(np.float64) / total_population
    table = pa.table(
        {
            "source_cell_id": pa.array(source_cell_ids, type=pa.string()),
            "latitude": pa.array(latitudes, type=pa.float64()),
            "longitude": pa.array(longitudes, type=pa.float64()),
            "population": pa.array(populations, type=pa.int64()),
            "population_weight": pa.array(weights, type=pa.float64()),
        }
    )
    _atomic_parquet(table, population_path)
    _atomic_json(
        {
            "schema_version": POPULATION_SCHEMA_VERSION,
            "source": "LandScan Global 2024",
            "source_raster": str(raster_path),
            "source_sha256": _sha256(raster_path),
            "source_nodata": LANDSCAN_NODATA,
            "source_resolution_degrees": LANDSCAN_RESOLUTION_DEGREES,
            "era5_spacing_degrees": ERA5_SPACING_DEGREES,
            "era5_cell_count": table.num_rows,
            "grid_definition": "global-era5-0.25-degree-v1",
            "population_total": total_population,
            "population_weight_definition": "population / global_population_total",
            "polar_row_source_pixel_height": LANDSCAN_PIXELS_PER_ERA5_CELL // 2,
            "regular_cell_source_pixels": [
                LANDSCAN_PIXELS_PER_ERA5_CELL,
                LANDSCAN_PIXELS_PER_ERA5_CELL,
            ],
        },
        metadata_path,
    )
    return PopulationWeightOutputs(population_path, metadata_path)


def _validate_landscan_raster(source: rasterio.io.DatasetReader) -> None:
    if source.count != 1:
        raise ValueError("LandScan raster must have exactly one band")
    if source.width != 43_200 or source.height != 21_600:
        raise ValueError("LandScan raster must be a 43200 by 21600 global grid")
    if source.crs is None or source.crs.to_epsg() != 4326:
        raise ValueError("LandScan raster must use EPSG:4326")
    transform = source.transform
    expected = (
        LANDSCAN_RESOLUTION_DEGREES,
        0.0,
        -180.0,
        0.0,
        -LANDSCAN_RESOLUTION_DEGREES,
        90.0,
    )
    actual = (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )
    if not all(
        math.isclose(left, right, abs_tol=1e-10)
        for left, right in zip(actual, expected)
    ):
        raise ValueError(
            "LandScan raster transform is not the expected global 30-arc-second grid"
        )
    if source.nodata != LANDSCAN_NODATA:
        raise ValueError(f"LandScan raster nodata must be {LANDSCAN_NODATA}")


def _aggregate_population(source: rasterio.io.DatasetReader) -> np.ndarray:
    latitude_units = range(900_000, -900_001, -2_500)
    longitude_count = 1_440
    populations = np.empty((len(latitude_units), longitude_count), dtype=np.int64)
    for row_index, latitude_units_value in enumerate(latitude_units):
        row_start, row_end = _landscan_row_bounds(latitude_units_value)
        band = source.read(
            1,
            window=Window(0, row_start, source.width, row_end - row_start),
        )
        populations[row_index] = _sum_latitude_band(band)
    return populations.reshape(-1)


def _landscan_row_bounds(latitude_units: int) -> tuple[int, int]:
    latitude = latitude_units / 10_000.0
    upper = min(90.0, latitude + ERA5_SPACING_DEGREES / 2.0)
    lower = max(-90.0, latitude - ERA5_SPACING_DEGREES / 2.0)
    row_start = round((90.0 - upper) / LANDSCAN_RESOLUTION_DEGREES)
    row_end = round((90.0 - lower) / LANDSCAN_RESOLUTION_DEGREES)
    return row_start, row_end


def _sum_latitude_band(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] % LANDSCAN_PIXELS_PER_ERA5_CELL:
        raise ValueError("LandScan latitude band has invalid dimensions")
    valid = values != LANDSCAN_NODATA
    if np.any(values[valid] < 0):
        raise ValueError("LandScan raster contains negative population values")
    population = np.where(valid, values, 0).astype(np.int64, copy=False)
    return population.reshape(
        values.shape[0],
        values.shape[1] // LANDSCAN_PIXELS_PER_ERA5_CELL,
        LANDSCAN_PIXELS_PER_ERA5_CELL,
    ).sum(axis=(0, 2), dtype=np.int64)


def _era5_coordinates() -> tuple[np.ndarray, np.ndarray, list[str]]:
    latitude_units = np.repeat(np.arange(900_000, -900_001, -2_500), 1_440)
    longitude_units = np.tile(np.arange(-1_800_000, 1_800_000, 2_500), 721)
    latitudes = latitude_units.astype(np.float64) / 10_000.0
    longitudes = longitude_units.astype(np.float64) / 10_000.0
    source_cell_ids = [
        _source_cell_id(latitude, longitude)
        for latitude, longitude in zip(latitude_units, longitude_units, strict=True)
    ]
    return latitudes, longitudes, source_cell_ids


def _source_cell_id(latitude_units: int, longitude_units: int) -> str:
    latitude_hemisphere = "n" if latitude_units >= 0 else "s"
    longitude_hemisphere = "e" if longitude_units >= 0 else "w"
    return (
        f"era5-global-{latitude_hemisphere}{abs(latitude_units):06d}-"
        f"{longitude_hemisphere}{abs(longitude_units):07d}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
