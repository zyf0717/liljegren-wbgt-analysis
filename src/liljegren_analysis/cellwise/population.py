"""Aggregate LandScan population counts onto the global ERA5 grid."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import rioxarray
import xarray as xr
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from ..io.artifacts import atomic_json as _atomic_json
from ..io.artifacts import atomic_parquet as _atomic_parquet

ERA5_SPACING_DEGREES = 0.25
LANDSCAN_RESOLUTION_DEGREES = 1.0 / 120.0
LANDSCAN_NODATA = -2_147_483_647
ERA5_LATITUDES = np.arange(90.0, -90.0 - ERA5_SPACING_DEGREES, -ERA5_SPACING_DEGREES)
ERA5_LONGITUDES = np.arange(-180.0, 180.0, ERA5_SPACING_DEGREES)
DEFAULT_LANDSCAN_PATH = Path(
    "data/population/landscan-global-2024/landscan-global-2024.tif"
)
DEFAULT_OUTPUT_ROOT = Path("data/cellwise_population_weight")
DEFAULT_POPULATION_WEIGHTS_PATH = DEFAULT_OUTPUT_ROOT / "population.parquet"


@dataclass(frozen=True)
class PopulationWeightOutputs:
    """Persisted global ERA5 population-weight artifacts."""

    population: Path
    metadata: Path


def load_population_weights(
    source_ids: list[str],
    path: str | Path = DEFAULT_POPULATION_WEIGHTS_PATH,
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
        raise ValueError("Population weights contain invalid source_cell_id or population values")
    population = np.asarray(counts.to_numpy(zero_copy_only=False), dtype=np.int64)
    if (population < 0).any():
        raise ValueError("Population weights contain negative population values")
    lookup = dict(zip(population_ids, population, strict=True))
    weights = np.asarray([lookup.get(source_id, -1) for source_id in source_ids], dtype=np.int64)
    if (weights < 0).any():
        raise ValueError("Population weights are missing a cell from the analysis grid")
    return weights, int(population.sum(dtype=np.int64))


def compute_era5_population_weights(
    input_raster: str | Path = DEFAULT_LANDSCAN_PATH,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> PopulationWeightOutputs:
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
    if not force and (population_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Population output already exists at {root}; pass force=True to replace it"
        )

    with rasterio.open(raster_path) as raster:
        _validate_landscan_raster(raster)
        with rioxarray.open_rasterio(raster, cache=False) as source:
            populations = _aggregate_population(source)

    total_population = int(populations.sum(dtype=np.int64))
    if total_population <= 0:
        raise ValueError("LandScan raster has no positive valid population")
    latitudes = np.repeat(ERA5_LATITUDES, len(ERA5_LONGITUDES))
    longitudes = np.tile(ERA5_LONGITUDES, len(ERA5_LATITUDES))
    source_cell_ids = [
        _source_cell_id(round(latitude * 10_000), round(longitude * 10_000))
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
    with raster_path.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    _atomic_parquet(table, population_path)
    _atomic_json(
        {
            "schema_version": 3,
            "source": "LandScan Global 2024",
            "source_raster": str(raster_path),
            "source_sha256": source_sha256,
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
    return PopulationWeightOutputs(population_path, metadata_path)


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


def _aggregate_population(source: xr.DataArray) -> np.ndarray:
    populations = np.empty((len(ERA5_LATITUDES), len(ERA5_LONGITUDES)), dtype=np.int64)
    half_spacing = ERA5_SPACING_DEGREES / 2
    transform = source.rio.transform()
    for row_index, latitude in enumerate(ERA5_LATITUDES):
        window = from_bounds(
            -180.0,
            max(-90.0, latitude - half_spacing),
            180.0,
            min(90.0, latitude + half_spacing),
            transform,
        ).round_offsets().round_lengths()
        band = source.rio.isel_window(window).isel(band=0).load()
        populations[row_index] = _sum_latitude_band(band)
    return populations.ravel()


def _sum_latitude_band(band: xr.DataArray) -> np.ndarray:
    """Sum a loaded latitude band into centered, dateline-wrapped ERA5 cells."""
    population = band.where(band != band.rio.nodata, 0)
    if bool((population < 0).any()):
        raise ValueError("LandScan raster contains negative population values")
    pixels_per_cell = round(ERA5_SPACING_DEGREES / band.rio.resolution()[0])
    # The western half of the -180-degree cell is the raster's final columns.
    return (
        population.sum("y", dtype=np.int64)
        .roll(x=pixels_per_cell // 2, roll_coords=False)
        .coarsen(x=pixels_per_cell, boundary="exact")
        .sum()
        .values
    )


def _source_cell_id(latitude_units: int, longitude_units: int) -> str:
    latitude_hemisphere = "n" if latitude_units >= 0 else "s"
    longitude_hemisphere = "e" if longitude_units >= 0 else "w"
    return (
        f"era5-global-{latitude_hemisphere}{abs(latitude_units):06d}-"
        f"{longitude_hemisphere}{abs(longitude_units):07d}"
    )
