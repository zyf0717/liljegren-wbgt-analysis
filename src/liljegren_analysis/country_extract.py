"""Extract valid country observations to a reusable Parquet cache."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa

from .country_cells import (
    Country,
    cached_natural_earth_boundaries,
    country_cells,
    load_countries,
)
from .results import CampaignResults


@dataclass(frozen=True)
class CountryExtraction:
    """Metadata describing one country observation cache."""

    schema_version: int
    generated_at_utc: str
    country_iso_a2: str
    country_iso_a3: str
    country_name: str
    start_date: str
    end_date: str
    selected_cell_count: int
    observation_count: int
    runs: tuple[str, ...]
    parquet_file: str
    parquet_size_bytes: int


def extract_country_observations(
    country_code: str,
    runs: Sequence[str | Path],
    start_date: date,
    end_date: date,
    output: str | Path,
    *,
    boundaries: str | Path | None = None,
    boundary_cache: str | Path | None = None,
    threads: int = 0,
    overwrite: bool = False,
) -> CountryExtraction:
    """Write valid observations for one ADM0 country to Parquet.

    A cell is included when its 0.25-degree footprint has positive-area overlap
    with the country boundary. The output contains only ``status = 'ok'`` rows
    with non-null values. ``threads=0`` leaves DuckDB's default unchanged.
    """

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if threads < 0:
        raise ValueError("threads must be non-negative")
    resolved_runs = tuple(Path(run).expanduser().resolve() for run in runs)
    if not resolved_runs:
        raise ValueError("runs is empty")
    missing_runs = [run for run in resolved_runs if not (run / "results").is_dir()]
    if missing_runs:
        raise FileNotFoundError(f"Missing campaign result roots: {missing_runs}")

    output_path = Path(output).expanduser().resolve()
    manifest_path = output_path.with_name("manifest.json")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Country cache already exists: {output_path}; "
            "pass overwrite=True to replace it"
        )

    boundary_path = _resolve_boundaries(boundaries, boundary_cache, output_path)
    country = load_countries(boundary_path, [country_code])[0]
    cells = country_cells(country)
    if not cells:
        raise ValueError(f"No ERA5 cells intersect {country.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_sibling(output_path)
    try:
        observation_count = _copy_observations(
            resolved_runs,
            start_date,
            end_date,
            tuple(cell.cell_id for cell in cells),
            temporary_path,
            threads,
        )
        if observation_count == 0:
            raise ValueError(
                f"No valid {country.name} observations found "
                f"from {start_date} to {end_date}"
            )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    extraction = _metadata(
        country,
        start_date,
        end_date,
        len(cells),
        observation_count,
        resolved_runs,
        output_path,
    )
    _write_manifest_atomic(manifest_path, extraction)
    return extraction


def _resolve_boundaries(
    boundaries: str | Path | None,
    boundary_cache: str | Path | None,
    output: Path,
) -> Path:
    if boundaries is not None:
        path = Path(boundaries).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Boundary file does not exist: {path}")
        return path
    cache = (
        Path(boundary_cache).expanduser().resolve()
        if boundary_cache is not None
        else output.parents[2] / "boundaries"
    )
    return cached_natural_earth_boundaries(cache)


def _copy_observations(
    runs: tuple[Path, ...],
    start_date: date,
    end_date: date,
    cell_ids: tuple[str, ...],
    output: Path,
    threads: int,
) -> int:
    with duckdb.connect() as connection:
        if threads > 0:
            connection.execute(f"SET threads = {threads}")
        connection.register("_country_cells", pa.table({"cell_id": cell_ids}))
        relations = [
            CampaignResults.from_campaign(run, connection=connection).scan(
                start_date=start_date,
                end_date=end_date,
            )
            for run in runs
        ]
        combined = relations[0]
        for relation in relations[1:]:
            combined = combined.union(relation)
        combined.create_view("_country_results", replace=True)

        destination = _sql_literal(output.as_posix())
        connection.execute(
            f"""
            COPY (
                SELECT
                    results.result_date,
                    results.run_id,
                    results.implementation,
                    results.cycle_ordinal,
                    results.row_id,
                    substring(results.row_id, 1, 28) AS cell_id,
                    results.row_ordinal,
                    results.value_c
                FROM _country_results AS results
                INNER JOIN _country_cells AS cells
                    ON substring(results.row_id, 1, 28) = cells.cell_id
                WHERE results.status = 'ok'
                  AND results.value_c IS NOT NULL
            ) TO {destination} (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """
        )
        return int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({destination})"
            ).fetchone()[0]
        )


def _metadata(
    country: Country,
    start_date: date,
    end_date: date,
    selected_cell_count: int,
    observation_count: int,
    runs: tuple[Path, ...],
    output: Path,
) -> CountryExtraction:
    return CountryExtraction(
        schema_version=1,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        country_iso_a2=country.iso_a2,
        country_iso_a3=country.iso_a3,
        country_name=country.name,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        selected_cell_count=selected_cell_count,
        observation_count=observation_count,
        runs=tuple(str(run) for run in runs),
        parquet_file=output.name,
        parquet_size_bytes=output.stat().st_size,
    )


def _temporary_sibling(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".parquet.tmp",
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def _write_manifest_atomic(path: Path, extraction: CountryExtraction) -> None:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(asdict(extraction), destination, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
