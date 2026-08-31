"""Extract matched country result rows to reusable Parquet caches."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ..results import _validate_result_schema
from .cells import (
    Country,
    cached_natural_earth_boundaries,
    country_cells,
    load_countries,
)


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
    result_row_count: int
    valid_observation_counts: dict[str, int]
    runs: tuple[str, ...]
    parquet_file: str
    parquet_size_bytes: int


@dataclass(frozen=True)
class CountryExtractionProgress:
    """Best-effort progress for a multi-country extraction."""

    schema_version: int
    job_id: str
    pid: int
    state: str
    phase: str
    started_at_utc: str
    updated_at_utc: str
    elapsed_seconds: float
    countries: tuple[str, ...]
    total_source_files: int
    selected_source_files: int
    duckdb_percentage: float | None = None
    error: str | None = None


ProgressCallback = Callable[[CountryExtractionProgress], None]


@dataclass(frozen=True)
class _CountrySelection:
    country: Country
    cell_ids: tuple[str, ...]
    output: Path

    @property
    def row_id_range(self) -> tuple[str, str]:
        return min(self.cell_ids) + "-", max(self.cell_ids) + "."


class _ProgressReporter:
    def __init__(
        self,
        selections: Sequence[_CountrySelection],
        callback: ProgressCallback | None,
        interval_seconds: float,
    ) -> None:
        self.job_id = uuid.uuid4().hex
        self.pid = os.getpid()
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.selections = tuple(selections)
        self.callback = callback
        self.interval_seconds = interval_seconds
        self.last_update_monotonic = float("-inf")
        self.last_phase: str | None = None

    def update(
        self,
        phase: str,
        *,
        state: str = "running",
        total_source_files: int = 0,
        selected_source_files: int = 0,
        duckdb_percentage: float | None = None,
        error: str | None = None,
        force: bool = False,
    ) -> None:
        now_monotonic = time.monotonic()
        if (
            not force
            and phase == self.last_phase
            and now_monotonic - self.last_update_monotonic < self.interval_seconds
        ):
            return
        if duckdb_percentage is not None and duckdb_percentage < 0:
            duckdb_percentage = None
        now = datetime.now(timezone.utc)
        progress = CountryExtractionProgress(
            schema_version=1,
            job_id=self.job_id,
            pid=self.pid,
            state=state,
            phase=phase,
            started_at_utc=self.started_at.isoformat(),
            updated_at_utc=now.isoformat(),
            elapsed_seconds=round(now_monotonic - self.started_monotonic, 3),
            countries=tuple(
                selection.country.iso_a3 for selection in self.selections
            ),
            total_source_files=total_source_files,
            selected_source_files=selected_source_files,
            duckdb_percentage=duckdb_percentage,
            error=error,
        )
        payload = asdict(progress)
        for selection in self.selections:
            try:
                _write_json_atomic(
                    selection.output.with_name("extraction-status.json"), payload
                )
            except OSError:
                pass
        if self.callback is not None:
            try:
                self.callback(progress)
            except Exception:
                pass
        self.last_phase = phase
        self.last_update_monotonic = now_monotonic


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
    progress: ProgressCallback | None = None,
    status_interval_seconds: float = 30.0,
) -> CountryExtraction:
    """Write matched result rows for one ADM0 country to Parquet.

    Cells are selected by positive-area country overlap. All matched result
    rows and component statuses are written. ``threads=0`` keeps DuckDB's
    default.
    """

    output_path = Path(output).expanduser().resolve()
    return _extract_countries_observations(
        (country_code,),
        runs,
        start_date,
        end_date,
        output_path.parent,
        boundaries=boundaries,
        boundary_cache=boundary_cache,
        threads=threads,
        overwrite=overwrite,
        progress=progress,
        status_interval_seconds=status_interval_seconds,
        explicit_output=output_path,
    )[0]


def extract_countries_observations(
    country_codes: Sequence[str],
    runs: Sequence[str | Path],
    start_date: date,
    end_date: date,
    output_directory: str | Path,
    *,
    boundaries: str | Path | None = None,
    boundary_cache: str | Path | None = None,
    threads: int = 0,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
    status_interval_seconds: float = 30.0,
) -> tuple[CountryExtraction, ...]:
    """Extract multiple countries with one pruned scan of the source results.

    Each country is written to
    ``<output_directory>/<iso3>/observations.parquet``. Country codes are
    deduplicated by resolved ISO alpha-3 code while preserving input order.
    """

    return _extract_countries_observations(
        country_codes,
        runs,
        start_date,
        end_date,
        Path(output_directory).expanduser().resolve(),
        boundaries=boundaries,
        boundary_cache=boundary_cache,
        threads=threads,
        overwrite=overwrite,
        progress=progress,
        status_interval_seconds=status_interval_seconds,
    )


def _extract_countries_observations(
    country_codes: Sequence[str],
    runs: Sequence[str | Path],
    start_date: date,
    end_date: date,
    output_directory: Path,
    *,
    boundaries: str | Path | None,
    boundary_cache: str | Path | None,
    threads: int,
    overwrite: bool,
    progress: ProgressCallback | None,
    status_interval_seconds: float,
    explicit_output: Path | None = None,
) -> tuple[CountryExtraction, ...]:
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if threads < 0:
        raise ValueError("threads must be non-negative")
    if status_interval_seconds <= 0:
        raise ValueError("status_interval_seconds must be positive")
    if isinstance(country_codes, (str, bytes)):
        raise TypeError("country_codes must be a sequence, not a string")
    requested_codes = list(country_codes)
    if not requested_codes:
        raise ValueError("country_codes is empty")

    resolved_runs = tuple(Path(run).expanduser().resolve() for run in runs)
    if not resolved_runs:
        raise ValueError("runs is empty")
    missing_runs = [run for run in resolved_runs if not (run / "results").is_dir()]
    if missing_runs:
        raise FileNotFoundError(f"Missing HSB run result roots: {missing_runs}")

    boundary_path = _resolve_boundaries(
        boundaries,
        boundary_cache,
        explicit_output or output_directory / "xxx" / "observations.parquet",
    )
    countries = _deduplicate_countries(load_countries(boundary_path, requested_codes))
    selections: list[_CountrySelection] = []
    for country in countries:
        cells = tuple(cell.cell_id for cell in country_cells(country))
        if not cells:
            raise ValueError(f"No ERA5 cells intersect {country.name}")
        output = (
            explicit_output
            if explicit_output is not None
            else output_directory
            / country.iso_a3.lower()
            / "observations.parquet"
        )
        selections.append(_CountrySelection(country, cells, output))

    _validate_output_targets(selections, overwrite)
    for selection in selections:
        selection.output.parent.mkdir(parents=True, exist_ok=True)

    reporter = _ProgressReporter(selections, progress, status_interval_seconds)
    total_files = 0
    selected_files = 0
    staging_root = output_directory if explicit_output is None else explicit_output.parent
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(dir=staging_root, prefix=f".country-extract-{reporter.job_id}-")
    )
    connection: duckdb.DuckDBPyConnection | None = None
    temporary_outputs: dict[str, Path] = {}
    temporary_manifests: dict[str, Path] = {}
    try:
        reporter.update("discovering", force=True)
        candidates = _result_files(resolved_runs, start_date, end_date)
        total_files = len(candidates)
        ranges = _merge_ranges(selection.row_id_range for selection in selections)
        selected = _select_result_files(candidates, ranges)
        selected_files = len(selected)
        if not selected:
            raise ValueError("No source result files overlap the requested countries")

        reporter.update(
            "scanning",
            total_source_files=total_files,
            selected_source_files=selected_files,
            force=True,
        )
        connection = duckdb.connect(str(staging_directory / "extraction.duckdb"))
        if threads > 0:
            connection.execute(f"SET threads = {threads}")
        connection.execute("SET enable_progress_bar = true")
        connection.execute("SET enable_progress_bar_print = false")
        _register_country_cells(connection, selections)
        _create_results_view(connection, selected)
        _execute_scan(
            connection,
            ranges,
            reporter,
            total_source_files=total_files,
            selected_source_files=selected_files,
        )

        counts = {
            iso_a3: int(count)
            for iso_a3, count in connection.execute(
                """
                SELECT country_iso_a3, count(*)
                FROM _matched_country_results
                GROUP BY country_iso_a3
                """
            ).fetchall()
        }
        valid_counts = {
            iso_a3: {
                "wbgt": int(wbgt),
                "tg": int(tg),
                "tnwb": int(tnwb),
            }
            for iso_a3, wbgt, tg, tnwb in connection.execute(
                """
                SELECT
                    country_iso_a3,
                    count(*) FILTER (
                        WHERE wbgt_status = 'ok' AND wbgt_c IS NOT NULL
                    ),
                    count(*) FILTER (
                        WHERE tg_status = 'ok' AND tg_c IS NOT NULL
                    ),
                    count(*) FILTER (
                        WHERE tnwb_status = 'ok' AND tnwb_c IS NOT NULL
                    )
                FROM _matched_country_results
                GROUP BY country_iso_a3
                """
            ).fetchall()
        }
        missing = [
            selection.country.name
            for selection in selections
            if counts.get(selection.country.iso_a3, 0) == 0
        ]
        if missing:
            raise ValueError(
                f"No result rows found for {', '.join(missing)} "
                f"from {start_date} to {end_date}"
            )

        reporter.update(
            "exporting",
            total_source_files=total_files,
            selected_source_files=selected_files,
            force=True,
        )
        extractions: list[CountryExtraction] = []
        for selection in selections:
            iso_a3 = selection.country.iso_a3
            temporary_output = _temporary_sibling(selection.output)
            temporary_outputs[iso_a3] = temporary_output
            _export_country(connection, iso_a3, temporary_output)
            extraction = _metadata(
                selection.country,
                start_date,
                end_date,
                len(selection.cell_ids),
                counts[iso_a3],
                valid_counts[iso_a3],
                resolved_runs,
                selection.output.name,
                temporary_output.stat().st_size,
            )
            temporary_manifest = _temporary_sibling(
                selection.output.with_name("manifest.json"), suffix=".json.tmp"
            )
            temporary_manifests[iso_a3] = temporary_manifest
            _write_json(temporary_manifest, asdict(extraction))
            extractions.append(extraction)

        reporter.update(
            "publishing",
            total_source_files=total_files,
            selected_source_files=selected_files,
            force=True,
        )
        for selection in selections:
            iso_a3 = selection.country.iso_a3
            os.replace(temporary_outputs[iso_a3], selection.output)
            os.replace(
                temporary_manifests[iso_a3], selection.output.with_name("manifest.json")
            )

        reporter.update(
            "complete",
            state="complete",
            total_source_files=total_files,
            selected_source_files=selected_files,
            duckdb_percentage=100.0,
            force=True,
        )
        return tuple(extractions)
    except KeyboardInterrupt:
        if connection is not None:
            connection.interrupt()
        reporter.update(
            "cancelled",
            state="cancelled",
            total_source_files=total_files,
            selected_source_files=selected_files,
            error="Interrupted",
            force=True,
        )
        raise
    except Exception as error:
        reporter.update(
            "failed",
            state="failed",
            total_source_files=total_files,
            selected_source_files=selected_files,
            error=str(error),
            force=True,
        )
        raise
    finally:
        if connection is not None:
            connection.close()
        for path in (*temporary_outputs.values(), *temporary_manifests.values()):
            path.unlink(missing_ok=True)
        shutil.rmtree(staging_directory, ignore_errors=True)


def _deduplicate_countries(countries: Sequence[Country]) -> tuple[Country, ...]:
    deduplicated: dict[str, Country] = {}
    for country in countries:
        deduplicated.setdefault(country.iso_a3, country)
    return tuple(deduplicated.values())


def _validate_output_targets(
    selections: Sequence[_CountrySelection], overwrite: bool
) -> None:
    outputs = [selection.output for selection in selections]
    if len(set(outputs)) != len(outputs):
        raise ValueError("multiple countries resolve to the same output path")
    if overwrite:
        return
    conflicts = [
        path
        for output in outputs
        for path in (output, output.with_name("manifest.json"))
        if path.exists()
    ]
    if conflicts:
        raise FileExistsError(
            f"Country cache already exists: {conflicts[0]}; "
            "pass overwrite=True to replace it"
        )


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


def _result_files(
    runs: tuple[Path, ...], start_date: date, end_date: date
) -> tuple[Path, ...]:
    paths: list[Path] = []
    day = start_date
    while day <= end_date:
        date_part = day.isoformat()
        for run in runs:
            date_root = run / "results" / date_part
            paths.extend(sorted(date_root.glob("*/*/results.parquet")))
        day += timedelta(days=1)
    if not paths:
        raise FileNotFoundError(
            f"No result files found from {start_date} to {end_date}"
        )
    return tuple(paths)


def _merge_ranges(
    ranges: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    merged: list[tuple[str, str]] = []
    for lower, upper in sorted(ranges):
        if merged and lower <= merged[-1][1]:
            previous_lower, previous_upper = merged[-1]
            merged[-1] = previous_lower, max(previous_upper, upper)
        else:
            merged.append((lower, upper))
    return tuple(merged)


def _parquet_may_overlap(
    path: Path, ranges: Sequence[tuple[str, str]]
) -> bool:
    metadata = pq.read_metadata(path)
    try:
        row_id_index = metadata.schema.names.index("row_id")
    except ValueError:
        return True
    if metadata.num_row_groups == 0:
        return False
    for index in range(metadata.num_row_groups):
        statistics = metadata.row_group(index).column(row_id_index).statistics
        if statistics is None or not statistics.has_min_max:
            return True
        minimum = _statistic_text(statistics.min)
        maximum = _statistic_text(statistics.max)
        if minimum is None or maximum is None:
            return True
        if any(maximum >= lower and minimum < upper for lower, upper in ranges):
            return True
    return False


def _select_result_files(
    files: Sequence[Path], ranges: Sequence[tuple[str, str]]
) -> tuple[Path, ...]:
    worker_count = min(32, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        overlaps = executor.map(
            lambda path: _parquet_may_overlap(path, ranges),
            files,
        )
        return tuple(path for path, overlap in zip(files, overlaps) if overlap)


def _statistic_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def _register_country_cells(
    connection: duckdb.DuckDBPyConnection,
    selections: Sequence[_CountrySelection],
) -> None:
    country_iso_a3: list[str] = []
    cell_ids: list[str] = []
    for selection in selections:
        country_iso_a3.extend([selection.country.iso_a3] * len(selection.cell_ids))
        cell_ids.extend(selection.cell_ids)
    connection.register(
        "_country_cells",
        pa.table({"country_iso_a3": country_iso_a3, "cell_id": cell_ids}),
    )


def _create_results_view(
    connection: duckdb.DuckDBPyConnection, files: Sequence[Path]
) -> None:
    relation = connection.read_parquet([path.as_posix() for path in files], filename=True)
    _validate_result_schema(relation)
    relation.project(
        """
        parse_path(filename)[-4]::DATE AS result_date,
        parse_path(filename)[-3] AS run_id,
        parse_path(filename)[-2] AS implementation,
        cycle_ordinal,
        row_id,
        row_ordinal,
        wbgt_status,
        wbgt_c,
        tg_status,
        tg_c,
        tnwb_status,
        tnwb_c
        """
    ).create_view("_country_results", replace=True)


def _execute_scan(
    connection: duckdb.DuckDBPyConnection,
    ranges: Sequence[tuple[str, str]],
    reporter: _ProgressReporter,
    *,
    total_source_files: int,
    selected_source_files: int,
) -> None:
    range_predicate = " OR ".join(
        f"(results.row_id >= {_sql_literal(lower)} "
        f"AND results.row_id < {_sql_literal(upper)})"
        for lower, upper in ranges
    )
    statement = f"""
        CREATE TABLE _matched_country_results AS
        WITH selected_results AS (
            SELECT
                result_date,
                run_id,
                implementation,
                cycle_ordinal,
                row_id,
                substring(row_id, 1, 28) AS cell_id,
                row_ordinal,
                wbgt_status,
                wbgt_c,
                tg_status,
                tg_c,
                tnwb_status,
                tnwb_c
            FROM _country_results AS results
            WHERE ({range_predicate})
        )
        SELECT
            cells.country_iso_a3,
            results.result_date,
            results.run_id,
            results.implementation,
            results.cycle_ordinal,
            results.row_id,
            results.cell_id,
            results.row_ordinal,
            results.wbgt_status,
            results.wbgt_c,
            results.tg_status,
            results.tg_c,
            results.tnwb_status,
            results.tnwb_c
        FROM selected_results AS results
        INNER JOIN _country_cells AS cells USING (cell_id)
    """
    failure: list[BaseException] = []

    def execute() -> None:
        try:
            connection.execute(statement)
        except BaseException as error:
            failure.append(error)

    worker = threading.Thread(target=execute, name="country-extraction-scan")
    worker.start()
    try:
        while worker.is_alive():
            worker.join(timeout=1.0)
            if worker.is_alive():
                try:
                    duckdb_percentage = connection.query_progress()
                except duckdb.Error:
                    duckdb_percentage = None
                reporter.update(
                    "scanning",
                    total_source_files=total_source_files,
                    selected_source_files=selected_source_files,
                    duckdb_percentage=duckdb_percentage,
                )
    except KeyboardInterrupt:
        connection.interrupt()
        worker.join()
        raise
    if failure:
        raise failure[0]


def _export_country(
    connection: duckdb.DuckDBPyConnection, iso_a3: str, output: Path
) -> None:
    destination = _sql_literal(output.as_posix())
    country = _sql_literal(iso_a3)
    connection.execute(
        f"""
        COPY (
            SELECT
                result_date,
                run_id,
                implementation,
                cycle_ordinal,
                row_id,
                cell_id,
                row_ordinal,
                wbgt_status,
                wbgt_c,
                tg_status,
                tg_c,
                tnwb_status,
                tnwb_c
            FROM _matched_country_results
            WHERE country_iso_a3 = {country}
        ) TO {destination} (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """
    )


def _metadata(
    country: Country,
    start_date: date,
    end_date: date,
    selected_cell_count: int,
    result_row_count: int,
    valid_observation_counts: dict[str, int],
    runs: tuple[Path, ...],
    parquet_file: str,
    parquet_size_bytes: int,
) -> CountryExtraction:
    return CountryExtraction(
        schema_version=2,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        country_iso_a2=country.iso_a2,
        country_iso_a3=country.iso_a3,
        country_name=country.name,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        selected_cell_count=selected_cell_count,
        result_row_count=result_row_count,
        valid_observation_counts=valid_observation_counts,
        runs=tuple(str(run) for run in runs),
        parquet_file=parquet_file,
        parquet_size_bytes=parquet_size_bytes,
    )


def _temporary_sibling(path: Path, *, suffix: str = ".parquet.tmp") -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=suffix,
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
