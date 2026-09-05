"""Bounded-memory Arrow extraction of country result observations."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..results import validate_result_schema
from .cells import Country, cached_natural_earth_boundaries, country_cells, load_countries


@dataclass(frozen=True)
class CountryExtraction:
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
    schema_version: int
    job_id: str
    pid: int
    state: str
    phase: str
    started_at_utc: str
    updated_at_utc: str
    elapsed_seconds: float
    countries: tuple[str, ...]
    files_completed: int
    files_total: int
    rows_scanned: int
    rows_selected: int
    error: str | None = None


ProgressCallback = Callable[[CountryExtractionProgress], None]
_OUTPUT_COLUMNS = (
    "result_date", "run_id", "implementation", "cycle_ordinal", "row_id", "cell_id",
    "row_ordinal", "wbgt_status", "wbgt_c", "tg_status", "tg_c", "tnwb_status", "tnwb_c",
)
_SOURCE_COLUMNS = tuple(column for column in _OUTPUT_COLUMNS if column not in {"result_date", "run_id", "implementation", "cell_id"})


@dataclass(frozen=True)
class _Selection:
    country: Country
    cells: tuple[str, ...]
    output: Path


class _Reporter:
    def __init__(self, selections: Sequence[_Selection], callback: ProgressCallback | None) -> None:
        self.job_id = uuid.uuid4().hex
        self.selections = tuple(selections)
        self.callback = callback
        self.started = datetime.now(timezone.utc)
        self.clock = time.monotonic()

    def update(self, phase: str, *, state: str = "running", files_completed: int = 0, files_total: int = 0, rows_scanned: int = 0, rows_selected: int = 0, error: str | None = None) -> None:
        progress = CountryExtractionProgress(
            2, self.job_id, os.getpid(), state, phase, self.started.isoformat(),
            datetime.now(timezone.utc).isoformat(), round(time.monotonic() - self.clock, 3),
            tuple(item.country.iso_a3 for item in self.selections), files_completed, files_total,
            rows_scanned, rows_selected, error,
        )
        payload = asdict(progress)
        for item in self.selections:
            _write_json_atomic(item.output.with_name("extraction-status.json"), payload)
        if self.callback is not None:
            self.callback(progress)


def extract_country_observations(country_code: str, runs: Sequence[str | Path], start_date: date, end_date: date, output: str | Path, *, boundaries: str | Path | None = None, boundary_cache: str | Path | None = None, overwrite: bool = False, progress: ProgressCallback | None = None) -> CountryExtraction:
    return _extract((country_code,), runs, start_date, end_date, Path(output).expanduser().resolve().parent, boundaries=boundaries, boundary_cache=boundary_cache, overwrite=overwrite, progress=progress, explicit_output=Path(output).expanduser().resolve())[0]


def extract_countries_observations(country_codes: Sequence[str], runs: Sequence[str | Path], start_date: date, end_date: date, output_directory: str | Path, *, boundaries: str | Path | None = None, boundary_cache: str | Path | None = None, overwrite: bool = False, progress: ProgressCallback | None = None) -> tuple[CountryExtraction, ...]:
    return _extract(country_codes, runs, start_date, end_date, Path(output_directory).expanduser().resolve(), boundaries=boundaries, boundary_cache=boundary_cache, overwrite=overwrite, progress=progress)


def _extract(country_codes: Sequence[str], runs: Sequence[str | Path], start_date: date, end_date: date, output_directory: Path, *, boundaries: str | Path | None, boundary_cache: str | Path | None, overwrite: bool, progress: ProgressCallback | None, explicit_output: Path | None = None) -> tuple[CountryExtraction, ...]:
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if isinstance(country_codes, (str, bytes)) or not country_codes:
        raise ValueError("country_codes is empty")
    resolved_runs = tuple(Path(run).expanduser().resolve() for run in runs)
    if not resolved_runs or any(not (run / "results").is_dir() for run in resolved_runs):
        raise FileNotFoundError("Missing HSB run result roots")
    boundary_path = _resolve_boundaries(boundaries, boundary_cache, explicit_output or output_directory / "xxx" / "observations.parquet")
    countries = _deduplicate_countries(load_countries(boundary_path, country_codes))
    selections = tuple(
        _Selection(country, tuple(cell.cell_id for cell in country_cells(country)), explicit_output if explicit_output is not None else output_directory / country.iso_a3.lower() / "observations.parquet")
        for country in countries
    )
    if any(not item.cells for item in selections):
        raise ValueError("No ERA5 cells intersect a requested country")
    _validate_outputs(selections, overwrite)
    reporter = _Reporter(selections, progress)
    files = _select_result_files(_result_files(resolved_runs, start_date, end_date), _merge_ranges((min(item.cells) + "-", max(item.cells) + ".") for item in selections))
    if not files:
        raise ValueError("No source result files overlap the requested countries")
    temporary = {item.country.iso_a3: _temporary_sibling(item.output) for item in selections}
    writers: dict[str, pq.ParquetWriter] = {}
    counts = {item.country.iso_a3: 0 for item in selections}
    valid = {item.country.iso_a3: {quantity: 0 for quantity in ("wbgt", "tg", "tnwb")} for item in selections}
    membership = {cell: [] for item in selections for cell in item.cells}
    for item in selections:
        for cell in item.cells:
            membership[cell].append(item.country.iso_a3)
    scanned = selected = 0
    try:
        reporter.update("discovering", files_total=len(files))
        for index, path in enumerate(files, 1):
            reader = pq.ParquetFile(path)
            validate_result_schema(reader.schema_arrow)
            result_date, run_id, implementation = path.parents[2].name, path.parents[1].name, path.parent.name
            for batch in reader.iter_batches(columns=list(_SOURCE_COLUMNS), batch_size=65_536):
                scanned += batch.num_rows
                row_ids = batch.column(batch.schema.get_field_index("row_id")).to_pylist()
                groups: dict[str, list[int]] = {}
                for row, row_id in enumerate(row_ids):
                    for iso_a3 in membership.get(row_id[:28], ()):
                        groups.setdefault(iso_a3, []).append(row)
                for iso_a3, rows in groups.items():
                    taken = pa.Table.from_batches([batch]).take(pa.array(rows, type=pa.int64()))
                    table = pa.table({
                        "result_date": pa.array([date.fromisoformat(result_date)] * len(rows), type=pa.date32()),
                        "run_id": pa.array([run_id] * len(rows)), "implementation": pa.array([implementation] * len(rows)),
                        "cycle_ordinal": taken["cycle_ordinal"], "row_id": taken["row_id"],
                        "cell_id": pa.array([row_id[:28] for row_id in taken["row_id"].to_pylist()]),
                        "row_ordinal": taken["row_ordinal"], "wbgt_status": taken["wbgt_status"], "wbgt_c": taken["wbgt_c"],
                        "tg_status": taken["tg_status"], "tg_c": taken["tg_c"], "tnwb_status": taken["tnwb_status"], "tnwb_c": taken["tnwb_c"],
                    }).select(_OUTPUT_COLUMNS)
                    if iso_a3 not in writers:
                        writers[iso_a3] = pq.ParquetWriter(temporary[iso_a3], table.schema, compression="zstd")
                    writers[iso_a3].write_table(table)
                    counts[iso_a3] += table.num_rows
                    selected += table.num_rows
                    for quantity in valid[iso_a3]:
                        valid[iso_a3][quantity] += int(pc.sum(pc.and_(pc.equal(table[f"{quantity}_status"], "ok"), pc.is_valid(table[f"{quantity}_c"]))).as_py() or 0)
            reporter.update("scanning", files_completed=index, files_total=len(files), rows_scanned=scanned, rows_selected=selected)
        missing = [item.country.name for item in selections if not counts[item.country.iso_a3]]
        if missing:
            raise ValueError(f"No result rows found for {', '.join(missing)} from {start_date} to {end_date}")
        for writer in writers.values():
            writer.close()
        reporter.update("publishing", files_completed=len(files), files_total=len(files), rows_scanned=scanned, rows_selected=selected)
        result = []
        for item in selections:
            iso_a3 = item.country.iso_a3
            item.output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary[iso_a3], item.output)
            extraction = CountryExtraction(2, datetime.now(timezone.utc).isoformat(), item.country.iso_a2, iso_a3, item.country.name, start_date.isoformat(), end_date.isoformat(), len(item.cells), counts[iso_a3], valid[iso_a3], tuple(map(str, resolved_runs)), item.output.name, item.output.stat().st_size)
            _write_json_atomic(item.output.with_name("manifest.json"), asdict(extraction))
            result.append(extraction)
        reporter.update("complete", state="complete", files_completed=len(files), files_total=len(files), rows_scanned=scanned, rows_selected=selected)
        return tuple(result)
    except KeyboardInterrupt:
        reporter.update("cancelled", state="cancelled", files_total=len(files), rows_scanned=scanned, rows_selected=selected, error="Interrupted")
        raise
    except Exception as error:
        reporter.update("failed", state="failed", files_total=len(files), rows_scanned=scanned, rows_selected=selected, error=str(error))
        raise
    finally:
        for writer in writers.values():
            writer.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)


def _resolve_boundaries(boundaries: str | Path | None, boundary_cache: str | Path | None, output: Path) -> Path:
    if boundaries is not None:
        path = Path(boundaries).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Boundary file does not exist: {path}")
        return path
    return cached_natural_earth_boundaries(Path(boundary_cache).expanduser().resolve() if boundary_cache else output.parents[2] / "boundaries")


def _deduplicate_countries(countries: Sequence[Country]) -> tuple[Country, ...]:
    return tuple({country.iso_a3: country for country in countries}.values())


def _validate_outputs(selections: Sequence[_Selection], overwrite: bool) -> None:
    if not overwrite:
        for item in selections:
            if item.output.exists() or item.output.with_name("manifest.json").exists():
                raise FileExistsError(f"Country cache already exists: {item.output}; pass overwrite=True to replace it")


def _result_files(runs: tuple[Path, ...], start_date: date, end_date: date) -> tuple[Path, ...]:
    paths = tuple(path for offset in range((end_date - start_date).days + 1) for run in runs for path in sorted((run / "results" / (start_date + timedelta(days=offset)).isoformat()).glob("*/*/results.parquet")))
    if not paths:
        raise FileNotFoundError(f"No result files found from {start_date} to {end_date}")
    return paths


def _merge_ranges(ranges: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    merged: list[tuple[str, str]] = []
    for lower, upper in sorted(ranges):
        if merged and lower <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], upper)
        else:
            merged.append((lower, upper))
    return tuple(merged)


def _parquet_may_overlap(path: Path, ranges: Sequence[tuple[str, str]]) -> bool:
    metadata = pq.read_metadata(path)
    if "row_id" not in metadata.schema.names:
        return True
    row = metadata.schema.names.index("row_id")
    for index in range(metadata.num_row_groups):
        stats = metadata.row_group(index).column(row).statistics
        if stats is None or not stats.has_min_max:
            return True
        minimum, maximum = _statistic_text(stats.min), _statistic_text(stats.max)
        if minimum is None or maximum is None:
            return True
        if any(maximum >= lower and minimum < upper for lower, upper in ranges):
            return True
    return False


def _select_result_files(files: Sequence[Path], ranges: Sequence[tuple[str, str]]) -> tuple[Path, ...]:
    return tuple(path for path in files if _parquet_may_overlap(path, ranges))


def _statistic_text(value: object) -> str | None:
    return value if isinstance(value, str) else value.decode() if isinstance(value, bytes) else None


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".parquet.tmp")
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
