"""Discover trusted HeatStressBench shards and read bounded Arrow batches."""

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .grid import CellGrid, grid_from_source_cell_ids
from .io.artifacts import input_fingerprint

ResultQuantity = Literal["wbgt", "tg", "tnwb"]
RESULT_QUANTITIES: tuple[ResultQuantity, ...] = ("wbgt", "tg", "tnwb")
RESULT_SCHEMA_ID = "hsb-wbgt-comparison-results-parquet-float32-v1"
RESULT_COLUMNS = (
    "cycle_ordinal",
    "row_id",
    "row_ordinal",
    *(
        column
        for quantity in RESULT_QUANTITIES
        for column in (f"{quantity}_status", f"{quantity}_c")
    ),
)


@dataclass(frozen=True)
class ResultFile:
    path: Path
    result_date: date
    run_id: str
    implementation: str


def date_range(start: date, end: date) -> tuple[date, ...]:
    """Inclusive dates for one analysis request."""
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if start > end:
        raise ValueError("start_date must not be after end_date")
    return tuple(start + timedelta(days=i) for i in range((end - start).days + 1))


def discover_results(
    runs: Sequence[Path],
    dates: Sequence[date],
    implementations: Sequence[str] | None = None,
) -> tuple[ResultFile, ...]:
    """List shards once, retaining provenance and rejecting overlapping runs."""
    selected = set(implementations) if implementations is not None else None
    files = []
    scopes = set()
    for day in sorted(set(dates)):
        for run in sorted(runs):
            for path in sorted(
                (run / "results" / day.isoformat()).glob("*/*/results.parquet")
            ):
                name, run_id = path.parent.name, path.parents[1].name
                if selected is not None and name not in selected:
                    continue
                scope = (day, run_id, name)
                if scope in scopes:
                    raise ValueError(f"Overlapping result scope: {scope}")
                scopes.add(scope)
                files.append(ResultFile(path, day, run_id, name))
    return tuple(files)


def iter_result_batches(
    files: Sequence[ResultFile], columns: Sequence[str], *, batch_size: int = 65_536
) -> Iterator[pa.RecordBatch]:
    """Read only the requested columns, without loading whole shards."""
    for item in files:
        yield from pq.ParquetFile(item.path).iter_batches(
            columns=list(columns), batch_size=batch_size, use_threads=False
        )


_ROUTING_COLUMNS = ("result_date", "implementation", "run_id")


@dataclass(frozen=True)
class SourceBatch:
    batch: pa.RecordBatch
    run_id: str


@dataclass(frozen=True)
class HSBSource:
    runs: tuple[Path, ...]
    files: dict[tuple[date, str], tuple[ResultFile, ...]]
    fingerprint: str


@dataclass(frozen=True)
class CountryParquetSource:
    path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    scopes: dict[tuple[date, str], tuple[tuple[str, tuple[int, ...]], ...]]
    fingerprint: str
    manifest_fingerprint: str


CellwiseSource = HSBSource | CountryParquetSource


def resolve_source(
    hsb_paths: str | Path | Sequence[str | Path] | None,
    country_cache: str | Path | None,
    dates: tuple[date, ...],
    implementations: tuple[str, ...],
) -> CellwiseSource:
    """Resolve and validate exactly one supported cellwise input."""

    if (hsb_paths is None) == (country_cache is None):
        raise ValueError("provide exactly one of hsb_run_paths or country_cache")
    if country_cache is None:
        assert hsb_paths is not None
        hsb_paths = (hsb_paths,) if isinstance(hsb_paths, (str, Path)) else hsb_paths
        runs = tuple(Path(path).expanduser().resolve() for path in hsb_paths)
        inventory = discover_results(runs, dates, implementations)
        grouped = {}
        for item in inventory:
            grouped.setdefault((item.result_date, item.implementation), []).append(item)
        files = {key: tuple(items) for key, items in grouped.items()}
        for day in dates:
            reference = None
            for name in implementations:
                shards = files.get((day, name), ())
                scopes = tuple(sorted(item.run_id for item in shards))
                if not scopes:
                    raise ValueError(f"Missing {name} result shards on {day}")
                if reference is not None and scopes != reference:
                    raise ValueError(f"Implementation shard scopes differ on {day}")
                reference = scopes
        return HSBSource(
            runs,
            files,
            input_fingerprint(tuple(sorted(item.path for item in inventory))),
        )
    return _country_source(Path(country_cache), dates, implementations)


def iter_source_batches(
    source: CellwiseSource,
    current_date: date,
    implementation: str,
    columns: tuple[str, ...],
    *,
    batch_size: int = 65_536,
) -> Iterator[SourceBatch]:
    """Yield exact date/implementation batches with stable run provenance."""

    if isinstance(source, HSBSource):
        return _iter_hsb_batches(
            source, current_date, implementation, columns, batch_size
        )
    return _iter_country_batches(
        source, current_date, implementation, columns, batch_size
    )


def build_source_grid(
    source: CellwiseSource,
    current_date: date,
    implementations: tuple[str, ...],
) -> CellGrid:
    """Build a deterministic grid from the selected source and first date."""

    source_ids: set[str] = set()
    for implementation in implementations:
        for item in iter_source_batches(
            source, current_date, implementation, ("row_id",)
        ):
            row_ids = item.batch.column("row_id")
            source_ids.update(pc.utf8_slice_codeunits(row_ids, 0, 28).to_pylist())
    return grid_from_source_cell_ids(sorted(source_ids))


def source_metadata(source: CellwiseSource) -> dict[str, object]:
    if isinstance(source, HSBSource):
        return {"input_runs": [str(run) for run in source.runs]}
    keys = (
        "country_iso_a2",
        "country_iso_a3",
        "country_name",
        "selection_bounds",
        "start_date",
        "end_date",
        "selected_cell_count",
        "boundary_path",
    )
    return {
        "input_source": "country-parquet",
        "country_cache": str(source.path),
        "country_manifest_fingerprint": source.manifest_fingerprint,
        "country_selection": {
            key: source.manifest[key] for key in keys if key in source.manifest
        },
    }


def current_source_fingerprint(source: CellwiseSource) -> str:
    if isinstance(source, HSBSource):
        paths = tuple(
            sorted(item.path for values in source.files.values() for item in values)
        )
        return input_fingerprint(paths)
    manifest_fingerprint = hashlib.sha256(source.manifest_path.read_bytes()).hexdigest()
    return _country_fingerprint(source.path, manifest_fingerprint)


def _iter_hsb_batches(
    source: HSBSource,
    current_date: date,
    implementation: str,
    columns: tuple[str, ...],
    batch_size: int,
):
    for item in source.files[current_date, implementation]:
        for batch in iter_result_batches((item,), columns, batch_size=batch_size):
            yield SourceBatch(batch, item.run_id)


def _iter_country_batches(
    source: CountryParquetSource,
    current_date: date,
    implementation: str,
    columns: tuple[str, ...],
    batch_size: int,
):
    reader = _country_reader(str(source.path), source.fingerprint)
    routed = source.scopes[current_date, implementation]
    row_groups = sorted({group for _, groups in routed for group in groups})
    read_columns = list(dict.fromkeys((*columns, *_ROUTING_COLUMNS)))
    day = pa.scalar(current_date, type=pa.date32())
    target = pa.scalar(implementation, type=pa.string())
    for batch in reader.iter_batches(
        row_groups=row_groups,
        columns=read_columns,
        batch_size=batch_size,
        use_threads=False,
    ):
        table = pa.Table.from_batches([batch])
        selected = pc.fill_null(
            pc.and_(
                pc.equal(table["result_date"], day),
                pc.equal(table["implementation"], target),
            ),
            False,
        )
        table = table.filter(selected)
        if not table.num_rows:
            continue
        run_ids = table["run_id"].combine_chunks()
        for run_id in pc.unique(run_ids).to_pylist():
            run_table = table.filter(pc.equal(table["run_id"], run_id)).select(columns)
            for result_batch in run_table.to_batches(max_chunksize=batch_size):
                yield SourceBatch(result_batch, run_id)


def _country_source(
    raw_path: Path,
    dates: tuple[date, ...],
    implementations: tuple[str, ...],
) -> CountryParquetSource:
    path = raw_path.expanduser().resolve()
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (
        date.fromisoformat(manifest["start_date"]) > dates[0]
        or date.fromisoformat(manifest["end_date"]) < dates[-1]
    ):
        raise ValueError("Requested dates exceed the country cache range")
    manifest_fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    fingerprint = _country_fingerprint(path, manifest_fingerprint)
    if "scopes" not in manifest:
        raise ValueError(
            "Country cache lacks routing metadata; re-extract into a fresh output directory"
        )
    triples = {
        (
            date.fromisoformat(row["result_date"]),
            row["implementation"],
            row["run_id"],
        ): tuple(row["row_groups"])
        for row in manifest["scopes"]
    }
    scopes = _validate_country_inventory(triples, dates, implementations)
    return CountryParquetSource(
        path, manifest_path, manifest, scopes, fingerprint, manifest_fingerprint
    )


def _country_fingerprint(path: Path, manifest_fingerprint: str) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(path.as_posix().encode())
    digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    digest.update(manifest_fingerprint.encode())
    return digest.hexdigest()


def _validate_country_inventory(
    triples: dict[tuple[date, str, str], tuple[int, ...]],
    dates: tuple[date, ...],
    implementations: tuple[str, ...],
) -> dict[tuple[date, str], tuple[tuple[str, tuple[int, ...]], ...]]:
    result: dict[tuple[date, str], tuple[tuple[str, tuple[int, ...]], ...]] = {}
    for current_date in dates:
        reference_runs: tuple[str, ...] | None = None
        for implementation in implementations:
            values = tuple(
                sorted(
                    (run_id, groups)
                    for (day, name, run_id), groups in triples.items()
                    if day == current_date and name == implementation
                )
            )
            if not values:
                raise ValueError(
                    f"Country cache lacks {implementation} on {current_date}"
                )
            runs = tuple(run_id for run_id, _ in values)
            if reference_runs is None:
                reference_runs = runs
            elif runs != reference_runs:
                raise ValueError(f"Country cache run scopes differ on {current_date}")
            result[current_date, implementation] = values
    return result


@cache
def _country_reader(path: str, fingerprint: str) -> pq.ParquetFile:
    del fingerprint  # Part of the cache key; refresh the reader when the input changes.
    return pq.ParquetFile(path)
