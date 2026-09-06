"""Shared input and persistence helpers for cellwise pipelines."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

from ..comparison.distributions import WBGTHistogramSpec
from ..io.results import iter_result_batches
from ..results import HSBRunResults
from .grid import CellGrid, grid_from_source_cell_ids

DEFAULT_PROBABILITIES = (0.50, 0.90, 0.95, 0.99)


def hsb_run_paths(values: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    raw = (values,) if isinstance(values, (str, Path)) else tuple(values)
    if not raw:
        raise ValueError("hsb_run_paths must not be empty")
    runs = tuple(Path(value).expanduser().resolve() for value in raw)
    if len(runs) != len(set(runs)):
        raise ValueError("hsb_run_paths must be unique")
    for run in runs:
        HSBRunResults.from_hsb_run(run)
    return runs


def date_range(start: str | date, end: str | date) -> tuple[date, ...]:
    first = start if isinstance(start, date) else date.fromisoformat(start)
    last = end if isinstance(end, date) else date.fromisoformat(end)
    if first > last:
        raise ValueError("start_date must not be after end_date")
    return tuple(
        first + timedelta(days=offset) for offset in range((last - first).days + 1)
    )


def input_files(
    runs: tuple[Path, ...], dates: tuple[date, ...], implementation: str
) -> tuple[Path, ...]:
    files: list[Path] = []
    scopes: set[tuple[str, str]] = set()
    for current_date in dates:
        day = [
            path
            for run in runs
            for path in sorted(
                (run / "results" / current_date.isoformat()).glob(
                    f"*/{implementation}/results.parquet"
                )
            )
        ]
        if not day:
            raise FileNotFoundError(
                f"No {implementation} result shards found for {current_date.isoformat()}"
            )
        for path in day:
            scope = current_date.isoformat(), path.parents[1].name
            if scope in scopes:
                raise ValueError(f"Overlapping result scope for {scope}: {path}")
            scopes.add(scope)
        files.extend(day)
    return tuple(files)


def validate_run_inventory(
    runs: tuple[Path, ...],
    dates: tuple[date, ...],
    implementations: tuple[str, ...],
    *,
    allow_unexpected: bool = False,
) -> None:
    expected = set(implementations)
    for current_date in dates:
        entries = tuple(
            (run, path)
            for run in runs
            for path in sorted(
                (run / "results" / current_date.isoformat()).glob("*/*/results.parquet")
            )
        )
        discovered = {path.parent.name for _, path in entries}
        missing = expected - discovered
        unexpected = discovered - expected
        if missing or (unexpected and not allow_unexpected):
            raise ValueError(
                f"Implementation inventory differs for {current_date}: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        scopes = {
            implementation: tuple(
                sorted(
                    (run.as_posix(), path.parents[1].name)
                    for run, path in entries
                    if path.parent.name == implementation
                )
            )
            for implementation in implementations
        }
        reference = scopes[implementations[0]]
        if not reference or any(value != reference for value in scopes.values()):
            raise ValueError(
                f"Implementation shard scopes differ on {current_date}: {scopes}"
            )


def input_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.as_posix().encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def result_files(runs: tuple[Path, ...], current_date: date, implementation: str):
    files = tuple(
        item
        for run in runs
        for item in HSBRunResults.from_hsb_run(run).files(
            result_date=current_date, implementation=implementation
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No {implementation} result shards found for {current_date.isoformat()}"
        )
    return files


def build_grid(
    runs: tuple[Path, ...], current_date: date, implementation: str
) -> CellGrid:
    source_ids = {
        row_id[:28]
        for batch in iter_result_batches(
            result_files(runs, current_date, implementation), ("row_id",)
        )
        for row_id in batch.column(0).to_pylist()
    }
    return grid_from_source_cell_ids(sorted(source_ids))


def histogram_metadata(spec: WBGTHistogramSpec) -> dict[str, float | int]:
    return {
        "minimum_c": spec.minimum_c,
        "maximum_c": spec.maximum_c,
        "width_c": spec.width_c,
        "regular_bin_count": spec.regular_bin_count,
        "bin_count": spec.bin_count,
        "version": spec.version,
    }
