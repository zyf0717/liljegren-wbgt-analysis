"""Streaming daily-shard to cellwise WBGT quantile pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from time import monotonic

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..comparison.distributions import DEFAULT_WBGT_HISTOGRAM, WBGTHistogramSpec
from ..io.results import iter_result_batches
from ..results import HSBRunResults, RESULT_SCHEMA_ID
from .accumulator import CellwiseHistogramAccumulator, counter_dtype
from .grid import CellGrid, grid_from_source_cell_ids, load_grid
from .quantiles import quantile_table


CELLWISE_SCHEMA_VERSION = 1
DEFAULT_PROBABILITIES = (0.50, 0.90, 0.95, 0.99)


def compute_cellwise_wbgt_quantiles(
    hsb_run_paths: str | Path | Sequence[str | Path],
    *,
    implementation: str,
    start_date: str | date,
    end_date: str | date,
    output_root: str | Path = Path("data/cellwise_wbgt_quantiles"),
    histogram_spec: WBGTHistogramSpec = DEFAULT_WBGT_HISTOGRAM,
    expected_observations_per_cell_per_day: int = 24,
    probabilities: tuple[float, ...] = DEFAULT_PROBABILITIES,
    exact_validation_cells: int = 100,
    exact_validation_seed: int = 0,
) -> Path:
    """Persist temporal raw-WBGT quantiles for one implementation.

    Each requested date is independently scanned from its source shards.  No
    observation-level derivative and no daily cellwise histogram is persisted;
    only one implementation's dense disk-backed histogram exists at a time.
    """

    hsb_runs = _hsb_run_paths(hsb_run_paths)
    _validate_implementation(implementation)
    dates = _date_range(start_date, end_date)
    if expected_observations_per_cell_per_day < 1:
        raise ValueError("expected_observations_per_cell_per_day must be positive")
    if probabilities != tuple(sorted(set(probabilities))):
        raise ValueError("probabilities must be unique and ascending")
    if any(probability <= 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError("probabilities must be in (0, 1]")
    if exact_validation_cells < 0:
        raise ValueError("exact_validation_cells must not be negative")

    input_files = _input_files(hsb_runs, dates, implementation)
    input_fingerprint = _input_fingerprint(input_files)
    root = Path(output_root).expanduser().resolve()
    work = root / ".work"
    root.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=True)

    grid_path = root / "grid.parquet"
    grid = (
        load_grid(grid_path)
        if grid_path.exists()
        else _build_grid(hsb_runs, dates[0], implementation)
    )
    if not grid_path.exists():
        _atomic_parquet(grid.table, grid_path)

    _validate_metadata(
        root / "metadata.json",
        grid=grid,
        dates=dates,
        histogram_spec=histogram_spec,
        probabilities=probabilities,
    )
    dtype = counter_dtype(len(dates) * expected_observations_per_cell_per_day)
    progress_path = work / f"{implementation}.progress.json"
    expected_progress = {
        "schema_version": CELLWISE_SCHEMA_VERSION,
        "implementation": implementation,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "histogram_version": histogram_spec.version,
        "histogram": _histogram_metadata(histogram_spec),
        "grid_fingerprint": grid.fingerprint,
        "input_fingerprint": input_fingerprint,
        "counter_dtype": dtype.name,
    }
    accumulator, completed = _open_accumulator(
        work,
        progress_path,
        implementation,
        grid.cell_count,
        histogram_spec,
        dtype,
        expected_progress,
    )
    started = monotonic()
    processed_rows = 0
    last_completed_date = dates[completed] if completed >= 0 else None
    day_dirty = False
    try:
        for current_date in dates[completed + 1 :]:
            _atomic_json(
                {**expected_progress, "in_progress_date": current_date.isoformat()},
                progress_path,
            )
            day_dirty = True
            processed_rows += _scan_valid_day(
                hsb_runs,
                current_date,
                implementation,
                grid,
                accumulator,
            )
            accumulator.flush()
            completed = dates.index(current_date)
            _atomic_json(
                {**expected_progress, "last_completed_date": current_date.isoformat()},
                progress_path,
            )
            last_completed_date = current_date
            day_dirty = False

        expected_valid = int(accumulator.valid.astype(np.uint64).sum())
        histogram_valid = int(accumulator.histogram.astype(np.uint64).sum())
        if expected_valid != histogram_valid:
            raise ValueError(
                f"Histogram count {histogram_valid} differs from valid count {expected_valid}"
            )
        result = quantile_table(
            cell_ids=grid.table["cell_id"].combine_chunks().to_numpy(zero_copy_only=False),
            histogram=accumulator.histogram,
            valid=accumulator.valid,
            underflow=accumulator.underflow,
            overflow=accumulator.overflow,
            histogram_spec=histogram_spec,
            probabilities=probabilities,
        )
        exact_validation = (
            _validate_exact_quantile_sample(
                hsb_runs,
                dates,
                implementation,
                grid,
                result,
                histogram_spec,
                probabilities,
                exact_validation_cells,
                exact_validation_seed,
            )
            if exact_validation_cells
            else None
        )
        output_path = root / "implementations" / f"{implementation}.parquet"
        _atomic_parquet(result, output_path)
        _write_metadata(
            root / "metadata.json",
            grid=grid,
            dates=dates,
            histogram_spec=histogram_spec,
            probabilities=probabilities,
            implementation=implementation,
            input_fingerprint=input_fingerprint,
            processed_rows=processed_rows,
            total_valid_rows=expected_valid,
            elapsed_seconds=monotonic() - started,
            exact_validation=exact_validation,
            counter_dtype=dtype,
            histogram_memmap_bytes=(
                grid.cell_count * histogram_spec.bin_count * dtype.itemsize
            ),
        )
    except BaseException:
        accumulator.close()
        if day_dirty or last_completed_date is None:
            _cleanup_work(accumulator.paths, progress_path)
        else:
            _atomic_json(
                {
                    **expected_progress,
                    "last_completed_date": last_completed_date.isoformat(),
                },
                progress_path,
            )
        raise
    else:
        accumulator.close()
        _cleanup_work(accumulator.paths, progress_path)
        return output_path


def _hsb_run_paths(
    values: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    raw = (values,) if isinstance(values, (str, Path)) else tuple(values)
    if not raw:
        raise ValueError("hsb_run_paths must not be empty")
    hsb_runs = tuple(Path(value).expanduser().resolve() for value in raw)
    if len(hsb_runs) != len(set(hsb_runs)):
        raise ValueError("hsb_run_paths must be unique")
    for hsb_run in hsb_runs:
        if not (hsb_run / "results").is_dir():
            raise FileNotFoundError(f"HSB run results directory does not exist: {hsb_run}")
    return hsb_runs


def _validate_implementation(implementation: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not implementation or any(character not in allowed for character in implementation):
        raise ValueError("implementation must be a non-empty result-directory name")


def _date_range(start: str | date, end: str | date) -> tuple[date, ...]:
    first = start if isinstance(start, date) else date.fromisoformat(start)
    last = end if isinstance(end, date) else date.fromisoformat(end)
    if first > last:
        raise ValueError("start_date must not be after end_date")
    return tuple(first + timedelta(days=offset) for offset in range((last - first).days + 1))


def _input_files(
    hsb_runs: tuple[Path, ...], dates: tuple[date, ...], implementation: str
) -> tuple[Path, ...]:
    result: list[Path] = []
    scopes: set[tuple[str, str]] = set()
    for current_date in dates:
        date_files: list[Path] = []
        for hsb_run in hsb_runs:
            date_files.extend(
                sorted(
                    (hsb_run / "results" / current_date.isoformat()).glob(
                        f"*/{implementation}/results.parquet"
                    )
                )
            )
        if not date_files:
            raise FileNotFoundError(
                f"No {implementation} result shards found for {current_date.isoformat()}"
            )
        for path in date_files:
            scope = (current_date.isoformat(), path.parents[1].name)
            if scope in scopes:
                raise ValueError(f"Overlapping result scope for {scope}: {path}")
            scopes.add(scope)
        result.extend(date_files)
    return tuple(result)


def _input_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.as_posix().encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _day_files(
    hsb_runs: tuple[Path, ...],
    current_date: date,
    implementation: str,
):
    files = tuple(
        item
        for hsb_run in hsb_runs
        for item in HSBRunResults.from_hsb_run(hsb_run).files(
            result_date=current_date, implementation=implementation
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No {implementation} result shards found for {current_date.isoformat()}"
        )
    return files


def _build_grid(
    hsb_runs: tuple[Path, ...],
    current_date: date,
    implementation: str,
) -> CellGrid:
    source_ids: set[str] = set()
    for batch in iter_result_batches(_day_files(hsb_runs, current_date, implementation), ("row_id",)):
        source_ids.update(row_id[:28] for row_id in batch.column(0).to_pylist())
    return grid_from_source_cell_ids(sorted(source_ids))


def _scan_valid_day(
    hsb_runs: tuple[Path, ...],
    current_date: date,
    implementation: str,
    grid: CellGrid,
    accumulator: CellwiseHistogramAccumulator,
) -> int:
    """Validate grid coverage and update the disk-backed histogram per batch."""

    lookup = dict(
        zip(grid.lookup_table["source_cell_id"].to_pylist(), grid.lookup_table["cell_id"].to_pylist(), strict=True)
    )
    source_cells: set[str] = set()
    matched_cells: set[int] = set()
    unknown_cells: set[str] = set()
    valid_rows = 0
    for batch in iter_result_batches(
        _day_files(hsb_runs, current_date, implementation),
        ("row_id", "wbgt_status", "wbgt_c"),
    ):
        row_ids = batch.column(0).to_pylist()
        cells = [row_id[:28] for row_id in row_ids]
        source_cells.update(cells)
        cell_ids = np.asarray([lookup.get(cell, -1) for cell in cells], dtype=np.int64)
        matched_cells.update(cell_ids[cell_ids >= 0])
        unknown_cells.update(cell for cell, cell_id in zip(cells, cell_ids, strict=True) if cell_id < 0)
        statuses = np.asarray(batch.column(1).to_pylist(), dtype=object)
        values = np.asarray(batch.column(2).to_pylist(), dtype=np.float64)
        valid = (statuses == "ok") & np.isfinite(values) & (cell_ids >= 0)
        if valid.any():
            accumulator.update(cell_ids[valid], values[valid])
            valid_rows += int(valid.sum())
    if len(source_cells) != grid.cell_count or len(matched_cells) != grid.cell_count or unknown_cells:
        raise ValueError(
            f"Cell identity coverage mismatch for {implementation} {current_date}: "
            f"source={len(source_cells)}, matched={len(matched_cells)}, "
            f"expected={grid.cell_count}, unknown={len(unknown_cells)}"
        )
    return valid_rows


def _validate_exact_quantile_sample(
    hsb_runs: tuple[Path, ...],
    dates: tuple[date, ...],
    implementation: str,
    grid: CellGrid,
    result: pa.Table,
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    sample_size: int,
    seed: int,
) -> dict[str, float | int]:
    """Validate histogram values against exact order statistics for sampled cells."""

    cell_ids = result["cell_id"].combine_chunks().to_numpy(zero_copy_only=False)
    valid = result["n_valid"].combine_chunks().to_numpy(zero_copy_only=False)
    eligible = cell_ids[valid > 0]
    if not len(eligible):
        return {
            "sampled_cell_count": 0,
            "checked_quantile_count": 0,
            "max_absolute_error_c": 0.0,
            "mean_absolute_error_c": 0.0,
            "p95_absolute_error_c": 0.0,
            "seed": seed,
        }
    selected = np.random.default_rng(seed).choice(
        eligible, size=min(sample_size, len(eligible)), replace=False
    )
    values_by_cell = {int(cell_id): [] for cell_id in selected}
    selected_set = set(values_by_cell)
    lookup = dict(
        zip(grid.lookup_table["source_cell_id"].to_pylist(), grid.lookup_table["cell_id"].to_pylist(), strict=True)
    )
    for current_date in dates:
        for batch in iter_result_batches(
            _day_files(hsb_runs, current_date, implementation),
            ("row_id", "wbgt_status", "wbgt_c"),
        ):
            for row_id, status, value in zip(*[batch.column(index).to_pylist() for index in range(3)], strict=True):
                cell_id = lookup.get(row_id[:28])
                if cell_id in selected_set and status == "ok" and value is not None and np.isfinite(value):
                    values_by_cell[cell_id].append(float(value))

    result_ids = result["cell_id"].to_pylist()
    result_index = {cell_id: index for index, cell_id in enumerate(result_ids)}
    errors: list[float] = []
    checks = 0
    for cell_id, values in values_by_cell.items():
        if not values:
            raise ValueError(
                f"Exact validation found no valid observations for sampled cell {cell_id}"
            )
        ordered = np.sort(np.asarray(values, dtype=np.float64))
        index = result_index[cell_id]
        for probability in probabilities:
            rank = int(np.ceil(probability * len(ordered))) - 1
            exact = float(ordered[rank])
            label = f"p{int(probability * 100):02d}"
            lower = result[f"{label}_lower_c"][index].as_py()
            upper = result[f"{label}_upper_c"][index].as_py()
            estimate = result[f"{label}_c"][index].as_py()
            if lower is None or upper is None or estimate is None:
                continue
            if not lower - 1e-12 <= exact < upper + 1e-12:
                raise ValueError(
                    f"Exact {label} for cell {cell_id} is outside its histogram bin"
                )
            error = abs(float(estimate) - exact)
            if error > histogram_spec.width_c + 1e-12:
                raise ValueError(
                    f"Histogram {label} error for cell {cell_id} exceeds bin width"
                )
            errors.append(error)
            checks += 1
    return {
        "sampled_cell_count": len(selected),
        "checked_quantile_count": checks,
        "max_absolute_error_c": max(errors, default=0.0),
        "mean_absolute_error_c": float(np.mean(errors)) if errors else 0.0,
        "p95_absolute_error_c": float(np.quantile(errors, 0.95)) if errors else 0.0,
        "seed": seed,
    }


def _open_accumulator(
    work: Path,
    progress_path: Path,
    implementation: str,
    cell_count: int,
    histogram_spec: WBGTHistogramSpec,
    dtype: np.dtype[np.unsignedinteger],
    expected_progress: dict[str, object],
) -> tuple[CellwiseHistogramAccumulator, int]:
    paths = tuple(
        work / f"{implementation}.{kind}.{dtype.name}.mmap"
        for kind in ("hist", "valid", "underflow", "overflow")
    )
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if "in_progress_date" in progress:
            _cleanup_work(paths, progress_path)
            raise RuntimeError("Interrupted cellwise day was discarded; rerun to start safely")
        if any(progress.get(key) != value for key, value in expected_progress.items()):
            raise ValueError("Cellwise resume metadata does not match this requested run")
        if not all(path.exists() for path in paths):
            _cleanup_work(paths, progress_path)
            raise RuntimeError(
                "Incomplete cellwise work files were discarded; rerun to start safely"
            )
        completed_date = progress.get("last_completed_date")
        start = expected_progress["start_date"]
        end = expected_progress["end_date"]
        if not isinstance(completed_date, str) or not (start <= completed_date <= end):
            raise ValueError("Cellwise resume progress has an invalid completed date")
        completed = (date.fromisoformat(completed_date) - date.fromisoformat(start)).days
        return (
            CellwiseHistogramAccumulator(
                work,
                implementation,
                cell_count=cell_count,
                histogram_spec=histogram_spec,
                dtype=dtype,
                mode="r+",
            ),
            completed,
        )
    if any(path.exists() for path in paths):
        _cleanup_work(paths, progress_path)
    return (
        CellwiseHistogramAccumulator(
            work,
            implementation,
            cell_count=cell_count,
            histogram_spec=histogram_spec,
            dtype=dtype,
            mode="w+",
        ),
        -1,
    )


def _histogram_metadata(spec: WBGTHistogramSpec) -> dict[str, float | int]:
    return {
        "minimum_c": spec.minimum_c,
        "maximum_c": spec.maximum_c,
        "width_c": spec.width_c,
        "regular_bin_count": spec.regular_bin_count,
        "bin_count": spec.bin_count,
        "version": spec.version,
    }


def _validate_metadata(
    path: Path,
    *,
    grid: CellGrid,
    dates: tuple[date, ...],
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
) -> None:
    if not path.exists():
        return
    metadata = json.loads(path.read_text())
    expected = {
        "schema_version": CELLWISE_SCHEMA_VERSION,
        "histogram_version": histogram_spec.version,
        "result_schema_id": RESULT_SCHEMA_ID,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "probabilities": list(probabilities),
        "wbgt_bin_min_c": histogram_spec.minimum_c,
        "wbgt_bin_max_c": histogram_spec.maximum_c,
        "wbgt_bin_width_c": histogram_spec.width_c,
        "cell_count": grid.cell_count,
        "grid_fingerprint": grid.fingerprint,
        "grid_definition": "hsb-run-result-cell-universe-v1",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing cellwise metadata is incompatible with this run")


def _write_metadata(
    path: Path,
    *,
    grid: CellGrid,
    dates: tuple[date, ...],
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    implementation: str,
    input_fingerprint: str,
    processed_rows: int,
    total_valid_rows: int,
    elapsed_seconds: float,
    exact_validation: dict[str, float | int] | None,
    counter_dtype: np.dtype[np.unsignedinteger],
    histogram_memmap_bytes: int,
) -> None:
    previous = json.loads(path.read_text()) if path.exists() else {}
    implementations = sorted(set(previous.get("implementations", [])) | {implementation})
    provenance = dict(previous.get("implementation_provenance", {}))
    provenance[implementation] = {
        "input_fingerprint": input_fingerprint,
        "rows_processed_this_invocation": processed_rows,
        "total_valid_observation_count": total_valid_rows,
        "rows_per_second_this_invocation": (
            processed_rows / elapsed_seconds if elapsed_seconds else None
        ),
        "counter_dtype": counter_dtype.name,
        "histogram_memmap_bytes": histogram_memmap_bytes,
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "histogram_update_strategy": "sparse_grouped_unique",
        "exact_quantile_validation": exact_validation,
    }
    _atomic_json(
        {
            "schema_version": CELLWISE_SCHEMA_VERSION,
            "histogram_version": histogram_spec.version,
            "result_schema_id": RESULT_SCHEMA_ID,
            "start_date": dates[0].isoformat(),
            "end_date": dates[-1].isoformat(),
            "probabilities": list(probabilities),
            "wbgt_bin_min_c": histogram_spec.minimum_c,
            "wbgt_bin_max_c": histogram_spec.maximum_c,
            "wbgt_bin_width_c": histogram_spec.width_c,
            "wbgt_regular_bin_count": histogram_spec.regular_bin_count,
            "wbgt_total_bin_count": histogram_spec.bin_count,
            "cell_count": grid.cell_count,
            "grid_fingerprint": grid.fingerprint,
            "grid_definition": "hsb-run-result-cell-universe-v1",
            "implementations": implementations,
            "implementation_provenance": provenance,
            "git_commit": _git_commit(),
        },
        path,
    )


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _cleanup_work(paths: tuple[Path, ...], progress_path: Path) -> None:
    for path in (*paths, progress_path):
        path.unlink(missing_ok=True)
