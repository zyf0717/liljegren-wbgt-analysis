"""Joint-support cellwise temperature quantiles with bounded memory."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import monotonic

import numpy as np
import pyarrow as pa

from ..comparison.config import EXPECTED_IMPLEMENTATIONS
from ..comparison.distributions import DEFAULT_WBGT_HISTOGRAM, WBGTHistogramSpec
from ..io.results import iter_result_batches
from ..results import (
    RESULT_QUANTITIES,
    RESULT_QUANTITY_SPECS,
    RESULT_SCHEMA_ID,
    HSBRunResults,
    ResultQuantity,
)
from .accumulator import CellwiseHistogramAccumulator, counter_dtype
from .grid import CellGrid, load_grid
from .pipeline import (
    DEFAULT_PROBABILITIES,
    _atomic_json,
    _atomic_parquet,
    _build_grid,
    _date_range,
    _git_commit,
    _histogram_metadata,
    _hsb_run_paths,
    _input_files,
    _input_fingerprint,
    _validate_canonical_run_inventory,
)
from .quantiles import quantile_table

JOINT_CELLWISE_SCHEMA_VERSION = 2
JOINT_SUPPORT_SCOPE = "all-implementations-all-quantities"
HOURS_PER_DAY = 24
REPLAY_STRATEGY = "implementation-all-quantities-in-memory-v3"


@dataclass(frozen=True)
class JointMaskCache:
    """Packed daily common-support masks retained in process memory."""

    packed_by_date: dict[date, np.ndarray]
    selected_count_by_date: dict[date, int]

    def select(self, current_date: date, flat_ids: np.ndarray) -> np.ndarray:
        packed = self.packed_by_date[current_date]
        byte_ids = flat_ids >> 3
        bit_ids = flat_ids & 7
        return ((packed[byte_ids] >> bit_ids) & 1).astype(bool, copy=False)


def compute_joint_cellwise_temperature_quantiles(
    hsb_run_paths: str | Path | Sequence[str | Path],
    *,
    start_date: str | date,
    end_date: str | date,
    output_root: str | Path = Path("data/cellwise_temperature_quantiles"),
    histogram_spec: WBGTHistogramSpec = DEFAULT_WBGT_HISTOGRAM,
    probabilities: tuple[float, ...] = DEFAULT_PROBABILITIES,
    exact_validation_cells: int = 100,
    exact_validation_seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Compute WBGT, Tg and Tnwb quantiles on one strict common support.

    A daily bit-packed mask is first derived by reading all canonical
    implementations and all three quantities together. Replay then scans one
    implementation's three quantities together while updating three concurrent
    sparse disk-backed histograms.
    """

    hsb_runs = _hsb_run_paths(hsb_run_paths)
    dates = _date_range(start_date, end_date)
    _validate_canonical_run_inventory(hsb_runs, dates, EXPECTED_IMPLEMENTATIONS)
    _validate_options(probabilities, exact_validation_cells)
    root = Path(output_root).expanduser().resolve()
    if overwrite and root.exists():
        _validate_reset_target(root)
        shutil.rmtree(root)
    work = root / ".work"
    mask_directory = work / "joint_mask"
    for directory in (root, work, mask_directory):
        directory.mkdir(parents=True, exist_ok=True)

    input_files = tuple(
        path
        for implementation in EXPECTED_IMPLEMENTATIONS
        for path in _input_files(hsb_runs, dates, implementation)
    )
    input_fingerprint = _input_fingerprint(tuple(sorted(input_files)))
    grid_path = root / "grid.parquet"
    grid = (
        load_grid(grid_path)
        if grid_path.exists()
        else _build_grid(hsb_runs, dates[0], EXPECTED_IMPLEMENTATIONS[0])
    )
    if not grid_path.exists():
        _atomic_parquet(grid.table, grid_path)

    manifest_path = work / "manifest.json"
    expected_manifest = _expected_manifest(
        dates, grid, histogram_spec, probabilities, input_fingerprint
    )
    manifest = _open_manifest(manifest_path, expected_manifest, root / "metadata.json")
    if (root / "metadata.json").exists() and all(
        (root / "implementations" / f"{implementation}.parquet").exists()
        for implementation in EXPECTED_IMPLEMENTATIONS
    ):
        shutil.rmtree(work)
        return root
    manifest = _prepare_replay_manifest(manifest, manifest_path)
    _discard_legacy_replay_files(work)
    started = monotonic()

    mask_fingerprints = dict(manifest.get("mask_fingerprints", {}))
    masks_reused = 0
    masks_generated = 0
    for current_date in dates:
        mask_path = _mask_path(mask_directory, current_date)
        date_key = current_date.isoformat()
        existing_fingerprint = mask_fingerprints.get(date_key)
        if (
            _valid_mask_file(mask_path, grid.cell_count)
            and existing_fingerprint == _file_sha256(mask_path)
        ):
            masks_reused += 1
            continue
        mask_path.unlink(missing_ok=True)
        mask = _scan_joint_day_mask(
            hsb_runs,
            current_date,
            grid,
        )
        _atomic_mask(mask, mask_path)
        masks_generated += 1
        mask_fingerprints[date_key] = _file_sha256(mask_path)
        manifest["mask_fingerprints"] = mask_fingerprints
        _atomic_json(manifest, manifest_path)

    joint_masks, joint_counts = _load_joint_masks(mask_directory, dates, grid.cell_count)
    sample_cells = _sample_cells(joint_counts, exact_validation_cells, exact_validation_seed)
    dtype = counter_dtype(len(dates) * HOURS_PER_DAY)
    provenance = dict(manifest.get("unit_provenance", {}))

    for implementation in EXPECTED_IMPLEMENTATIONS:
        output_path = root / "implementations" / f"{implementation}.parquet"
        if implementation not in provenance or not output_path.exists():
            partials, implementation_provenance = _replay_implementation(
                hsb_runs=hsb_runs,
                dates=dates,
                implementation=implementation,
                grid=grid,
                joint_counts=joint_counts,
                joint_masks=joint_masks,
                histogram_spec=histogram_spec,
                probabilities=probabilities,
                dtype=dtype,
                work=work,
                sample_cells=sample_cells,
                sample_seed=exact_validation_seed,
            )
            wide = _wide_implementation_table(
                grid=grid,
                dates=dates,
                joint_counts=joint_counts,
                partials=partials,
            )
            _atomic_parquet(wide, output_path)
            provenance[implementation] = implementation_provenance
            manifest["unit_provenance"] = provenance
            _atomic_json(manifest, manifest_path)

    metadata = _final_metadata(
        dates=dates,
        grid=grid,
        histogram_spec=histogram_spec,
        probabilities=probabilities,
        input_fingerprint=input_fingerprint,
        joint_counts=joint_counts,
        dtype=dtype,
        provenance=provenance,
        mask_fingerprints=mask_fingerprints,
        masks_generated=masks_generated,
        masks_reused=masks_reused,
        elapsed_seconds=monotonic() - started,
    )
    _atomic_json(metadata, root / "metadata.json")
    shutil.rmtree(work)
    return root


def _validate_reset_target(path: Path) -> None:
    protected = {Path(path.anchor), Path.home().resolve(), Path.cwd().resolve()}
    if path in protected or path.parent == path:
        raise ValueError(f"Refusing to reset broad output path: {path}")


def _validate_options(
    probabilities: tuple[float, ...], exact_validation_cells: int
) -> None:
    if probabilities != tuple(sorted(set(probabilities))):
        raise ValueError("probabilities must be unique and ascending")
    if any(probability <= 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError("probabilities must be in (0, 1]")
    if exact_validation_cells < 0:
        raise ValueError("exact_validation_cells must not be negative")


def _expected_manifest(
    dates: tuple[date, ...],
    grid: CellGrid,
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    input_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": JOINT_CELLWISE_SCHEMA_VERSION,
        "support_scope": JOINT_SUPPORT_SCOPE,
        "result_schema_id": RESULT_SCHEMA_ID,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "implementations": list(EXPECTED_IMPLEMENTATIONS),
        "quantities": list(RESULT_QUANTITIES),
        "probabilities": list(probabilities),
        "histogram": _histogram_metadata(histogram_spec),
        "grid_fingerprint": grid.fingerprint,
        "input_fingerprint": input_fingerprint,
        "mask_encoding": "numpy-packbits-little-v1",
        "hours_per_day": HOURS_PER_DAY,
    }


def _open_manifest(
    path: Path,
    expected: dict[str, object],
    metadata_path: Path,
) -> dict[str, object]:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing joint cellwise metadata is incompatible")
    if path.exists():
        manifest = json.loads(path.read_text())
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Joint cellwise resume metadata does not match this run")
        return manifest
    manifest = {**expected, "unit_provenance": {}}
    _atomic_json(manifest, path)
    return manifest


def _prepare_replay_manifest(
    manifest: dict[str, object],
    manifest_path: Path,
) -> dict[str, object]:
    strategy = manifest.get("replay_strategy")
    if strategy == REPLAY_STRATEGY:
        return manifest
    if strategy == "implementation-all-quantities-v2":
        manifest["replay_strategy"] = REPLAY_STRATEGY
        _atomic_json(manifest, manifest_path)
        return manifest
    if strategy is not None:
        raise ValueError(f"Unsupported joint replay strategy: {strategy}")
    # Quantity-level work cannot be an implementation-level checkpoint.
    manifest["unit_provenance"] = {}
    manifest["replay_strategy"] = REPLAY_STRATEGY
    _atomic_json(manifest, manifest_path)
    return manifest


def _discard_legacy_replay_files(work: Path) -> None:
    """Drop obsolete memmaps and partials; completed wide files are checkpoints."""

    for path in work.glob("*.mmap"):
        path.unlink()
    shutil.rmtree(work / "partials", ignore_errors=True)


def _day_files(
    hsb_runs: tuple[Path, ...], current_date: date, implementation: str
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


def _grid_lookup(grid: CellGrid) -> dict[str, int]:
    return dict(
        zip(
            grid.lookup_table["source_cell_id"].to_pylist(),
            grid.lookup_table["cell_id"].to_pylist(),
            strict=True,
        )
    )


def _identity_batch(
    batch: pa.RecordBatch,
    *,
    lookup: dict[str, int],
    current_date: date,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    run_ids = batch.column(0).to_pylist()
    cycles = np.asarray(batch.column(1).to_pylist(), dtype=np.uint32)
    row_ids = batch.column(2).to_pylist()
    ordinals = np.asarray(batch.column(3).to_pylist(), dtype=np.uint32)
    date_token = current_date.strftime("%Y%m%d")
    cells: list[int] = []
    hours: list[int] = []
    for row_id in row_ids:
        if (
            row_id is None
            or len(row_id) != 39
            or row_id[28] != "-"
            or row_id[29:37] != date_token
            or not row_id[-2:].isdigit()
            or int(row_id[-2:]) > 23
            or (cell := lookup.get(row_id[:28])) is None
        ):
            raise ValueError(f"Malformed or unknown result identity on {current_date}")
        cells.append(cell)
        hours.append(int(row_id[-2:]))
    return (
        np.asarray(cells, dtype=np.int64),
        np.asarray(hours, dtype=np.int64),
        np.asarray(run_ids, dtype=object),
        cycles,
        ordinals,
    )


def _scan_joint_day_mask(
    hsb_runs: tuple[Path, ...], current_date: date, grid: CellGrid
) -> np.ndarray:
    """Build a common-support mask with disk-backed identity state.

    The fixed grid/hour index bounds memory: one implementation is scanned in
    Arrow batches while identity and validity arrays are held as temporary
    memmaps, never as observation-scale Python objects.
    """

    size = grid.cell_count * HOURS_PER_DAY
    lookup = _grid_lookup(grid)
    run_codes: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="joint-mask-") as temporary:
        root = Path(temporary)
        seen_count = np.memmap(root / "seen.u1", dtype=np.uint8, mode="w+", shape=size)
        all_valid = np.memmap(root / "valid.u1", dtype=np.uint8, mode="w+", shape=size)
        ref_run = np.memmap(root / "run.u4", dtype=np.uint32, mode="w+", shape=size)
        ref_cycle = np.memmap(root / "cycle.u4", dtype=np.uint32, mode="w+", shape=size)
        ref_ordinal = np.memmap(root / "ordinal.u4", dtype=np.uint32, mode="w+", shape=size)
        seen_count[:] = 0
        all_valid[:] = 1
        for implementation in EXPECTED_IMPLEMENTATIONS:
            implementation_seen = np.memmap(
                root / f"{implementation}.seen.u1", dtype=np.uint8, mode="w+", shape=size
            )
            implementation_seen[:] = 0
            columns = (
                "cycle_ordinal", "row_id", "row_ordinal",
                "wbgt_status", "wbgt_c", "tg_status", "tg_c", "tnwb_status", "tnwb_c",
            )
            for item in _day_files(hsb_runs, current_date, implementation):
                for batch in iter_result_batches((item,), columns):
                    annotated = pa.RecordBatch.from_arrays(
                        [pa.array([item.run_id] * batch.num_rows), *batch.columns],
                        names=("run_id", *columns),
                    )
                    cells, hours, runs, cycles, ordinals = _identity_batch(
                        annotated, lookup=lookup, current_date=current_date
                    )
                    flat = cells * HOURS_PER_DAY + hours
                    if len(np.unique(flat)) != len(flat) or implementation_seen[flat].any():
                        raise ValueError(f"Duplicate or inconsistent result identity on {current_date}")
                    implementation_seen[flat] = 1
                    encoded_runs = np.asarray(
                        [run_codes.setdefault(run, len(run_codes) + 1) for run in runs], dtype=np.uint32
                    )
                    first = seen_count[flat] == 0
                    if np.any(~first) and (
                        np.any(ref_run[flat[~first]] != encoded_runs[~first])
                        or np.any(ref_cycle[flat[~first]] != cycles[~first])
                        or np.any(ref_ordinal[flat[~first]] != ordinals[~first])
                    ):
                        raise ValueError(f"Duplicate or inconsistent result identity on {current_date}")
                    ref_run[flat[first]] = encoded_runs[first]
                    ref_cycle[flat[first]] = cycles[first]
                    ref_ordinal[flat[first]] = ordinals[first]
                    statuses = [batch.column(index).to_pylist() for index in (3, 5, 7)]
                    values = [np.asarray(batch.column(index).to_pylist(), dtype=np.float64) for index in (4, 6, 8)]
                    valid = np.logical_and.reduce([
                        (np.asarray(status, dtype=object) == "ok") & np.isfinite(value)
                        for status, value in zip(statuses, values, strict=True)
                    ])
                    all_valid[flat] &= valid
                    seen_count[flat] += 1
            implementation_seen.flush()
            del implementation_seen
        if not np.all(np.any(seen_count.reshape(grid.cell_count, HOURS_PER_DAY), axis=1)):
            raise ValueError(f"Cell identity coverage mismatch on {current_date}")
        mask = np.asarray((seen_count == len(EXPECTED_IMPLEMENTATIONS)) & (all_valid != 0), dtype=bool)
        for array in (seen_count, all_valid, ref_run, ref_cycle, ref_ordinal):
            array.flush()
        return mask


def _mask_path(directory: Path, current_date: date) -> Path:
    return directory / f"{current_date.isoformat()}.mask.bin"


def _valid_mask_file(path: Path, cell_count: int) -> bool:
    return path.exists() and path.stat().st_size == (cell_count * HOURS_PER_DAY + 7) // 8


def _atomic_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(mask, bitorder="little")
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(packed.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_joint_masks(
    directory: Path, dates: tuple[date, ...], cell_count: int
) -> tuple[JointMaskCache, np.ndarray]:
    counts = np.zeros(cell_count, dtype=np.uint32)
    packed_by_date: dict[date, np.ndarray] = {}
    selected_count_by_date: dict[date, int] = {}
    for current_date in dates:
        path = _mask_path(directory, current_date)
        if not _valid_mask_file(path, cell_count):
            raise ValueError(f"Invalid joint mask cache file: {path}")
        packed = np.fromfile(path, dtype=np.uint8)
        mask = np.unpackbits(
            packed, bitorder="little", count=cell_count * HOURS_PER_DAY
        )
        counts += mask.reshape(cell_count, HOURS_PER_DAY).sum(axis=1, dtype=np.uint32)
        packed_by_date[current_date] = packed
        selected_count_by_date[current_date] = int(mask.sum())
    return JointMaskCache(packed_by_date, selected_count_by_date), counts


def _scan_native_valid_day(
    hsb_runs: tuple[Path, ...],
    current_date: date,
    implementation: str,
    grid: CellGrid,
) -> Iterator[
    tuple[
        np.ndarray,
        np.ndarray,
        dict[ResultQuantity, np.ndarray],
        dict[ResultQuantity, np.ndarray],
    ]
]:
    """Return bounded Arrow-derived batches containing any native-valid row."""

    lookup = _grid_lookup(grid)
    columns = (
        "cycle_ordinal", "row_id", "row_ordinal",
        "wbgt_status", "wbgt_c", "tg_status", "tg_c", "tnwb_status", "tnwb_c",
    )
    for item in _day_files(hsb_runs, current_date, implementation):
        for batch in iter_result_batches((item,), columns):
            annotated = pa.RecordBatch.from_arrays(
                [pa.array([item.run_id] * batch.num_rows), *batch.columns],
                names=("run_id", *columns),
            )
            cells, hours, _, _, _ = _identity_batch(
                annotated, lookup=lookup, current_date=current_date
            )
            values = {
                quantity: np.asarray(
                    batch.column(4 + index * 2).to_pylist(), dtype=np.float64
                )
                for index, quantity in enumerate(RESULT_QUANTITIES)
            }
            valid = {
                quantity: (
                    np.asarray(batch.column(3 + index * 2).to_pylist(), dtype=object)
                    == "ok"
                ) & np.isfinite(values[quantity])
                for index, quantity in enumerate(RESULT_QUANTITIES)
            }
            any_valid = np.logical_or.reduce(tuple(valid.values()))
            if any_valid.any():
                yield (
                    cells[any_valid], hours[any_valid],
                    {quantity: values[quantity][any_valid] for quantity in RESULT_QUANTITIES},
                    {quantity: valid[quantity][any_valid] for quantity in RESULT_QUANTITIES},
                )


def _replay_implementation(
    *,
    hsb_runs: tuple[Path, ...],
    dates: tuple[date, ...],
    implementation: str,
    grid: CellGrid,
    joint_counts: np.ndarray,
    joint_masks: JointMaskCache,
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    dtype: np.dtype[np.unsignedinteger],
    work: Path,
    sample_cells: np.ndarray,
    sample_seed: int,
) -> tuple[dict[ResultQuantity, pa.Table], dict[str, object]]:
    accumulators: dict[ResultQuantity, CellwiseHistogramAccumulator] = {}
    native_counts = {
        quantity: np.zeros(grid.cell_count, dtype=dtype)
        for quantity in RESULT_QUANTITIES
    }
    sample_index = np.full(grid.cell_count, -1, dtype=np.int32)
    sample_index[sample_cells] = np.arange(len(sample_cells), dtype=np.int32)
    exact_values = {
        quantity: [[] for _ in sample_cells] for quantity in RESULT_QUANTITIES
    }
    rows_with_any_native_valid = 0
    started = monotonic()
    try:
        for quantity in RESULT_QUANTITIES:
            accumulators[quantity] = CellwiseHistogramAccumulator(
                work,
                f"{implementation}.{quantity}",
                cell_count=grid.cell_count,
                histogram_spec=histogram_spec,
                dtype=dtype,
                mode="w+",
        )
        for current_date in dates:
            seen = np.zeros(grid.cell_count * HOURS_PER_DAY, dtype=bool)
            selected_count = 0
            for cells, hours, values, native_valid in _scan_native_valid_day(
                hsb_runs, current_date, implementation, grid
            ):
                flat = cells * HOURS_PER_DAY + hours
                if len(np.unique(flat)) != len(flat) or seen[flat].any():
                    raise ValueError(f"Duplicate {implementation} identity on {current_date}")
                seen[flat] = True
                selected = joint_masks.select(current_date, flat)
                selected_count += int(selected.sum())
                joint_cells = cells[selected]
                sample_positions = sample_index[joint_cells]
                sampled = sample_positions >= 0
                for quantity in RESULT_QUANTITIES:
                    valid = native_valid[quantity]
                    _increment_counts(native_counts[quantity], cells[valid])
                    if not valid[selected].all():
                        raise ValueError(
                            f"Joint mask selects invalid {implementation}/{quantity} "
                            f"on {current_date}"
                        )
                    joint_values = values[quantity][selected]
                    accumulators[quantity].update(joint_cells, joint_values)
                    for position, value in zip(
                        sample_positions[sampled], joint_values[sampled], strict=True
                    ):
                        exact_values[quantity][int(position)].append(float(value))
                rows_with_any_native_valid += len(cells)
            if selected_count != joint_masks.selected_count_by_date[current_date]:
                raise ValueError(
                    f"Joint count reconciliation failed for {implementation} on {current_date}"
                )
        scan_update_elapsed = monotonic() - started
        partials: dict[ResultQuantity, pa.Table] = {}
        quantity_provenance: dict[str, object] = {}
        joint_total = int(joint_counts.astype(np.uint64).sum())
        cell_ids = grid.table["cell_id"].combine_chunks().to_numpy(zero_copy_only=False)
        for quantity in RESULT_QUANTITIES:
            accumulator = accumulators[quantity]
            if not np.array_equal(accumulator.valid, joint_counts.astype(dtype)):
                raise ValueError(f"Joint cell counts differ for {implementation}/{quantity}")
            if int(accumulator.histogram.astype(np.uint64).sum()) != joint_total:
                raise ValueError(f"Histogram coverage differs for {implementation}/{quantity}")
            result = quantile_table(
                cell_ids=cell_ids,
                histogram=accumulator.histogram,
                valid=accumulator.valid,
                underflow=accumulator.underflow,
                overflow=accumulator.overflow,
                histogram_spec=histogram_spec,
                probabilities=probabilities,
            )
            _validate_regular_quantiles(result, probabilities, implementation, quantity)
            validation = _validate_exact_values(
                result,
                sample_cells,
                exact_values[quantity],
                histogram_spec,
                probabilities,
                sample_seed,
            )
            partials[quantity] = _quantity_partial(
                result, native_counts[quantity], quantity
            )
            quantity_provenance[quantity] = {
                "native_valid_observation_count": int(
                    native_counts[quantity].astype(np.uint64).sum()
                ),
                "joint_observation_count": joint_total,
                "exact_quantile_validation": validation,
            }
            accumulator.close()
            del accumulators[quantity]
        elapsed = monotonic() - started
        provenance: dict[str, object] = {
            "replay_strategy": REPLAY_STRATEGY,
            "rows_with_any_native_valid": rows_with_any_native_valid,
            "materialized_rows_per_second": (
                rows_with_any_native_valid / elapsed if elapsed else None
            ),
            "joint_histogram_updates_per_second": (
                joint_total * len(RESULT_QUANTITIES) / elapsed
                if elapsed
                else None
            ),
            "elapsed_seconds": elapsed,
            "scan_update_elapsed_seconds": scan_update_elapsed,
            "quantile_finalize_elapsed_seconds": elapsed - scan_update_elapsed,
            "counter_dtype": dtype.name,
            "histogram_memmap_bytes_each": (
                grid.cell_count * histogram_spec.bin_count * dtype.itemsize
            ),
            "concurrent_histogram_memmap_bytes": (
                grid.cell_count
                * histogram_spec.bin_count
                * dtype.itemsize
                * len(RESULT_QUANTITIES)
            ),
            "histogram_update_strategy": "sparse_grouped_unique",
            "histogram_storage": "disk-backed-memmap",
            "quantities": quantity_provenance,
        }
        return partials, provenance
    finally:
        for accumulator in accumulators.values():
            accumulator.close()


def _increment_counts(target: np.ndarray, cells: np.ndarray) -> None:
    unique, increments = np.unique(cells, return_counts=True)
    maximum = np.iinfo(target.dtype).max
    if np.any(increments > maximum - target[unique]):
        raise OverflowError(f"{target.dtype.name} native counter overflow")
    target[unique] = target[unique] + increments


def _sample_cells(counts: np.ndarray, size: int, seed: int) -> np.ndarray:
    eligible = np.flatnonzero(counts > 0)
    if not size or not len(eligible):
        return np.empty(0, dtype=np.int64)
    return np.sort(
        np.random.default_rng(seed).choice(
            eligible, size=min(size, len(eligible)), replace=False
        )
    )


def _validate_exact_values(
    result: pa.Table,
    sample_cells: np.ndarray,
    values: list[list[float]],
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    seed: int,
) -> dict[str, float | int]:
    errors: list[float] = []
    for cell_id, observations in zip(sample_cells, values, strict=True):
        if not observations:
            raise ValueError(f"No exact validation values for joint cell {cell_id}")
        ordered = np.sort(np.asarray(observations, dtype=np.float64))
        for probability in probabilities:
            label = f"p{int(probability * 100):02d}"
            exact = ordered[int(np.ceil(probability * len(ordered))) - 1]
            lower = result[f"{label}_lower_c"][int(cell_id)].as_py()
            upper = result[f"{label}_upper_c"][int(cell_id)].as_py()
            estimate = result[f"{label}_c"][int(cell_id)].as_py()
            if lower is None or upper is None or estimate is None:
                continue
            if not lower - 1e-12 <= exact < upper + 1e-12:
                raise ValueError(f"Exact {label} for cell {cell_id} is outside its bin")
            error = abs(float(estimate) - float(exact))
            if error > histogram_spec.width_c + 1e-12:
                raise ValueError(f"Histogram {label} error exceeds bin width")
            errors.append(error)
    return {
        "sampled_cell_count": len(sample_cells),
        "checked_quantile_count": len(errors),
        "max_absolute_error_c": max(errors, default=0.0),
        "mean_absolute_error_c": float(np.mean(errors)) if errors else 0.0,
        "p95_absolute_error_c": float(np.quantile(errors, 0.95)) if errors else 0.0,
        "seed": seed,
    }


def _validate_regular_quantiles(
    result: pa.Table,
    probabilities: tuple[float, ...],
    implementation: str,
    quantity: ResultQuantity,
) -> None:
    active = result["n_valid"].combine_chunks().to_numpy(zero_copy_only=False) > 0
    for probability in probabilities:
        label = f"p{int(probability * 100):02d}"
        for suffix in ("c", "lower_c", "upper_c"):
            values = result[f"{label}_{suffix}"].combine_chunks().to_numpy(
                zero_copy_only=False
            )
            if not np.isfinite(values[active]).all():
                raise ValueError(
                    f"{implementation}/{quantity} {label} reaches an open histogram edge bin"
                )


def _quantity_partial(
    result: pa.Table,
    native_counts: np.ndarray,
    quantity: ResultQuantity,
) -> pa.Table:
    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        "cell_id": result["cell_id"],
        f"{quantity}_native_valid_n": pa.array(native_counts),
    }
    for name in result.column_names:
        if name in {"cell_id", "n_valid"}:
            continue
        column = result[name]
        if name in {"underflow_n", "overflow_n"}:
            # These two Arrow arrays can be zero-copy views of accumulator
            # memmaps, which are closed immediately after this unit returns.
            column = pa.array(
                np.array(column.combine_chunks().to_numpy(zero_copy_only=False), copy=True)
            )
        columns[f"{quantity}_{name}"] = column
    return pa.table(columns)


def _wide_implementation_table(
    *,
    grid: CellGrid,
    dates: tuple[date, ...],
    joint_counts: np.ndarray,
    partials: dict[ResultQuantity, pa.Table],
) -> pa.Table:
    n_expected = len(dates) * HOURS_PER_DAY
    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        "cell_id": grid.table["cell_id"],
        "n_expected": pa.array(
            np.full(grid.cell_count, n_expected, dtype=np.uint32)
        ),
        "n_joint": pa.array(joint_counts),
        "joint_coverage_fraction": pa.array(joint_counts / n_expected),
    }
    for quantity in RESULT_QUANTITIES:
        partial = partials[quantity]
        if not partial["cell_id"].equals(grid.table["cell_id"]):
            raise ValueError(f"Partial cell order differs for {quantity}")
        native_name = f"{quantity}_native_valid_n"
        native = partial[native_name].combine_chunks().to_numpy(zero_copy_only=False)
        if np.any(native < joint_counts):
            raise ValueError(f"Native coverage is smaller than joint coverage for {quantity}")
        for name in partial.column_names:
            if name != "cell_id":
                columns[name] = partial[name]
        columns[f"{quantity}_native_coverage_fraction"] = pa.array(native / n_expected)
    return pa.table(columns)


def _final_metadata(
    *,
    dates: tuple[date, ...],
    grid: CellGrid,
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    input_fingerprint: str,
    joint_counts: np.ndarray,
    dtype: np.dtype[np.unsignedinteger],
    provenance: dict[str, object],
    mask_fingerprints: dict[str, object],
    masks_generated: int,
    masks_reused: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "schema_version": JOINT_CELLWISE_SCHEMA_VERSION,
        "support_scope": JOINT_SUPPORT_SCOPE,
        "replay_strategy": REPLAY_STRATEGY,
        "result_schema_id": RESULT_SCHEMA_ID,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "probabilities": list(probabilities),
        "histogram": _histogram_metadata(histogram_spec),
        "histogram_version": histogram_spec.version,
        "temperature_bin_min_c": histogram_spec.minimum_c,
        "temperature_bin_max_c": histogram_spec.maximum_c,
        "temperature_bin_width_c": histogram_spec.width_c,
        "cell_count": grid.cell_count,
        "grid_fingerprint": grid.fingerprint,
        "grid_definition": "hsb-run-result-cell-universe-v1",
        "input_fingerprint": input_fingerprint,
        "mask_encoding": "numpy-packbits-little-v1",
        "support_mask_fingerprint": _mask_fingerprint(
            input_fingerprint, grid.fingerprint, mask_fingerprints
        ),
        "support_mask_bytes": sum(
            (grid.cell_count * HOURS_PER_DAY + 7) // 8 for _ in dates
        ),
        "support_masks_generated_this_invocation": masks_generated,
        "support_masks_reused_this_invocation": masks_reused,
        "implementations": list(EXPECTED_IMPLEMENTATIONS),
        "quantities": list(RESULT_QUANTITIES),
        "hours_per_day": HOURS_PER_DAY,
        "expected_observations_per_cell": len(dates) * HOURS_PER_DAY,
        "joint_observation_count": int(joint_counts.astype(np.uint64).sum()),
        "cells_with_joint_observations": int(np.count_nonzero(joint_counts)),
        "counter_dtype": dtype.name,
        "unit_provenance": provenance,
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "git_commit": _git_commit(),
    }


def _mask_fingerprint(
    input_fingerprint: str,
    grid_fingerprint: str,
    mask_fingerprints: dict[str, object],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{JOINT_SUPPORT_SCOPE}\0{input_fingerprint}\0{grid_fingerprint}".encode())
    for result_date, fingerprint in sorted(mask_fingerprints.items()):
        digest.update(f"\0{result_date}\0{fingerprint}".encode())
    return digest.hexdigest()
