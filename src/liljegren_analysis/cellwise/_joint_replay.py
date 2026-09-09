"""Implementation replay and histogram finalization for joint cellwise quantiles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from multiprocessing import get_context
from pathlib import Path
from time import monotonic

import numpy as np
import pyarrow as pa

from ..comparison.distributions import WBGTHistogramSpec
from ..io.artifacts import atomic_json as _atomic_json
from ..io.artifacts import atomic_parquet as _atomic_parquet
from ..results import RESULT_QUANTITIES, ResultQuantity
from ._joint_io import HOURS_PER_DAY, JointMaskCache, scan_native_valid_day
from ._joint_masks import load_joint_masks
from .accumulator import CellwiseHistogramAccumulator
from .grid import CellGrid, load_grid
from .quantiles import quantile_table

REPLAY_STRATEGY = "implementation-all-quantities-in-memory-v3"


@dataclass(frozen=True)
class _ReplayWorkerState:
    hsb_runs: tuple[Path, ...]
    dates: tuple[date, ...]
    grid: CellGrid
    joint_masks: JointMaskCache
    joint_counts: np.ndarray
    histogram_spec: WBGTHistogramSpec
    probabilities: tuple[float, ...]
    dtype: np.dtype[np.unsignedinteger]
    work: Path
    sample_cells: np.ndarray
    sample_seed: int
    output_root: Path


_REPLAY_WORKER_STATE: _ReplayWorkerState | None = None


def replay_implementations(
    *,
    hsb_runs: tuple[Path, ...],
    dates: tuple[date, ...],
    implementations: tuple[str, ...],
    grid: CellGrid,
    grid_path: Path,
    joint_masks: JointMaskCache,
    joint_counts: np.ndarray,
    mask_directory: Path,
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    dtype: np.dtype[np.unsignedinteger],
    work: Path,
    sample_cells: np.ndarray,
    sample_seed: int,
    output_root: Path,
    provenance: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path,
    workers: int,
) -> dict[str, object]:
    pending = [
        implementation
        for implementation in implementations
        if implementation not in provenance
        or not (output_root / "implementations" / f"{implementation}.parquet").exists()
    ]

    def checkpoint(result: tuple[str, dict[str, object]]) -> None:
        implementation, implementation_provenance = result
        provenance[implementation] = implementation_provenance
        manifest["unit_provenance"] = provenance
        _atomic_json(manifest, manifest_path)

    if workers == 1:
        for implementation in pending:
            checkpoint(
                _replay_and_persist_implementation(
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
                    sample_seed=sample_seed,
                    output_root=output_root,
                )
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)),
            mp_context=get_context("spawn"),
            initializer=_initialize_replay_worker,
            initargs=(
                hsb_runs,
                dates,
                grid_path,
                mask_directory,
                histogram_spec,
                probabilities,
                dtype,
                work,
                len(sample_cells),
                sample_seed,
                output_root,
            ),
        ) as executor:
            futures = [
                executor.submit(_replay_worker, implementation)
                for implementation in pending
            ]
            for future in as_completed(futures):
                checkpoint(future.result())
    return {
        implementation: provenance[implementation] for implementation in implementations
    }


def _initialize_replay_worker(
    hsb_runs: tuple[Path, ...],
    dates: tuple[date, ...],
    grid_path: Path,
    mask_directory: Path,
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    dtype: np.dtype[np.unsignedinteger],
    work: Path,
    sample_size: int,
    sample_seed: int,
    output_root: Path,
) -> None:
    global _REPLAY_WORKER_STATE
    grid = load_grid(grid_path)
    joint_masks, joint_counts = load_joint_masks(mask_directory, dates, grid.cell_count)
    _REPLAY_WORKER_STATE = _ReplayWorkerState(
        hsb_runs=hsb_runs,
        dates=dates,
        grid=grid,
        joint_masks=joint_masks,
        joint_counts=joint_counts,
        histogram_spec=histogram_spec,
        probabilities=probabilities,
        dtype=dtype,
        work=work,
        sample_cells=sample_cells(joint_counts, sample_size, sample_seed),
        sample_seed=sample_seed,
        output_root=output_root,
    )


def _replay_worker(implementation: str) -> tuple[str, dict[str, object]]:
    if _REPLAY_WORKER_STATE is None:
        raise RuntimeError("Replay worker is not initialized")
    state = _REPLAY_WORKER_STATE
    return _replay_and_persist_implementation(
        hsb_runs=state.hsb_runs,
        dates=state.dates,
        implementation=implementation,
        grid=state.grid,
        joint_counts=state.joint_counts,
        joint_masks=state.joint_masks,
        histogram_spec=state.histogram_spec,
        probabilities=state.probabilities,
        dtype=state.dtype,
        work=state.work,
        sample_cells=state.sample_cells,
        sample_seed=state.sample_seed,
        output_root=state.output_root,
    )


def _replay_and_persist_implementation(
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
    output_root: Path,
) -> tuple[str, dict[str, object]]:
    partials, provenance = _replay_implementation(
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
        sample_seed=sample_seed,
    )
    wide = _wide_implementation_table(
        grid=grid,
        dates=dates,
        joint_counts=joint_counts,
        partials=partials,
    )
    _atomic_parquet(wide, output_root / "implementations" / f"{implementation}.parquet")
    return implementation, provenance


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
            for cells, hours, values, native_valid in scan_native_valid_day(
                hsb_runs, current_date, implementation, grid
            ):
                flat = cells * HOURS_PER_DAY + hours
                if len(np.unique(flat)) != len(flat) or seen[flat].any():
                    raise ValueError(
                        f"Duplicate {implementation} identity on {current_date}"
                    )
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
                raise ValueError(
                    f"Joint cell counts differ for {implementation}/{quantity}"
                )
            if int(accumulator.histogram.astype(np.uint64).sum()) != joint_total:
                raise ValueError(
                    f"Histogram coverage differs for {implementation}/{quantity}"
                )
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
                joint_total * len(RESULT_QUANTITIES) / elapsed if elapsed else None
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


def sample_cells(counts: np.ndarray, size: int, seed: int) -> np.ndarray:
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
            values = (
                result[f"{label}_{suffix}"]
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
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
                np.array(
                    column.combine_chunks().to_numpy(zero_copy_only=False), copy=True
                )
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
        "n_expected": pa.array(np.full(grid.cell_count, n_expected, dtype=np.uint32)),
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
            raise ValueError(
                f"Native coverage is smaller than joint coverage for {quantity}"
            )
        for name in partial.column_names:
            if name != "cell_id":
                columns[name] = partial[name]
        columns[f"{quantity}_native_coverage_fraction"] = pa.array(native / n_expected)
    return pa.table(columns)
