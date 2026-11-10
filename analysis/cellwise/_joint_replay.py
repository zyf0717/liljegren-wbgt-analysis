"""Implementation replay and histogram finalization for joint cellwise quantiles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa

from ..comparison.distributions import WBGTHistogramSpec
from ..grid import CellGrid, load_grid
from ..io.artifacts import atomic_json, atomic_parquet
from ..results import RESULT_QUANTITIES, CellwiseSource, ResultQuantity
from ._joint_io import HOURS_PER_DAY, JointMaskCache, scan_native_valid_day
from ._joint_masks import load_joint_masks
from .accumulator import CellwiseHistogramAccumulator, increment_counts
from .quantiles import quantile_table


@dataclass(frozen=True)
class ReplayState:
    source: CellwiseSource
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


_REPLAY_WORKER_STATE: ReplayState | None = None


def replay_implementations(
    state: ReplayState,
    implementations: tuple[str, ...],
    *,
    grid_path: Path,
    mask_directory: Path,
    manifest: dict,
    manifest_path: Path,
    workers: int,
) -> dict:
    provenance = dict(manifest.get("unit_provenance", {}))
    pending = [
        name
        for name in implementations
        if name not in provenance
        or not (state.output_root / "implementations" / f"{name}.parquet").exists()
    ]

    def checkpoint(name, result):
        provenance[name] = result
        manifest["unit_provenance"] = provenance
        atomic_json(manifest, manifest_path)

    if workers == 1:
        for name in pending:
            checkpoint(name, _replay_implementation(state, name))
    elif pending:
        # Load large arrays once per worker; do not pickle masks for every task.
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)),
            mp_context=get_context("spawn"),
            initializer=_initialize_replay_worker,
            initargs=(
                state.source,
                state.dates,
                grid_path,
                mask_directory,
                state.histogram_spec,
                state.probabilities,
                state.dtype,
                state.work,
                len(state.sample_cells),
                state.sample_seed,
                state.output_root,
            ),
        ) as executor:
            futures = {executor.submit(_replay_worker, name): name for name in pending}
            for future in as_completed(futures):
                checkpoint(futures[future], future.result())
    return {name: provenance[name] for name in implementations}


def _initialize_replay_worker(
    source: CellwiseSource,
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
    _REPLAY_WORKER_STATE = ReplayState(
        source=source,
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


def _replay_worker(implementation: str) -> dict:
    return _replay_implementation(_REPLAY_WORKER_STATE, implementation)


def _replay_implementation(state: ReplayState, implementation: str) -> dict:
    """Replay one implementation from its beginning and publish after all quantities finish."""
    grid, dates = state.grid, state.dates
    joint_counts, joint_masks = state.joint_counts, state.joint_masks
    histogram_spec, probabilities, dtype = (
        state.histogram_spec,
        state.probabilities,
        state.dtype,
    )
    sample_cells = state.sample_cells
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
    try:
        for quantity in RESULT_QUANTITIES:
            accumulators[quantity] = CellwiseHistogramAccumulator(
                state.work,
                f"{implementation}.{quantity}",
                cell_count=grid.cell_count,
                histogram_spec=histogram_spec,
                dtype=dtype,
            )
        for current_date in dates:
            selected_count = 0
            for cells, hours, values, native_valid in scan_native_valid_day(
                state.source, current_date, implementation, grid
            ):
                flat = cells * HOURS_PER_DAY + hours
                selected = joint_masks.select(current_date, flat)
                selected_count += int(selected.sum())
                joint_cells = cells[selected]
                sample_positions = sample_index[joint_cells]
                sampled = sample_positions >= 0
                for quantity in RESULT_QUANTITIES:
                    valid = native_valid[quantity]
                    increment_counts(native_counts[quantity], cells[valid])
                    joint_values = values[quantity][selected]
                    accumulators[quantity].update(joint_cells, joint_values)
                    for position, value in zip(
                        sample_positions[sampled], joint_values[sampled], strict=True
                    ):
                        exact_values[quantity][int(position)].append(float(value))
            if selected_count != joint_masks.selected_count_by_date[current_date]:
                raise ValueError(
                    f"Joint count reconciliation failed for {implementation} on {current_date}"
                )
        n_expected = len(dates) * HOURS_PER_DAY
        columns = {
            "cell_id": grid.table["cell_id"],
            "n_expected": pa.array(
                np.full(grid.cell_count, n_expected, dtype=np.uint32)
            ),
            "n_joint": pa.array(joint_counts),
            "joint_coverage_fraction": pa.array(joint_counts / n_expected),
        }
        quantity_provenance: dict[str, object] = {}
        joint_total = int(joint_counts.astype(np.uint64).sum())
        cell_ids = grid.table["cell_id"].combine_chunks().to_numpy(zero_copy_only=False)
        for quantity in RESULT_QUANTITIES:
            accumulator = accumulators[quantity]
            result = quantile_table(
                cell_ids=cell_ids,
                histogram=accumulator.histogram,
                histogram_spec=histogram_spec,
                probabilities=probabilities,
            )
            if not np.array_equal(np.asarray(result["n_valid"]), joint_counts):
                raise ValueError(
                    f"Joint cell counts differ for {implementation}/{quantity}"
                )
            native = native_counts[quantity]
            if np.any(native < joint_counts):
                raise ValueError(
                    f"Native coverage is smaller than joint coverage for {quantity}"
                )
            _validate_regular_quantiles(result, probabilities, implementation, quantity)
            validation = _validate_exact_values(
                result,
                sample_cells,
                exact_values[quantity],
                histogram_spec,
                probabilities,
                state.sample_seed,
            )
            columns[f"{quantity}_native_valid_n"] = pa.array(native)
            columns.update(
                {
                    f"{quantity}_{name}": result[name]
                    for name in result.column_names
                    if name not in {"cell_id", "n_valid"}
                }
            )
            columns[f"{quantity}_native_coverage_fraction"] = pa.array(
                native / n_expected
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
        atomic_parquet(
            pa.table(columns),
            state.output_root / "implementations" / f"{implementation}.parquet",
        )
        return {"quantities": quantity_provenance}
    finally:
        for accumulator in accumulators.values():
            accumulator.close()


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
