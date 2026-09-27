"""Daily common-support mask generation and persistence."""

from __future__ import annotations

import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ..grid import CellGrid, load_grid
from ..io.artifacts import atomic_path
from ..results import RESULT_COLUMNS, CellwiseSource, iter_source_batches
from ._joint_io import (
    HOURS_PER_DAY,
    JointMaskCache,
    identity_batch,
    quantity_values_and_validity,
)


@dataclass(frozen=True)
class _MaskWorkerState:
    source: CellwiseSource
    implementations: tuple[str, ...]
    grid: CellGrid
    mask_directory: Path


_MASK_WORKER_STATE: _MaskWorkerState | None = None


def generate_joint_masks(
    *,
    source: CellwiseSource,
    implementations: tuple[str, ...],
    dates: tuple[date, ...],
    grid: CellGrid,
    grid_path: Path,
    mask_directory: Path,
    workers: int,
) -> tuple[int, int]:
    pending = [
        day
        for day in dates
        if not valid_mask_file(joint_mask_path(mask_directory, day), grid.cell_count)
    ]
    reused = len(dates) - len(pending)

    if workers == 1:
        for current_date in pending:
            _scan_and_persist_joint_day_mask(
                _MaskWorkerState(source, implementations, grid, mask_directory),
                current_date,
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)),
            mp_context=get_context("spawn"),
            initializer=_initialize_mask_worker,
            initargs=(source, implementations, grid_path, mask_directory),
        ) as executor:
            futures = [
                executor.submit(_mask_worker, current_date) for current_date in pending
            ]
            for future in as_completed(futures):
                future.result()
    return len(pending), reused


def _initialize_mask_worker(
    source: CellwiseSource,
    implementations: tuple[str, ...],
    grid_path: Path,
    mask_directory: Path,
) -> None:
    global _MASK_WORKER_STATE
    _MASK_WORKER_STATE = _MaskWorkerState(
        source=source,
        implementations=implementations,
        grid=load_grid(grid_path),
        mask_directory=mask_directory,
    )


def _mask_worker(current_date: date) -> None:
    _scan_and_persist_joint_day_mask(_MASK_WORKER_STATE, current_date)


def _scan_and_persist_joint_day_mask(
    state: _MaskWorkerState, current_date: date
) -> None:
    mask = scan_joint_day_mask(
        state.source, state.implementations, current_date, state.grid
    )
    atomic_mask(mask, joint_mask_path(state.mask_directory, current_date))


def scan_joint_day_mask(
    source: CellwiseSource,
    implementations: tuple[str, ...],
    current_date: date,
    grid: CellGrid,
) -> np.ndarray:
    """Build a common-support mask with disk-backed identity state.

    The fixed grid/hour index bounds memory: one implementation is scanned in
    Arrow batches while identity and validity arrays are held as temporary
    memmaps, never as observation-scale Python objects.
    """

    size = grid.cell_count * HOURS_PER_DAY
    source_cell_ids = grid.table["source_cell_id"].combine_chunks()
    run_codes: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="joint-mask-") as temporary:
        root = Path(temporary)
        seen_count = np.memmap(root / "seen.u1", dtype=np.uint8, mode="w+", shape=size)
        all_valid = np.memmap(root / "valid.u1", dtype=np.uint8, mode="w+", shape=size)
        ref_run = np.memmap(root / "run.u4", dtype=np.uint32, mode="w+", shape=size)
        ref_cycle = np.memmap(root / "cycle.u4", dtype=np.uint32, mode="w+", shape=size)
        ref_ordinal = np.memmap(
            root / "ordinal.u4", dtype=np.uint32, mode="w+", shape=size
        )
        seen_count[:] = 0
        all_valid[:] = 1
        for implementation in implementations:
            implementation_seen = np.memmap(
                root / f"{implementation}.seen.u1",
                dtype=np.uint8,
                mode="w+",
                shape=size,
            )
            implementation_seen[:] = 0
            for item in iter_source_batches(
                source, current_date, implementation, RESULT_COLUMNS
            ):
                batch = item.batch
                cells, hours, cycles, ordinals = identity_batch(
                    batch,
                    source_cell_ids=source_cell_ids,
                )
                flat = cells * HOURS_PER_DAY + hours
                if len(np.unique(flat)) != len(flat) or implementation_seen[flat].any():
                    raise ValueError(
                        f"Duplicate or inconsistent result identity on {current_date}"
                    )
                implementation_seen[flat] = 1
                encoded_run = run_codes.setdefault(item.run_id, len(run_codes) + 1)
                first = seen_count[flat] == 0
                if np.any(~first) and (
                    np.any(ref_run[flat[~first]] != encoded_run)
                    or np.any(ref_cycle[flat[~first]] != cycles[~first])
                    or np.any(ref_ordinal[flat[~first]] != ordinals[~first])
                ):
                    raise ValueError(
                        f"Duplicate or inconsistent result identity on {current_date}"
                    )
                ref_run[flat[first]] = encoded_run
                ref_cycle[flat[first]] = cycles[first]
                ref_ordinal[flat[first]] = ordinals[first]
                _, quantity_valid = quantity_values_and_validity(batch)
                valid = np.logical_and.reduce(tuple(quantity_valid.values()))
                all_valid[flat] &= valid
                seen_count[flat] += 1
            implementation_seen.flush()
            del implementation_seen
        if not np.all(
            np.any(seen_count.reshape(grid.cell_count, HOURS_PER_DAY), axis=1)
        ):
            raise ValueError(f"Cell identity coverage mismatch on {current_date}")
        mask = np.asarray(
            (seen_count == len(implementations)) & (all_valid != 0), dtype=bool
        )
        for array in (seen_count, all_valid, ref_run, ref_cycle, ref_ordinal):
            array.flush()
        return mask


def joint_mask_path(directory: Path, current_date: date) -> Path:
    return directory / f"{current_date.isoformat()}.mask.bin"


def valid_mask_file(path: Path, cell_count: int) -> bool:
    return (
        path.exists() and path.stat().st_size == (cell_count * HOURS_PER_DAY + 7) // 8
    )


def atomic_mask(mask: np.ndarray, path: Path) -> None:
    with atomic_path(path) as temporary:
        temporary.write_bytes(np.packbits(mask, bitorder="little").tobytes())


def load_joint_masks(
    directory: Path, dates: tuple[date, ...], cell_count: int
) -> tuple[JointMaskCache, np.ndarray]:
    counts = np.zeros(cell_count, dtype=np.uint32)
    packed_by_date: dict[date, np.ndarray] = {}
    selected_count_by_date: dict[date, int] = {}
    for current_date in dates:
        path = joint_mask_path(directory, current_date)
        if not valid_mask_file(path, cell_count):
            raise ValueError(f"Invalid joint mask cache file: {path}")
        packed = np.fromfile(path, dtype=np.uint8)
        mask = np.unpackbits(
            packed, bitorder="little", count=cell_count * HOURS_PER_DAY
        )
        counts += mask.reshape(cell_count, HOURS_PER_DAY).sum(axis=1, dtype=np.uint32)
        packed_by_date[current_date] = packed
        selected_count_by_date[current_date] = int(mask.sum())
    return JointMaskCache(packed_by_date, selected_count_by_date), counts
