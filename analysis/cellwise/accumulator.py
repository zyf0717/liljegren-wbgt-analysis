"""Dense, disk-backed per-cell raw-WBGT histogram accumulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..comparison.distributions import WBGTHistogramSpec


def counter_dtype(expected_maximum_observations: int) -> np.dtype[np.unsignedinteger]:
    """Choose the smallest supported counter that cannot overflow by design."""

    if expected_maximum_observations < 1:
        raise ValueError("expected_maximum_observations must be positive")
    if expected_maximum_observations <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if expected_maximum_observations <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise ValueError("Requested temporal range exceeds uint32 counter capacity")


class CellwiseHistogramAccumulator:
    """One implementation's bounded-memory cell × WBGT-bin accumulator."""

    def __init__(
        self,
        work_directory: Path,
        implementation: str,
        *,
        cell_count: int,
        histogram_spec: WBGTHistogramSpec,
        dtype: np.dtype[np.unsignedinteger],
        mode: str,
    ) -> None:
        self.work_directory = work_directory
        self.implementation = implementation
        self.cell_count = cell_count
        self.histogram_spec = histogram_spec
        self.dtype = dtype
        self.histogram_path = (
            work_directory / f"{implementation}.hist.{dtype.name}.mmap"
        )
        self.valid_path = work_directory / f"{implementation}.valid.{dtype.name}.mmap"
        self.underflow_path = (
            work_directory / f"{implementation}.underflow.{dtype.name}.mmap"
        )
        self.overflow_path = (
            work_directory / f"{implementation}.overflow.{dtype.name}.mmap"
        )
        self.histogram = np.memmap(
            self.histogram_path,
            dtype=dtype,
            mode=mode,
            shape=(cell_count, histogram_spec.bin_count),
        )
        self.valid = np.memmap(
            self.valid_path, dtype=dtype, mode=mode, shape=(cell_count,)
        )
        self.underflow = np.memmap(
            self.underflow_path, dtype=dtype, mode=mode, shape=(cell_count,)
        )
        self.overflow = np.memmap(
            self.overflow_path, dtype=dtype, mode=mode, shape=(cell_count,)
        )
        if mode == "w+":
            # ``w+`` truncates/extends each file, whose new bytes read as zero.
            # Do not eagerly touch the dense histogram: doing so physically
            # writes about 4.93 GB for every annual global replay unit and
            # defeats the sparse-file benefit of a new memmap.
            self.flush()

    @property
    def paths(self) -> tuple[Path, ...]:
        return (
            self.histogram_path,
            self.valid_path,
            self.underflow_path,
            self.overflow_path,
        )

    def update(self, cell_ids: np.ndarray, wbgt_c: np.ndarray) -> None:
        """Apply one daily batch using sparse grouped updates.

        ``np.unique`` only allocates for observed `(cell, bin)` combinations.
        It was selected over a flattened ``bincount`` because the latter needs
        a temporary spanning every global cell/bin combination (about 20 GB
        at the default global grid), independent of a daily shard's size.
        """

        cells = np.asarray(cell_ids, dtype=np.int64)
        values = np.asarray(wbgt_c, dtype=np.float64)
        bin_ids = self.histogram_spec.bin_ids(values)
        increment_counts(self.valid, cells)
        increment_counts(
            self.underflow, cells[bin_ids == self.histogram_spec.underflow_bin_id]
        )
        increment_counts(
            self.overflow, cells[bin_ids == self.histogram_spec.overflow_bin_id]
        )
        flat_ids = cells * self.histogram_spec.bin_count + bin_ids
        unique_ids, increments = np.unique(flat_ids, return_counts=True)
        flat_histogram = self.histogram.reshape(-1)
        increment_positions(flat_histogram, unique_ids, increments)

    def flush(self) -> None:
        self.histogram.flush()
        self.valid.flush()
        self.underflow.flush()
        self.overflow.flush()

    def close(self) -> None:
        for array in (self.histogram, self.valid, self.underflow, self.overflow):
            array.flush()
            array._mmap.close()  # type: ignore[attr-defined]


def increment_counts(target: np.ndarray, positions: np.ndarray) -> None:
    unique, increments = np.unique(positions, return_counts=True)
    increment_positions(target, unique, increments)


def increment_positions(
    target: np.ndarray, positions: np.ndarray, increments: np.ndarray
) -> None:
    current = target[positions]
    if np.any(increments > np.iinfo(target.dtype).max - current):
        raise OverflowError(f"{target.dtype.name} counter overflow")
    target[positions] = current + increments
