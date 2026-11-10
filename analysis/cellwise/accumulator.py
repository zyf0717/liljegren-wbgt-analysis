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
    ) -> None:
        self.histogram_spec = histogram_spec
        # New files are sparse zero-filled storage. Incomplete replay always starts over.
        self.histogram = np.memmap(
            work_directory / f"{implementation}.hist.{dtype.name}.mmap",
            dtype=dtype,
            mode="w+",
            shape=(cell_count, histogram_spec.bin_count),
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
        flat_ids = cells * self.histogram_spec.bin_count + bin_ids
        unique_ids, increments = np.unique(flat_ids, return_counts=True)
        flat_histogram = self.histogram.reshape(-1)
        increment_positions(flat_histogram, unique_ids, increments)

    def close(self) -> None:
        self.histogram.flush()
        self.histogram._mmap.close()


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
