"""Arrow batch decoding shared by joint cellwise stages."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import numpy as np
import pyarrow as pa

from ..grid import CellGrid, identity_batch
from ..results import (
    RESULT_COLUMNS,
    RESULT_QUANTITIES,
    CellwiseSource,
    ResultQuantity,
    iter_source_batches,
    valid_quantity,
)

HOURS_PER_DAY = 24


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


def quantity_values_and_validity(
    batch: pa.RecordBatch,
) -> tuple[dict[ResultQuantity, np.ndarray], dict[ResultQuantity, np.ndarray]]:
    values: dict[ResultQuantity, np.ndarray] = {}
    valid: dict[ResultQuantity, np.ndarray] = {}
    for quantity in RESULT_QUANTITIES:
        values[quantity] = np.asarray(batch[quantity + "_c"], dtype=np.float64)
        valid[quantity] = valid_quantity(batch, quantity)
    return values, valid


def scan_native_valid_day(
    source: CellwiseSource,
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

    source_cell_ids = grid.table["source_cell_id"].combine_chunks()
    for item in iter_source_batches(
        source, current_date, implementation, RESULT_COLUMNS
    ):
        batch = item.batch
        cells, hours, _, _ = identity_batch(
            batch,
            source_cell_ids=source_cell_ids,
        )
        values, valid = quantity_values_and_validity(batch)
        any_valid = np.logical_or.reduce(tuple(valid.values()))
        if any_valid.any():
            yield (
                cells[any_valid],
                hours[any_valid],
                {
                    quantity: values[quantity][any_valid]
                    for quantity in RESULT_QUANTITIES
                },
                {
                    quantity: valid[quantity][any_valid]
                    for quantity in RESULT_QUANTITIES
                },
            )
