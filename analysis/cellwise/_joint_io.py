"""Arrow batch decoding shared by joint cellwise stages."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..grid import CellGrid
from ..results import (
    RESULT_COLUMNS,
    RESULT_QUANTITIES,
    CellwiseSource,
    ResultQuantity,
    iter_source_batches,
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


def identity_batch(
    batch: pa.RecordBatch, *, source_cell_ids: pa.Array
) -> tuple[np.ndarray, ...]:
    row_ids = batch.column("row_id")
    cells = pc.index_in(
        pc.utf8_slice_codeunits(row_ids, 0, 28), value_set=source_cell_ids
    )
    # A changed cell universe would make the dense cell/hour index invalid.
    if cells.null_count:
        raise ValueError("Result contains cells outside the analysis grid")
    hours = pc.cast(pc.utf8_slice_codeunits(row_ids, 37, 39), pa.uint8())
    return (
        np.asarray(cells, dtype=np.int64),
        np.asarray(hours, dtype=np.int64),
        np.asarray(batch.column("cycle_ordinal"), dtype=np.uint32),
        np.asarray(batch.column("row_ordinal"), dtype=np.uint32),
    )


def quantity_values_and_validity(
    batch: pa.RecordBatch,
) -> tuple[dict[ResultQuantity, np.ndarray], dict[ResultQuantity, np.ndarray]]:
    values: dict[ResultQuantity, np.ndarray] = {}
    valid: dict[ResultQuantity, np.ndarray] = {}
    for index, quantity in enumerate(RESULT_QUANTITIES):
        value_array = pc.cast(batch.column(4 + index * 2), pa.float64(), safe=True)
        valid_array = pc.fill_null(
            pc.and_(
                pc.equal(batch.column(3 + index * 2), "ok"),
                pc.is_finite(value_array),
            ),
            False,
        )
        values[quantity] = value_array.to_numpy(zero_copy_only=False)
        valid[quantity] = valid_array.to_numpy(zero_copy_only=False)
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
