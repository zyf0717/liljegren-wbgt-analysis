"""Arrow batch decoding shared by joint cellwise stages."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..io.results import iter_result_batches
from ..results import RESULT_QUANTITIES, HSBRunResults, ResultQuantity
from .grid import CellGrid

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


def day_files(hsb_runs: tuple[Path, ...], current_date: date, implementation: str):
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


def identity_batch(
    batch: pa.RecordBatch,
    *,
    source_cell_ids: pa.Array,
    current_date: date,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cycles_column = batch.column(0)
    row_ids = batch.column(1)
    ordinals_column = batch.column(2)
    date_token = current_date.strftime("%Y%m%d")
    valid_format = pc.fill_null(
        pc.and_(
            pc.equal(pc.utf8_length(row_ids), 39),
            pc.and_(
                pc.equal(pc.utf8_slice_codeunits(row_ids, 28, 29), "-"),
                pc.equal(pc.utf8_slice_codeunits(row_ids, 29, 37), date_token),
            ),
        ),
        False,
    )
    cell_indices = pc.index_in(
        pc.utf8_slice_codeunits(row_ids, 0, 28), value_set=source_cell_ids
    )
    if not pc.all(valid_format).as_py() or cell_indices.null_count:
        raise ValueError(f"Malformed or unknown result identity on {current_date}")
    hours = pc.cast(
        pc.utf8_slice_codeunits(row_ids, 37, 39), pa.uint8(), safe=True
    ).to_numpy(zero_copy_only=False)
    if np.any(hours > 23):
        raise ValueError(f"Malformed or unknown result identity on {current_date}")
    if cycles_column.null_count or ordinals_column.null_count:
        raise ValueError(f"Malformed or unknown result identity on {current_date}")
    return (
        np.asarray(cell_indices.to_numpy(zero_copy_only=False), dtype=np.int64),
        np.asarray(hours, dtype=np.int64),
        np.asarray(
            pc.cast(cycles_column, pa.uint32(), safe=True).to_numpy(
                zero_copy_only=False
            ),
            dtype=np.uint32,
        ),
        np.asarray(
            pc.cast(ordinals_column, pa.uint32(), safe=True).to_numpy(
                zero_copy_only=False
            ),
            dtype=np.uint32,
        ),
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

    source_cell_ids = grid.table["source_cell_id"].combine_chunks()
    columns = (
        "cycle_ordinal",
        "row_id",
        "row_ordinal",
        "wbgt_status",
        "wbgt_c",
        "tg_status",
        "tg_c",
        "tnwb_status",
        "tnwb_c",
    )
    for item in day_files(hsb_runs, current_date, implementation):
        for batch in iter_result_batches((item,), columns):
            cells, hours, _, _ = identity_batch(
                batch,
                source_cell_ids=source_cell_ids,
                current_date=current_date,
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
