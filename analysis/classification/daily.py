"""Aligned daily ERA5 WBGT classes, shared by country and cellwise summaries."""

from collections import Counter
from dataclasses import dataclass
from datetime import date

import numpy as np
import pyarrow as pa

from ..grid import identity_batch
from ..results import (
    RESULT_COLUMNS,
    CellwiseSource,
    iter_source_batches,
    valid_quantity,
)

WBGT_COLUMNS = ("cycle_ordinal", "row_id", "row_ordinal", "wbgt_status", "wbgt_c")


@dataclass
class DailyClasses:
    classes: np.ndarray
    rows: int
    cells_seen: np.ndarray
    coverage: Counter


def load_daily_classes(
    source: CellwiseSource,
    day: date,
    names: tuple[str, ...],
    thresholds: tuple[float, ...],
    source_ids: pa.Array,
    *,
    batch_size: int = 250_000,
    include_coverage: bool = False,
) -> DailyClasses:
    """Fill one implementation × cell-hour array; -1 represents invalid or missing."""
    cell_count = len(source_ids)
    size = cell_count * 24
    # Signed storage reserves -1 for missing values, without assuming few thresholds.
    classes = np.full(
        (len(names), size), -1, dtype=np.min_scalar_type(-len(thresholds) - 1)
    )
    result = DailyClasses(classes, 0, np.zeros(cell_count, dtype=bool), Counter())
    reference_seen = np.zeros(size, dtype=bool)
    reference = np.zeros((3, size), dtype=np.uint32)
    scopes = {}
    columns = RESULT_COLUMNS if include_coverage else WBGT_COLUMNS
    for index, name in enumerate(names):
        seen = np.zeros(size, dtype=bool)
        for item in iter_source_batches(
            source, day, name, columns, batch_size=batch_size
        ):
            batch = item.batch
            scope = scopes.setdefault(item.run_id, len(scopes) + 1)
            cells, hours, cycles, ordinals = identity_batch(
                batch, source_cell_ids=source_ids
            )
            flat = cells * 24 + hours
            if len(np.unique(flat)) != len(flat) or seen[flat].any():
                raise ValueError(f"Duplicate cell/hour identity for {name} on {day}")
            seen[flat] = True
            known = reference_seen[flat]
            if (
                np.any(reference[0, flat[known]] != scope)
                or np.any(reference[1, flat[known]] != cycles[known])
                or np.any(reference[2, flat[known]] != ordinals[known])
            ):
                raise ValueError(f"Conflicting result provenance for {name} on {day}")
            fresh = flat[~known]
            reference[:, fresh] = np.array(
                [np.full(len(fresh), scope), cycles[~known], ordinals[~known]]
            )
            reference_seen[flat] = True
            valid = valid_quantity(batch, "wbgt")
            classes[index, flat[valid]] = np.searchsorted(
                thresholds, np.asarray(batch["wbgt_c"])[valid], side="right"
            )
            result.rows += len(batch)
            result.cells_seen[cells] = True
            if include_coverage:
                for quantity in ("wbgt", "tg", "tnwb"):
                    grouped = pa.table(
                        {
                            "status": batch[quantity + "_status"],
                            "valid": valid_quantity(batch, quantity),
                        }
                    )
                    for row in (
                        grouped.group_by(["status", "valid"])
                        .aggregate([([], "count_all")])
                        .to_pylist()
                    ):
                        if row["status"] is not None:
                            result.coverage[
                                quantity, name, row["status"], row["valid"]
                            ] += row["count_all"]
    return result
