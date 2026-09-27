"""Bounded-memory ERA5 country-cache classification analysis."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..config import DEFAULT_IMPLEMENTATIONS, implementation_names
from ._era5_stream import CountryAccumulator
from .core import (
    validate_thresholds,
)

_CACHE_COLUMNS = (
    "result_date",
    "run_id",
    "implementation",
    "cycle_ordinal",
    "row_id",
    "cell_id",
    "wbgt_status",
    "wbgt_c",
    "tg_status",
    "tg_c",
    "tnwb_status",
    "tnwb_c",
)
_BATCH_SIZE = 250_000


def analyze_era5_country_cache(
    cache_path: str | Path,
    start_date: date,
    end_date: date,
    thresholds_c: Sequence[float],
    *,
    implementations: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Stream one preselected country cache and retain aggregate state only."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    path = Path(cache_path).expanduser().resolve()
    thresholds = validate_thresholds(thresholds_c)
    selected = implementation_names(
        DEFAULT_IMPLEMENTATIONS if implementations is None else implementations
    )
    reader = pq.ParquetFile(path)
    accumulator = CountryAccumulator(selected, thresholds)
    selected_values = pa.array(selected, type=pa.string())
    lower = pa.scalar(start_date, type=pa.date32())
    upper = pa.scalar(end_date, type=pa.date32())
    for batch in reader.iter_batches(
        columns=list(_CACHE_COLUMNS),
        batch_size=_BATCH_SIZE,
        use_threads=False,
    ):
        table = _select_rows(
            pa.Table.from_batches([batch]), selected_values, lower, upper
        )
        if table.num_rows:
            accumulator.consume(table, selected_values)

    return accumulator.finish()


def _select_rows(
    table: pa.Table,
    implementations: pa.Array,
    lower: pa.Scalar,
    upper: pa.Scalar,
) -> pa.Table:
    dates = table["result_date"]
    in_range = pc.fill_null(
        pc.and_(pc.greater_equal(dates, lower), pc.less_equal(dates, upper)), False
    )
    selected = pc.fill_null(
        pc.is_in(table["implementation"], value_set=implementations), False
    )
    return table.filter(pc.and_(in_range, selected))
