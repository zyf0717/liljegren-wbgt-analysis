"""Bounded-memory ERA5 country-cache classification analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ._era5_stream import CountryAccumulator
from .core import (
    ClassificationAnalysis,
    classification_implementations,
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


@dataclass(frozen=True)
class Era5ClassificationAnalysis:
    classification: ClassificationAnalysis
    observation_summary: pd.DataFrame
    quantity_coverage: pd.DataFrame

    @property
    def outputs(self) -> dict[str, pd.DataFrame]:
        return {
            "observation_summary.csv": self.observation_summary,
            "quantity_coverage.csv": self.quantity_coverage,
            **self.classification.outputs,
        }


def analyze_era5_country_cache(
    cache_path: str | Path,
    start_date: date,
    end_date: date,
    thresholds_c: Sequence[float],
    *,
    implementations: Sequence[str] | None = None,
) -> Era5ClassificationAnalysis:
    """Stream one preselected country cache and retain aggregate state only."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"country cache does not exist: {path}")
    thresholds = validate_thresholds(thresholds_c)
    selected = classification_implementations(implementations)
    reader = pq.ParquetFile(path)
    missing_columns = sorted(set(_CACHE_COLUMNS) - set(reader.schema_arrow.names))
    if missing_columns:
        raise ValueError(f"country cache is missing columns: {missing_columns}")

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

    classification, observation_summary, coverage = accumulator.finish()
    return Era5ClassificationAnalysis(classification, observation_summary, coverage)


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
