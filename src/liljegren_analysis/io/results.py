"""Bounded-memory Arrow readers for validated HSB result files."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..results import ResultFile, validate_result_schema


def iter_result_batches(
    files: Sequence[ResultFile], columns: Sequence[str], *, batch_size: int = 65_536
) -> Iterator[pa.RecordBatch]:
    """Yield selected Parquet columns one bounded record batch at a time."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for item in files:
        reader = pq.ParquetFile(item.path)
        validate_result_schema(reader.schema_arrow)
        yield from reader.iter_batches(columns=list(columns), batch_size=batch_size)
