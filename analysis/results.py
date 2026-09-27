"""Discover trusted HeatStressBench shards and read bounded Arrow batches."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

ResultQuantity = Literal["wbgt", "tg", "tnwb"]
RESULT_QUANTITIES: tuple[ResultQuantity, ...] = ("wbgt", "tg", "tnwb")
RESULT_SCHEMA_ID = "hsb-wbgt-comparison-results-parquet-float32-v1"
RESULT_COLUMNS = (
    "cycle_ordinal",
    "row_id",
    "row_ordinal",
    *(
        column
        for quantity in RESULT_QUANTITIES
        for column in (f"{quantity}_status", f"{quantity}_c")
    ),
)


@dataclass(frozen=True)
class ResultFile:
    path: Path
    result_date: date
    run_id: str
    implementation: str


def date_range(start: date, end: date) -> tuple[date, ...]:
    """Inclusive dates for one analysis request."""
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if start > end:
        raise ValueError("start_date must not be after end_date")
    return tuple(start + timedelta(days=i) for i in range((end - start).days + 1))


def discover_results(
    runs: Sequence[Path],
    dates: Sequence[date],
    implementations: Sequence[str] | None = None,
) -> tuple[ResultFile, ...]:
    """List shards once, retaining provenance and rejecting overlapping runs."""
    selected = set(implementations) if implementations is not None else None
    files = []
    scopes = set()
    for day in sorted(set(dates)):
        for run in sorted(runs):
            for path in sorted(
                (run / "results" / day.isoformat()).glob("*/*/results.parquet")
            ):
                name, run_id = path.parent.name, path.parents[1].name
                if selected is not None and name not in selected:
                    continue
                scope = (day, run_id, name)
                if scope in scopes:
                    raise ValueError(f"Overlapping result scope: {scope}")
                scopes.add(scope)
                files.append(ResultFile(path, day, run_id, name))
    return tuple(files)


def iter_result_batches(
    files: Sequence[ResultFile], columns: Sequence[str], *, batch_size: int = 65_536
) -> Iterator[pa.RecordBatch]:
    """Read only the requested columns, without loading whole shards."""
    for item in files:
        yield from pq.ParquetFile(item.path).iter_batches(
            columns=list(columns), batch_size=batch_size, use_threads=False
        )
