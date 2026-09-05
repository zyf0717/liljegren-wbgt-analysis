"""Atomic Parquet publication for compact daily pairwise summaries."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..results import RESULT_SCHEMA_ID
from .config import SAMPLE_SCOPES
from .distributions import ABS_HISTOGRAM_VERSION


_TABLE_COLUMNS = {
    "pairwise": (
        "result_date", "quantity", "implementation_a", "implementation_b", "n",
        "sum_difference_c", "sum_squared_difference_c2", "sum_absolute_difference_c",
        "min_difference_c", "max_difference_c", "max_absolute_difference_c", "area_weight_sum",
        "area_weighted_difference_sum_c", "area_weighted_squared_difference_sum_c2",
        "area_weighted_absolute_difference_sum_c", "n_absolute_difference_gt_0_1_c",
        "n_absolute_difference_gt_0_2_c", "n_absolute_difference_gt_0_5_c",
        "area_weight_absolute_difference_gt_0_1_c", "area_weight_absolute_difference_gt_0_2_c",
        "area_weight_absolute_difference_gt_0_5_c",
    ),
    "abs_histogram": (
        "result_date", "quantity", "implementation_a", "implementation_b", "abs_bin_id",
        "abs_bin_lower_c", "abs_bin_upper_c", "n", "area_weight_sum",
    ),
    "quantity_status": (
        "result_date", "quantity", "implementation", "status", "value_present", "n", "area_weight_sum",
    ),
}


def persist_date_partitions(
    output_root: Path, result_date: date, tables: dict[str, pa.Table], sample_scope: str
) -> Path:
    """Validate then atomically replace one date directory and its manifest."""

    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    if set(tables) != set(_TABLE_COLUMNS):
        raise ValueError("Daily summary tables differ")
    for name, table in tables.items():
        _validate_table(name, table, result_date)
    _validate_cross_table_totals(tables)
    root = output_root.expanduser().resolve() / sample_scope / "v1"
    _write_metadata(root, sample_scope)
    dates = root / "dates"
    dates.mkdir(parents=True, exist_ok=True)
    destination = dates / result_date.isoformat()
    temporary = Path(tempfile.mkdtemp(prefix=f".{result_date.isoformat()}.", dir=dates))
    try:
        filenames: list[str] = []
        for name, table in tables.items():
            filename = f"{name}.parquet"
            pq.write_table(table, temporary / filename, compression="zstd")
            filenames.append(filename)
        _atomic_json({"result_date": result_date.isoformat(), "files": filenames}, temporary / "manifest.json")
        backup = dates / f".{result_date.isoformat()}.{uuid.uuid4().hex}.old"
        if destination.exists():
            os.replace(destination, backup)
        os.replace(temporary, destination)
        shutil.rmtree(backup, ignore_errors=True)
        return destination
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _validate_table(name: str, table: pa.Table, result_date: date) -> None:
    expected = _TABLE_COLUMNS[name]
    if tuple(table.column_names) != expected:
        raise ValueError(f"{name} result schema differs")
    if any(value != result_date for value in table["result_date"].to_pylist()):
        raise ValueError(f"{name} contains a date other than {result_date}")


def _validate_cross_table_totals(tables: dict[str, pa.Table]) -> None:
    totals = {
        (row["quantity"], row["implementation_a"], row["implementation_b"]): row
        for row in tables["pairwise"].to_pylist()
    }
    histograms: dict[tuple[str, str, str], tuple[int, float]] = {}
    for row in tables["abs_histogram"].to_pylist():
        key = row["quantity"], row["implementation_a"], row["implementation_b"]
        count, weight = histograms.get(key, (0, 0.0))
        histograms[key] = count + row["n"], weight + row["area_weight_sum"]
    if totals.keys() != histograms.keys():
        raise ValueError("Daily histogram totals differ from pairwise totals")
    for key, row in totals.items():
        count, weight = histograms[key]
        if count != row["n"] or not np.isclose(weight, row["area_weight_sum"], rtol=1e-12, atol=1e-6):
            raise ValueError("Daily histogram totals differ from pairwise totals")
    by_pair: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in totals.values():
        by_pair.setdefault((row["implementation_a"], row["implementation_b"]), []).append((row["n"], row["area_weight_sum"]))
    if any(len({count for count, _ in rows}) != 1 or not np.allclose([weight for _, weight in rows], rows[0][1], rtol=1e-12, atol=1e-6) for rows in by_pair.values()):
        raise ValueError("Joint quantity populations differ across quantities")
    by_implementation: dict[str, list[tuple[int, float]]] = {}
    for row in tables["quantity_status"].to_pylist():
        by_implementation.setdefault(row["implementation"], []).append((row["n"], row["area_weight_sum"], row["quantity"]))
    for rows in by_implementation.values():
        quantities: dict[str, tuple[int, float]] = {}
        for count, weight, quantity in rows:
            old_count, old_weight = quantities.get(quantity, (0, 0.0))
            quantities[quantity] = old_count + count, old_weight + weight
        if len({count for count, _ in quantities.values()}) != 1 or not np.allclose([weight for _, weight in quantities.values()], next(iter(quantities.values()))[1], rtol=1e-12, atol=1e-6):
            raise ValueError("Quantity status totals differ across quantities")


def _write_metadata(root: Path, sample_scope: str) -> None:
    metadata = {
        "schema_version": 1, "sample_scope": sample_scope,
        "result_schema_id": RESULT_SCHEMA_ID, "abs_histogram_version": ABS_HISTOGRAM_VERSION,
    }
    path = root / "metadata.json"
    if path.exists():
        if json.loads(path.read_text()) != metadata:
            raise ValueError("Incompatible pairwise Parquet metadata")
        return
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(metadata, path)


def _atomic_json(value: dict[str, object], path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
