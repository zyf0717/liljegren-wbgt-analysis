"""Atomic Parquet publication for compact daily pairwise summaries."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..config import (
    DEFAULT_PAIRWISE_IMPLEMENTATIONS,
    SAMPLE_SCOPES,
    implementation_names,
)
from ..io.artifacts import atomic_json, clear_outputs
from ..results import RESULT_SCHEMA_ID
from .distributions import ABS_HISTOGRAM_VERSION

PAIRWISE_COLUMNS = (
    "result_date",
    "quantity",
    "implementation_a",
    "implementation_b",
    "n",
    "sum_difference_c",
    "sum_squared_difference_c2",
    "sum_absolute_difference_c",
    "min_difference_c",
    "max_difference_c",
    "max_absolute_difference_c",
    "area_weight_sum",
    "area_weighted_difference_sum_c",
    "area_weighted_squared_difference_sum_c2",
    "area_weighted_absolute_difference_sum_c",
    "n_absolute_difference_gt_0_1_c",
    "n_absolute_difference_gt_0_2_c",
    "n_absolute_difference_gt_0_5_c",
    "area_weight_absolute_difference_gt_0_1_c",
    "area_weight_absolute_difference_gt_0_2_c",
    "area_weight_absolute_difference_gt_0_5_c",
)
ABS_HISTOGRAM_COLUMNS = (
    "result_date",
    "quantity",
    "implementation_a",
    "implementation_b",
    "abs_bin_id",
    "abs_bin_lower_c",
    "abs_bin_upper_c",
    "n",
    "area_weight_sum",
)
QUANTITY_STATUS_COLUMNS = (
    "result_date",
    "quantity",
    "implementation",
    "status",
    "value_present",
    "n",
    "area_weight_sum",
)
_TABLE_COLUMNS = {
    "pairwise": PAIRWISE_COLUMNS,
    "abs_histogram": ABS_HISTOGRAM_COLUMNS,
    "quantity_status": QUANTITY_STATUS_COLUMNS,
}
DEFAULT_PAIRWISE_OUTPUT_ROOT = Path("data/analysis/pairwise/daily")


def pairwise_metadata(
    sample_scope: str, implementations: Sequence[str] | None = None
) -> dict[str, object]:
    """Return the canonical metadata for one pairwise product scope."""

    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    metadata: dict[str, object] = {
        "sample_scope": sample_scope,
        "result_schema_id": RESULT_SCHEMA_ID,
        "quantity_scope": "joint",
        "abs_histogram_version": ABS_HISTOGRAM_VERSION,
    }
    if implementations is None and sample_scope == "all-implementations":
        implementations = DEFAULT_PAIRWISE_IMPLEMENTATIONS
    metadata["implementation_scope"] = (
        "data-derived" if implementations is None else "configured"
    )
    if implementations is not None:
        metadata["implementations"] = list(implementation_names(implementations))
    return metadata


def initialize_pairwise_output(
    output_root: str | Path,
    sample_scope: str,
    *,
    implementations: Sequence[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate or explicitly reset one pairwise output."""

    root = Path(output_root)
    if overwrite and root.exists():
        clear_outputs(root, ("dates", ".work", "metadata.json", "settings.json"))
    _write_metadata(root, sample_scope, implementations)
    return root


def completed_pairwise_dates(output_root: str | Path) -> frozenset[date]:
    """Return dates whose atomic partition manifest and tables are complete."""

    dates_root = Path(output_root) / "dates"
    if not dates_root.is_dir():
        return frozenset()
    expected_files = {f"{name}.parquet" for name in _TABLE_COLUMNS}
    completed: set[date] = set()
    for directory in dates_root.iterdir():
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            result_date = date.fromisoformat(directory.name)
            manifest = json.loads((directory / "manifest.json").read_text())
        except (ValueError, FileNotFoundError, json.JSONDecodeError):
            continue
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if (
            isinstance(manifest, dict)
            and manifest.get("result_date") == directory.name
            and isinstance(files, list)
            and set(files) == expected_files
            and all((directory / filename).is_file() for filename in expected_files)
        ):
            completed.add(result_date)
    return frozenset(completed)


def persist_date_partitions(
    output_root: Path,
    result_date: date,
    tables: dict[str, pa.Table],
) -> Path:
    """Publish one date directory after all tables and its completion record are written."""

    root = Path(output_root)
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
        atomic_json(
            {"result_date": result_date.isoformat(), "files": filenames},
            temporary / "manifest.json",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return destination
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _write_metadata(
    root: Path,
    sample_scope: str,
    implementations: Sequence[str] | None = None,
) -> None:
    metadata = pairwise_metadata(sample_scope, implementations)
    path = root / "metadata.json"
    if path.exists():
        if json.loads(path.read_text()) != metadata:
            raise ValueError("Incompatible pairwise Parquet metadata")
        return
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(metadata, path)
