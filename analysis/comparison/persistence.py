"""Atomic Parquet publication for compact daily pairwise summaries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from ..io.artifacts import atomic_json, atomic_parquet, clear_outputs
from .distributions import ABS_HISTOGRAM_VERSION

if TYPE_CHECKING:
    from .pairwise import DailyPairwiseSummaries

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


def pairwise_metadata(
    sample_scope: str, implementations: Sequence[str]
) -> dict[str, object]:
    """Return the canonical metadata for one pairwise product scope."""

    return {
        "sample_scope": sample_scope,
        "quantity_scope": "joint",
        "abs_histogram_version": ABS_HISTOGRAM_VERSION,
        "implementations": list(implementations),
    }


def initialize_pairwise_output(
    output_root: str | Path,
    sample_scope: str,
    *,
    implementations: Sequence[str],
    overwrite: bool = False,
) -> Path:
    """Validate or explicitly reset one pairwise output."""

    root = Path(output_root)
    if overwrite and root.exists():
        clear_outputs(root, ("dates", ".work", "metadata.json", "settings.json"))
    metadata = pairwise_metadata(sample_scope, implementations)
    path = root / "metadata.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError("Incompatible pairwise Parquet metadata")
    atomic_json(metadata, path)
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


def persist_daily_pairwise(
    output_root: Path,
    result_date: date,
    summaries: DailyPairwiseSummaries,
) -> Path:
    """Publish the three daily tables, then mark the partition complete."""
    destination = output_root / "dates" / result_date.isoformat()
    manifest = destination / "manifest.json"
    # An interrupted rewrite must never retain an earlier completion record.
    manifest.unlink(missing_ok=True)
    filenames = []
    for name, table in vars(summaries).items():
        filename = f"{name}.parquet"
        atomic_parquet(table, destination / filename)
        filenames.append(filename)
    atomic_json({"result_date": result_date.isoformat(), "files": filenames}, manifest)
    return destination
