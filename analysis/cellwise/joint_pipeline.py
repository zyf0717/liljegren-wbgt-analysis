"""Joint-support cellwise temperature quantile orchestration."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from ..comparison.distributions import DEFAULT_WBGT_HISTOGRAM, WBGTHistogramSpec
from ..config import (
    DEFAULT_DATE_WORKERS,
    DEFAULT_IMPLEMENTATIONS,
    DEFAULT_JOINT_OUTPUT_ROOT,
    DEFAULT_PROBABILITIES,
    implementation_names,
)
from ..grid import CellGrid, load_grid
from ..io.artifacts import atomic_json, atomic_parquet, clear_outputs
from ..results import (
    RESULT_QUANTITIES,
    RESULT_SCHEMA_ID,
    build_source_grid,
    current_source_fingerprint,
    date_range,
    resolve_source,
    source_metadata,
)
from ._joint_io import HOURS_PER_DAY
from ._joint_masks import generate_joint_masks, load_joint_masks
from ._joint_replay import (
    ReplayState,
    replay_implementations,
)
from ._joint_replay import (
    sample_cells as _sample_cells,
)
from .accumulator import counter_dtype

JOINT_CELLWISE_SCHEMA_VERSION = 2
JOINT_SUPPORT_SCOPE = "all-implementations-all-quantities"
DEFAULT_REPLAY_WORKERS = 2


def compute_joint_cellwise_temperature_quantiles(
    hsb_run_paths: str | Path | Sequence[str | Path] | None = None,
    *,
    country_cache: str | Path | None = None,
    start_date: str | date,
    end_date: str | date,
    output_root: str | Path | None = None,
    histogram_spec: WBGTHistogramSpec = DEFAULT_WBGT_HISTOGRAM,
    probabilities: tuple[float, ...] = DEFAULT_PROBABILITIES,
    exact_validation_cells: int = 100,
    exact_validation_seed: int = 0,
    implementations: Sequence[str] = DEFAULT_IMPLEMENTATIONS,
    date_workers: int = DEFAULT_DATE_WORKERS,
    replay_workers: int = DEFAULT_REPLAY_WORKERS,
    overwrite: bool = False,
) -> Path:
    """Compute WBGT, Tg and Tnwb quantiles on one strict common support.

    A daily bit-packed mask is first derived by reading all canonical
    implementations and all three quantities together. Replay then scans one
    implementation's three quantities together while updating three concurrent
    sparse disk-backed histograms.
    """

    dates = date_range(start_date, end_date)
    implementations = implementation_names(implementations)
    _validate_options(
        probabilities, exact_validation_cells, date_workers, replay_workers
    )
    if country_cache is not None and output_root is None:
        raise ValueError("country_cache requires an explicit output_root")
    root = Path(output_root or DEFAULT_JOINT_OUTPUT_ROOT).expanduser().resolve()
    if overwrite and root.exists():
        clear_outputs(
            root,
            (
                ".work",
                "grid.parquet",
                "implementations",
                "metadata.json",
                "settings.json",
            ),
        )
    work = root / ".work"
    mask_directory = work / "joint_mask"
    for directory in (root, work, mask_directory):
        directory.mkdir(parents=True, exist_ok=True)
    source = resolve_source(
        hsb_run_paths,
        country_cache,
        dates,
        implementations,
    )
    input_fingerprint = source.fingerprint
    country_metadata = source_metadata(source)
    grid_path = root / "grid.parquet"
    grid = (
        load_grid(grid_path)
        if grid_path.exists()
        else build_source_grid(source, dates[0], implementations[:1])
    )
    if not grid_path.exists():
        atomic_parquet(grid.table, grid_path)

    manifest_path = work / "manifest.json"
    expected_manifest = _expected_manifest(
        dates,
        implementations,
        grid,
        histogram_spec,
        probabilities,
        input_fingerprint,
        country_metadata,
    )
    expected_manifest.update(
        exact_validation_cells=exact_validation_cells,
        exact_validation_seed=exact_validation_seed,
    )
    manifest = _open_manifest(manifest_path, expected_manifest, root / "metadata.json")
    if (root / "metadata.json").exists() and all(
        (root / "implementations" / f"{implementation}.parquet").exists()
        for implementation in implementations
    ):
        shutil.rmtree(work)
        return root

    generate_joint_masks(
        source=source,
        implementations=implementations,
        dates=dates,
        grid=grid,
        grid_path=grid_path,
        mask_directory=mask_directory,
        workers=date_workers,
    )

    joint_masks, joint_counts = load_joint_masks(mask_directory, dates, grid.cell_count)
    sample_cells = _sample_cells(
        joint_counts, exact_validation_cells, exact_validation_seed
    )
    dtype = counter_dtype(len(dates) * HOURS_PER_DAY)
    replay_state = ReplayState(
        source,
        dates,
        grid,
        joint_masks,
        joint_counts,
        histogram_spec,
        probabilities,
        dtype,
        work,
        sample_cells,
        exact_validation_seed,
        root,
    )
    provenance = replay_implementations(
        replay_state,
        implementations,
        grid_path=grid_path,
        mask_directory=mask_directory,
        manifest=manifest,
        manifest_path=manifest_path,
        workers=replay_workers,
    )
    if current_source_fingerprint(source) != input_fingerprint:
        raise ValueError(
            "Inputs changed during quantile analysis; outputs not published"
        )

    metadata = {
        **expected_manifest,
        "histogram_version": histogram_spec.version,
        "temperature_bin_min_c": histogram_spec.minimum_c,
        "temperature_bin_max_c": histogram_spec.maximum_c,
        "temperature_bin_width_c": histogram_spec.width_c,
        "cell_count": grid.cell_count,
        "grid_definition": "hsb-run-result-cell-universe-v1",
        "expected_observations_per_cell": len(dates) * HOURS_PER_DAY,
        "joint_observation_count": int(joint_counts.sum(dtype=np.uint64)),
        "cells_with_joint_observations": int(np.count_nonzero(joint_counts)),
        "counter_dtype": dtype.name,
        "unit_provenance": provenance,
        "date_workers": date_workers,
        "replay_workers": replay_workers,
    }
    atomic_json(metadata, root / "metadata.json")
    shutil.rmtree(work)
    return root


def _validate_options(
    probabilities: tuple[float, ...],
    exact_validation_cells: int,
    date_workers: int,
    replay_workers: int,
) -> None:
    if probabilities != tuple(sorted(set(probabilities))):
        raise ValueError("probabilities must be unique and ascending")
    if any(probability <= 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError("probabilities must be in (0, 1]")
    if exact_validation_cells < 0:
        raise ValueError("exact_validation_cells must not be negative")
    if date_workers < 1:
        raise ValueError("date_workers must be positive")
    if replay_workers < 1:
        raise ValueError("replay_workers must be positive")


def _expected_manifest(
    dates: tuple[date, ...],
    implementations: tuple[str, ...],
    grid: CellGrid,
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    input_fingerprint: str,
    country_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": JOINT_CELLWISE_SCHEMA_VERSION,
        "support_scope": JOINT_SUPPORT_SCOPE,
        "result_schema_id": RESULT_SCHEMA_ID,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "implementations": list(implementations),
        "quantities": list(RESULT_QUANTITIES),
        "probabilities": list(probabilities),
        "histogram": asdict(histogram_spec),
        "grid_fingerprint": grid.fingerprint,
        "input_fingerprint": input_fingerprint,
        "mask_encoding": "numpy-packbits-little-v1",
        "hours_per_day": HOURS_PER_DAY,
        **country_metadata,
    }


def _open_manifest(
    path: Path,
    expected: dict[str, object],
    metadata_path: Path,
) -> dict[str, object]:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing joint cellwise metadata is incompatible")
    if path.exists():
        manifest = json.loads(path.read_text())
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Joint cellwise resume metadata does not match this run")
        return manifest
    manifest = {**expected, "unit_provenance": {}}
    atomic_json(manifest, path)
    return manifest
