"""Joint-support cellwise temperature quantile orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from time import monotonic

import numpy as np

from ..comparison.config import DEFAULT_ISO7243_IMPLEMENTATIONS
from ..comparison.distributions import DEFAULT_WBGT_HISTOGRAM, WBGTHistogramSpec
from ..io.artifacts import atomic_json as _atomic_json
from ..io.artifacts import atomic_parquet as _atomic_parquet
from ..io.artifacts import git_commit as _git_commit
from ..results import RESULT_QUANTITIES, RESULT_SCHEMA_ID
from ._joint_io import HOURS_PER_DAY
from ._joint_masks import generate_joint_masks, load_joint_masks
from ._joint_replay import (
    REPLAY_STRATEGY,
    replay_implementations,
    sample_cells as _sample_cells,
)
from ._pipeline import (
    DEFAULT_DATE_WORKERS,
    DEFAULT_PROBABILITIES,
    date_range as _date_range,
    histogram_metadata as _histogram_metadata,
)
from .accumulator import counter_dtype
from .grid import CellGrid, load_grid
from .paths import DEFAULT_JOINT_OUTPUT_ROOT
from ._source import (
    CountryParquetSource,
    build_source_grid,
    current_source_fingerprint,
    resolve_source,
    source_metadata,
)

JOINT_CELLWISE_SCHEMA_VERSION = 2
JOINT_SUPPORT_SCOPE = "all-implementations-all-quantities"
DEFAULT_REPLAY_WORKERS = 2
DEFAULT_JOINT_IMPLEMENTATIONS = DEFAULT_ISO7243_IMPLEMENTATIONS


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
    implementations: Sequence[str] = DEFAULT_JOINT_IMPLEMENTATIONS,
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

    dates = _date_range(start_date, end_date)
    implementations = _implementation_names(implementations)
    _validate_options(
        probabilities, exact_validation_cells, date_workers, replay_workers
    )
    if country_cache is not None and output_root is None:
        raise ValueError("country_cache requires an explicit output_root")
    root = Path(output_root or DEFAULT_JOINT_OUTPUT_ROOT).expanduser().resolve()
    if overwrite and root.exists():
        _validate_reset_target(root)
        shutil.rmtree(root)
    work = root / ".work"
    mask_directory = work / "joint_mask"
    for directory in (root, work, mask_directory):
        directory.mkdir(parents=True, exist_ok=True)
    source = resolve_source(
        hsb_run_paths,
        country_cache,
        dates,
        implementations,
        index_path=root / ".source-index.json",
    )
    input_fingerprint = source.fingerprint
    country_metadata = (
        source_metadata(source) if isinstance(source, CountryParquetSource) else {}
    )
    grid_path = root / "grid.parquet"
    grid = (
        load_grid(grid_path)
        if grid_path.exists()
        else build_source_grid(source, dates[0], implementations[:1])
    )
    if not grid_path.exists():
        _atomic_parquet(grid.table, grid_path)

    manifest_path = work / "manifest.json"
    expected_manifest = _expected_manifest(
        dates, implementations, grid, histogram_spec, probabilities,
        input_fingerprint, country_metadata,
    )
    manifest = _open_manifest(manifest_path, expected_manifest, root / "metadata.json")
    if (root / "metadata.json").exists() and all(
        (root / "implementations" / f"{implementation}.parquet").exists()
        for implementation in implementations
    ):
        shutil.rmtree(work)
        return root
    started = monotonic()

    mask_started = monotonic()
    mask_fingerprints, masks_generated, masks_reused = generate_joint_masks(
        source=source,
        implementations=implementations,
        dates=dates,
        grid=grid,
        grid_path=grid_path,
        mask_directory=mask_directory,
        manifest=manifest,
        manifest_path=manifest_path,
        workers=date_workers,
    )
    mask_elapsed = monotonic() - mask_started

    joint_masks, joint_counts = load_joint_masks(mask_directory, dates, grid.cell_count)
    sample_cells = _sample_cells(
        joint_counts, exact_validation_cells, exact_validation_seed
    )
    dtype = counter_dtype(len(dates) * HOURS_PER_DAY)
    provenance = dict(manifest.get("unit_provenance", {}))
    replay_started = monotonic()
    provenance = replay_implementations(
        source=source,
        dates=dates,
        implementations=implementations,
        grid=grid,
        grid_path=grid_path,
        joint_masks=joint_masks,
        joint_counts=joint_counts,
        mask_directory=mask_directory,
        histogram_spec=histogram_spec,
        probabilities=probabilities,
        dtype=dtype,
        work=work,
        sample_cells=sample_cells,
        sample_seed=exact_validation_seed,
        output_root=root,
        provenance=provenance,
        manifest=manifest,
        manifest_path=manifest_path,
        workers=replay_workers,
    )
    replay_elapsed = monotonic() - replay_started
    if current_source_fingerprint(source) != input_fingerprint:
        raise ValueError(
            "Inputs changed during quantile analysis; outputs not published"
        )

    metadata = _final_metadata(
        dates=dates,
        implementations=implementations,
        grid=grid,
        histogram_spec=histogram_spec,
        probabilities=probabilities,
        input_fingerprint=input_fingerprint,
        joint_counts=joint_counts,
        dtype=dtype,
        provenance=provenance,
        mask_fingerprints=mask_fingerprints,
        masks_generated=masks_generated,
        masks_reused=masks_reused,
        date_workers=date_workers,
        replay_workers=replay_workers,
        mask_elapsed_seconds=mask_elapsed,
        replay_elapsed_seconds=replay_elapsed,
        elapsed_seconds=monotonic() - started,
        country_metadata=country_metadata,
    )
    _atomic_json(metadata, root / "metadata.json")
    shutil.rmtree(work)
    return root


def _validate_reset_target(path: Path) -> None:
    protected = {Path(path.anchor), Path.home().resolve(), Path.cwd().resolve()}
    if path in protected or path.parent == path:
        raise ValueError(f"Refusing to reset broad output path: {path}")


def _implementation_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("implementations must be a sequence, not a string")
    implementations = tuple(values)
    if len(implementations) < 2:
        raise ValueError("implementations must contain at least two targets")
    if any(not value for value in implementations):
        raise ValueError("implementation names must not be empty")
    if len(set(implementations)) != len(implementations):
        raise ValueError("implementations must be unique")
    return implementations


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
        "histogram": _histogram_metadata(histogram_spec),
        "grid_fingerprint": grid.fingerprint,
        "input_fingerprint": input_fingerprint,
        "mask_encoding": "numpy-packbits-little-v1",
        "hours_per_day": HOURS_PER_DAY,
        "replay_strategy": REPLAY_STRATEGY,
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
    _atomic_json(manifest, path)
    return manifest


def _final_metadata(
    *,
    dates: tuple[date, ...],
    implementations: tuple[str, ...],
    grid: CellGrid,
    histogram_spec: WBGTHistogramSpec,
    probabilities: tuple[float, ...],
    input_fingerprint: str,
    joint_counts: np.ndarray,
    dtype: np.dtype[np.unsignedinteger],
    provenance: dict[str, object],
    mask_fingerprints: dict[str, object],
    masks_generated: int,
    masks_reused: int,
    date_workers: int,
    replay_workers: int,
    mask_elapsed_seconds: float,
    replay_elapsed_seconds: float,
    elapsed_seconds: float,
    country_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": JOINT_CELLWISE_SCHEMA_VERSION,
        "support_scope": JOINT_SUPPORT_SCOPE,
        "replay_strategy": REPLAY_STRATEGY,
        "result_schema_id": RESULT_SCHEMA_ID,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "probabilities": list(probabilities),
        "histogram": _histogram_metadata(histogram_spec),
        "histogram_version": histogram_spec.version,
        "temperature_bin_min_c": histogram_spec.minimum_c,
        "temperature_bin_max_c": histogram_spec.maximum_c,
        "temperature_bin_width_c": histogram_spec.width_c,
        "cell_count": grid.cell_count,
        "grid_fingerprint": grid.fingerprint,
        "grid_definition": "hsb-run-result-cell-universe-v1",
        "input_fingerprint": input_fingerprint,
        "mask_encoding": "numpy-packbits-little-v1",
        "support_mask_fingerprint": _mask_fingerprint(
            input_fingerprint, grid.fingerprint, mask_fingerprints
        ),
        "support_mask_bytes": sum(
            (grid.cell_count * HOURS_PER_DAY + 7) // 8 for _ in dates
        ),
        "support_masks_generated_this_invocation": masks_generated,
        "support_masks_reused_this_invocation": masks_reused,
        "date_workers_this_invocation": date_workers,
        "replay_workers_this_invocation": replay_workers,
        "mask_elapsed_seconds_this_invocation": mask_elapsed_seconds,
        "replay_elapsed_seconds_this_invocation": replay_elapsed_seconds,
        "implementations": list(implementations),
        "quantities": list(RESULT_QUANTITIES),
        "hours_per_day": HOURS_PER_DAY,
        "expected_observations_per_cell": len(dates) * HOURS_PER_DAY,
        "joint_observation_count": int(joint_counts.astype(np.uint64).sum()),
        "cells_with_joint_observations": int(np.count_nonzero(joint_counts)),
        "counter_dtype": dtype.name,
        "unit_provenance": provenance,
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "git_commit": _git_commit(),
        **country_metadata,
    }


def _mask_fingerprint(
    input_fingerprint: str,
    grid_fingerprint: str,
    mask_fingerprints: dict[str, object],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{JOINT_SUPPORT_SCOPE}\0{input_fingerprint}\0{grid_fingerprint}".encode()
    )
    for result_date, fingerprint in sorted(mask_fingerprints.items()):
        digest.update(f"\0{result_date}\0{fingerprint}".encode())
    return digest.hexdigest()
