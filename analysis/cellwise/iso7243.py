"""Resumable hourly WBGT class differences and non-consensus on shared support."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..classification.core import class_definitions, validate_thresholds
from ..classification.daily import load_daily_classes
from ..config import (
    SAMPLE_SCOPES,
    SampleScope,
    implementation_names,
)
from ..grid import load_grid
from ..io.artifacts import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    atomic_path,
    clear_outputs,
)
from ..results import (
    CellwiseSource,
    build_source_grid,
    current_source_fingerprint,
    date_range,
    resolve_source,
    source_metadata,
)

ISO7243_SCHEMA_VERSION = 2
ANALYSIS = "iso7243-hourly-wbgt-classification-v2"
SUPPORT_SCOPE = "all-implementations-wbgt-only"
PAIRWISE_SUPPORT_SCOPE = "pairwise-wbgt-only"
DEFAULT_BATCH_SIZE = 250_000


@dataclass(frozen=True)
class ClassificationState:
    source: CellwiseSource
    names: tuple[str, ...]
    thresholds: tuple[float, ...]
    source_ids: pa.Array
    cell_count: int
    batch_size: int
    signature: str
    work: Path
    sample_scope: SampleScope


_WORKER: ClassificationState | None = None


def analyze_cellwise_iso7243_classification(
    hsb_run_paths: str | Path | Sequence[str | Path] | None = None,
    *,
    country_cache: str | Path | None = None,
    start_date: str | date,
    end_date: str | date,
    thresholds_c: Sequence[float],
    implementations: Sequence[str],
    sample_scope: SampleScope = "all-implementations",
    output_root: str | Path,
    date_workers: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
) -> Path:
    """Classify raw hourly WBGT once and persist annual directional counts.

    Compatible checkpoints and completed outputs are reused automatically.
    Overwrite discards this workflow's checkpoints and outputs.
    Source observations are never persisted here.
    Pairwise scope requires valid WBGT only in A and B for their comparison;
    non-consensus always requires all selected implementations to be valid.
    """
    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    thresholds = validate_thresholds(thresholds_c)
    names = implementation_names(implementations)
    if len(names) > 256:
        raise ValueError("Implementation IDs exceed uint8 capacity")
    if date_workers < 1 or batch_size < 1:
        raise ValueError("date_workers and batch_size must be positive")
    dates = date_range(start_date, end_date)
    if len(dates) * 24 > np.iinfo(np.uint32).max:
        raise ValueError("Date range exceeds uint32 per-cell counter capacity")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = resolve_source(
        hsb_run_paths,
        country_cache,
        dates,
        names,
    )
    fingerprint = source.fingerprint
    request = {
        "schema_version": ISO7243_SCHEMA_VERSION,
        "analysis": ANALYSIS,
        "observation_basis": "hourly",
        "classification_quantity": "wbgt",
        "support_scope": PAIRWISE_SUPPORT_SCOPE
        if sample_scope == "pairwise"
        else SUPPORT_SCOPE,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "thresholds_c": list(thresholds),
        "implementations": list(names),
        "input_fingerprint": fingerprint,
        **source_metadata(source),
    }
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode()
    ).hexdigest()
    if overwrite:
        clear_outputs(
            root,
            (
                ".work",
                "metadata.json",
                "settings.json",
                "cell_pair_differences.parquet",
                "cellwise_non_consensus.parquet",
                "cell_lookup.parquet",
                "classes.csv",
            ),
        )
    manifest_path = root / ".work" / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != request:
        raise ValueError(
            "Incompatible ISO 7243 resume configuration; use a new root or overwrite"
        )
    if (root / "metadata.json").exists():
        previous = json.loads((root / "metadata.json").read_text())
        if any(previous.get(key) != value for key, value in request.items()):
            raise ValueError("Incompatible existing ISO 7243 product")
        load_hourly_metadata(root)
        return root
    atomic_json(request, manifest_path)
    work = root / ".work"
    grid_path = work / "grid.parquet"
    if not grid_path.exists():
        atomic_parquet(build_source_grid(source, dates[0], names).table, grid_path)
    grid = load_grid(grid_path)
    if grid.cell_count > np.iinfo(np.int32).max:
        raise ValueError("Grid exceeds int32 cell-ID capacity")
    signature = hashlib.sha256(
        f"{request_hash}:{grid.fingerprint}".encode()
    ).hexdigest()
    pair_count = len(names) * (len(names) - 1) // 2
    stride = 3 if sample_scope == "pairwise" else 2
    total = np.zeros((2 + stride * pair_count, grid.cell_count), dtype=np.uint64)
    progress_label = _progress_label(source, thresholds)
    pending = []
    reused = 0
    for day in dates:
        checkpoint = _read_checkpoint(
            work / f"{day.isoformat()}.npz",
            signature,
            day,
            total.shape,
        )
        if checkpoint is None:
            pending.append(day)
        else:
            total += checkpoint
            reused += 1
    print(
        f"{progress_label}: {len(dates)} dates, {reused} cached, "
        f"{len(pending)} to scan",
        flush=True,
    )
    state = (
        source,
        names,
        thresholds,
        str(grid_path),
        batch_size,
        signature,
        str(work),
        sample_scope,
    )
    if date_workers == 1:
        _initialize_worker(*state)
        for day in pending:
            _run_day(day)
            _reduce_day(work, day, signature, total, sample_scope, progress_label)
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(date_workers, len(pending)),
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=state,
        ) as executor:
            futures = {executor.submit(_run_day, day): day for day in pending}
            try:
                for future in as_completed(futures):
                    day = futures[future]
                    future.result()
                    _reduce_day(
                        work,
                        day,
                        signature,
                        total,
                        sample_scope,
                        progress_label,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    if current_source_fingerprint(source) != fingerprint:
        raise ValueError(
            "Raw inputs changed during classification; outputs not published"
        )
    _validate_counts(total, len(dates) * 24, sample_scope)
    _write_outputs(root, grid, total, names, thresholds, sample_scope)
    atomic_json(
        {
            **request,
            "complete": True,
            "grid_fingerprint": grid.fingerprint,
            "sample_scope": sample_scope,
            "non_consensus_support_scope": SUPPORT_SCOPE,
            "implementation_ids": {name: index for index, name in enumerate(names)},
            "implementation_pair_count": pair_count,
            "cell_count": grid.cell_count,
            "row_count": grid.cell_count * pair_count,
            "cells_with_joint_observations": int(np.count_nonzero(total[0])),
            "joint_observation_count": int(total[0].sum(dtype=np.uint64)),
            "non_consensus_observation_count": int(total[1].sum(dtype=np.uint64)),
            "pair_observation_count": sum(
                int(compared.sum(dtype=np.uint64))
                for compared, _, _ in _pair_counts(total, sample_scope)
            ),
            "direction_definition": "higher/lower means class(A) >/< class(B)",
            "date_workers": date_workers,
            "batch_size": batch_size,
            "dates_reused": reused,
            "dates_scanned": len(pending),
        },
        (root / "metadata.json"),
    )
    return root


def country_classification_label(metadata: dict[str, object]) -> str | None:
    """Return the named country workload regime for a completed request."""

    selection = metadata.get("country_selection")
    if not isinstance(selection, dict):
        return None
    thresholds = tuple(float(value) for value in metadata.get("thresholds_c", ()))
    country_iso_a3 = selection.get("country_iso_a3")
    if country_iso_a3 == "JPN" and thresholds == (21.0, 25.0, 28.0, 31.0):
        return "JPN MoE"
    if country_iso_a3 == "THA" and thresholds == (30.0, 32.0, 34.0):
        return "Thailand workload limits"
    if (
        country_iso_a3 == "USA"
        and "selection_bounds" in selection
        and thresholds == (25.0, 26.0, 28.0, 30.0)
    ):
        return "CONUS OSHA"
    country_name = selection.get("country_name")
    return country_name if isinstance(country_name, str) and country_name else None


def _progress_label(source: CellwiseSource, thresholds: tuple[float, ...]) -> str:
    label = country_classification_label(
        {**source_metadata(source), "thresholds_c": list(thresholds)}
    )
    if label is not None:
        return f"{label} WBGT classification"
    return "ISO 7243"


def _initialize_worker(
    source, names, thresholds, grid_path, batch_size, signature, work, sample_scope
):
    global _WORKER
    grid = load_grid(Path(grid_path))
    _WORKER = ClassificationState(
        source=source,
        names=names,
        thresholds=thresholds,
        source_ids=grid.table["source_cell_id"].combine_chunks(),
        cell_count=grid.cell_count,
        batch_size=batch_size,
        signature=signature,
        work=Path(work),
        sample_scope=sample_scope,
    )


def _run_day(day: date) -> None:
    counts = _scan_day(_WORKER, day)
    path = _WORKER.work / f"{day.isoformat()}.npz"
    with atomic_path(path) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            counts=counts,
            signature=_WORKER.signature,
            day=day.isoformat(),
        )


def _scan_day(state: ClassificationState, day: date) -> np.ndarray:
    names, cell_count, sample_scope = state.names, state.cell_count, state.sample_scope
    classes = load_daily_classes(
        state.source,
        day,
        names,
        state.thresholds,
        state.source_ids,
        batch_size=state.batch_size,
    ).classes
    validity = classes >= 0
    joint = validity.all(axis=0)
    size = cell_count * 24
    pairs = tuple(combinations(range(len(names)), 2))
    stride = 3 if sample_scope == "pairwise" else 2
    counts = np.empty((2 + stride * len(pairs), cell_count), dtype=np.uint8)
    counts[0] = joint.reshape(cell_count, 24).sum(axis=1, dtype=np.uint8)
    non_consensus = np.zeros(size, dtype=bool)
    for index, (a, b) in enumerate(pairs):
        supported = (
            joint
            if sample_scope == "all-implementations"
            else validity[a] & validity[b]
        )
        offset = 2 + stride * index
        if sample_scope == "pairwise":
            counts[offset] = supported.reshape(cell_count, 24).sum(
                axis=1, dtype=np.uint8
            )
            offset += 1
        higher = supported & (classes[a] > classes[b])
        lower = supported & (classes[a] < classes[b])
        counts[offset] = higher.reshape(cell_count, 24).sum(axis=1, dtype=np.uint8)
        counts[offset + 1] = lower.reshape(cell_count, 24).sum(axis=1, dtype=np.uint8)
        non_consensus |= joint & (higher | lower)
    counts[1] = non_consensus.reshape(cell_count, 24).sum(axis=1, dtype=np.uint8)
    return counts


def _pair_counts(counts: np.ndarray, sample_scope: SampleScope):
    """Decode the common prefix and scope-dependent pair-count stride."""
    stride = 3 if sample_scope == "pairwise" else 2
    for index in range(2, len(counts), stride):
        if sample_scope == "pairwise":
            yield counts[index], counts[index + 1], counts[index + 2]
        else:
            yield counts[0], counts[index], counts[index + 1]


def _validate_counts(
    counts: np.ndarray, maximum: int, sample_scope: SampleScope = "all-implementations"
) -> None:
    if np.any(counts[0] > maximum) or np.any(counts[1] > counts[0]):
        raise ValueError("Invalid joint/non-consensus counts")
    changed_sum = np.zeros(counts.shape[1], dtype=np.uint64)
    for compared, higher, lower in _pair_counts(counts, sample_scope):
        changed = higher.astype(np.uint64) + lower
        if (
            np.any(compared > maximum)
            or np.any(compared < counts[0])
            or np.any(changed > compared)
        ):
            raise ValueError("Pair disagreement exceeds valid pair support")
        if sample_scope == "all-implementations" and np.any(changed > counts[1]):
            raise ValueError("Pair disagreement exceeds non-consensus support")
        changed_sum += changed
    if np.any(counts[1] > changed_sum):
        raise ValueError("Non-consensus exceeds all pair disagreements")


def _read_checkpoint(path, signature, day, shape):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (
                archive["signature"].item() != signature
                or archive["day"].item() != day.isoformat()
            ):
                return None
            counts = archive["counts"]
            if counts.shape != shape or counts.dtype != np.uint8:
                return None
            return counts
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return None


def _reduce_day(work, day, signature, total, sample_scope, progress_label):
    checkpoint = _read_checkpoint(
        work / f"{day.isoformat()}.npz",
        signature,
        day,
        total.shape,
    )
    if checkpoint is None:
        raise ValueError(f"Invalid completed checkpoint for {day}")
    counts = checkpoint
    total += counts
    print(
        f"{progress_label} {day}: joint={int(counts[0].sum()):,}, "
        f"non-consensus={int(counts[1].sum()):,}",
        flush=True,
    )


def _write_outputs(
    root, grid, counts, names, thresholds, sample_scope="all-implementations"
):
    if np.any(counts > np.iinfo(np.uint32).max):
        raise ValueError("Per-cell counts overflow uint32")
    ids = np.arange(grid.cell_count, dtype=np.int32)
    schema = pa.schema(
        [
            ("cell_id", pa.int32()),
            ("implementation_a_id", pa.uint8()),
            ("implementation_b_id", pa.uint8()),
            ("n_compared", pa.uint32()),
            ("n_higher", pa.uint32()),
            ("n_lower", pa.uint32()),
        ]
    )
    path = root / "cell_pair_differences.parquet"
    with (
        atomic_path(path) as temporary,
        pq.ParquetWriter(temporary, schema, compression="zstd") as writer,
    ):
        for (a, b), (compared, higher, lower) in zip(
            combinations(range(len(names)), 2),
            _pair_counts(counts, sample_scope),
            strict=True,
        ):
            writer.write_table(
                pa.Table.from_arrays(
                    [
                        pa.array(ids),
                        pa.array(np.full(len(ids), a, dtype=np.uint8)),
                        pa.array(np.full(len(ids), b, dtype=np.uint8)),
                        pa.array(compared, type=pa.uint32()),
                        pa.array(higher, type=pa.uint32()),
                        pa.array(lower, type=pa.uint32()),
                    ],
                    schema=schema,
                )
            )
    percent = np.divide(
        100.0 * counts[1], counts[0], out=np.full(len(ids), np.nan), where=counts[0] > 0
    )
    atomic_parquet(
        pa.table(
            {
                "cell_id": ids,
                "n_joint": pa.array(counts[0], type=pa.uint32()),
                "n_non_consensus": pa.array(counts[1], type=pa.uint32()),
                "non_consensus_percent": percent,
            }
        ),
        (root / "cellwise_non_consensus.parquet"),
    )
    atomic_parquet(
        grid.table.set_column(0, "cell_id", pa.array(ids)),
        (root / "cell_lookup.parquet"),
    )
    atomic_csv(
        pa.Table.from_pandas(class_definitions(thresholds), preserve_index=False),
        (root / "classes.csv"),
    )


def load_hourly_metadata(root: Path) -> dict[str, object]:
    """Reject incomplete, incompatible, or modified products before reuse."""
    path = root / "metadata.json"
    if not path.is_file():
        raise ValueError(f"Missing completion metadata: {path}")
    metadata = json.loads(path.read_text())
    expected = {
        "schema_version": ISO7243_SCHEMA_VERSION,
        "analysis": ANALYSIS,
        "complete": True,
        "observation_basis": "hourly",
        "classification_quantity": "wbgt",
    }
    sample_scope = metadata.get("sample_scope")
    expected_support = (
        PAIRWISE_SUPPORT_SCOPE if sample_scope == "pairwise" else SUPPORT_SCOPE
    )
    if (
        sample_scope not in SAMPLE_SCOPES
        or metadata.get("support_scope") != expected_support
        or metadata.get("non_consensus_support_scope") != SUPPORT_SCOPE
    ):
        raise ValueError("Invalid hourly WBGT sample scope")
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Input is not a completed hourly WBGT-only ISO 7243 product")
    for name in (
        "cell_pair_differences.parquet",
        "cellwise_non_consensus.parquet",
        "cell_lookup.parquet",
        "classes.csv",
    ):
        artifact = root / name
        if not artifact.is_file():
            raise ValueError(f"Incomplete ISO 7243 artifact: {artifact}")
    return metadata
