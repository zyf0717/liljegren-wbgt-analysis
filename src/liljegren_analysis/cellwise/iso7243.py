"""Resumable hourly WBGT class differences and non-consensus on shared support."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import resource
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from time import monotonic, time_ns
from zipfile import BadZipFile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..classification import validate_thresholds
from ..comparison.config import SAMPLE_SCOPES, SampleScope
from ..io.artifacts import atomic_json, atomic_parquet, git_commit
from ._pipeline import date_range
from .grid import CellGrid, load_grid
from ._joint_io import identity_batch
from .joint_pipeline import DEFAULT_JOINT_IMPLEMENTATIONS
from .paths import DEFAULT_ISO7243_OUTPUT_ROOT
from ._source import (
    CellwiseSource,
    build_source_grid,
    current_source_fingerprint,
    iter_source_batches,
    resolve_source,
    source_metadata,
)

ISO7243_SCHEMA_VERSION = 2
ANALYSIS = "iso7243-hourly-wbgt-classification-v2"
SUPPORT_SCOPE = "all-implementations-wbgt-only"
PAIRWISE_SUPPORT_SCOPE = "pairwise-wbgt-only"
DEFAULT_OUTPUT_ROOT = DEFAULT_ISO7243_OUTPUT_ROOT
DEFAULT_DATE_WORKERS = 4
DEFAULT_BATCH_SIZE = 250_000
_COLUMNS = ("cycle_ordinal", "row_id", "row_ordinal", "wbgt_status", "wbgt_c")
_WORKER = {}


@dataclass(frozen=True)
class ISO7243ClassificationOutputs:
    """Directional pair summaries and all-implementation unanimity counts."""

    cell_pair_differences: Path
    cellwise_non_consensus: Path
    cell_lookup: Path
    classes: Path
    metadata: Path


def _outputs(root: Path) -> ISO7243ClassificationOutputs:
    return ISO7243ClassificationOutputs(
        root / "cell_pair_differences.parquet",
        root / "cellwise_non_consensus.parquet",
        root / "cell_lookup.parquet", root / "classes.csv", root / "metadata.json",
    )


def analyze_cellwise_iso7243_classification(
    hsb_run_paths: str | Path | Sequence[str | Path] | None = None,
    *,
    country_cache: str | Path | None = None,
    start_date: str | date,
    end_date: str | date,
    thresholds_c: Sequence[float],
    implementations: Sequence[str] = DEFAULT_JOINT_IMPLEMENTATIONS,
    sample_scope: SampleScope = "all-implementations",
    output_root: str | Path | None = None,
    date_workers: int = DEFAULT_DATE_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> ISO7243ClassificationOutputs:
    """Classify raw hourly WBGT once and persist annual directional counts.

    Compatible checkpoints and completed outputs are reused automatically.
    Force rebuilds final artifacts; incompatible configurations get separate
    checkpoint generations. Source observations are never persisted here.
    Pairwise scope requires valid WBGT only in A and B for their comparison;
    non-consensus always requires all selected implementations to be valid.
    """
    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    thresholds = validate_thresholds(thresholds_c)
    names = tuple(implementations)
    if not 2 <= len(names) <= 256 or len(set(names)) != len(names):
        raise ValueError("Select 2–256 unique implementations")
    if any(not isinstance(name, str) or not name or Path(name).name != name
           or name in (".", "..") for name in names):
        raise ValueError("Invalid implementation name")
    if date_workers < 1 or batch_size < 1:
        raise ValueError("date_workers and batch_size must be positive")
    dates = date_range(start_date, end_date)
    if len(dates) * 24 > np.iinfo(np.uint32).max:
        raise ValueError("Date range exceeds uint32 per-cell counter capacity")
    if country_cache is not None and output_root is None:
        raise ValueError("country_cache requires an explicit output_root")
    root = Path(output_root or DEFAULT_OUTPUT_ROOT).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another ISO 7243 writer holds {root}") from error
        return _analyze_locked(
            root, hsb_run_paths, country_cache, dates, names, thresholds,
            date_workers, batch_size, force, sample_scope,
        )


def _analyze_locked(root, hsb_run_paths, country_cache, dates, names,
                    thresholds, date_workers, batch_size, force, sample_scope):
    started = monotonic()
    outputs = _outputs(root)
    source = resolve_source(
        hsb_run_paths, country_cache, dates, names,
        index_path=root / ".source-index.json",
    )
    fingerprint = source.fingerprint
    request = {
        "schema_version": ISO7243_SCHEMA_VERSION, "analysis": ANALYSIS,
        "observation_basis": "hourly", "classification_quantity": "wbgt",
        "support_scope": PAIRWISE_SUPPORT_SCOPE if sample_scope == "pairwise" else SUPPORT_SCOPE,
        "start_date": dates[0].isoformat(), "end_date": dates[-1].isoformat(),
        "thresholds_c": list(thresholds), "implementations": list(names),
        "input_fingerprint": fingerprint,
        **source_metadata(source),
    }
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    manifest_path = root / ".work" / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != request and not force:
            raise ValueError("Incompatible ISO 7243 resume configuration; use a new root or force")
    if outputs.metadata.exists():
        previous = json.loads(outputs.metadata.read_text())
        if any(previous.get(key) != value for key, value in request.items()) and not force:
            raise ValueError("Existing ISO 7243 product is incompatible")
        if not force:
            load_hourly_metadata(root)
            return outputs
        archive = root / ".work" / f"previous-metadata-{time_ns()}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(outputs.metadata, archive)
    atomic_json(request, manifest_path)
    work = root / ".work" / request_hash
    work.mkdir(parents=True, exist_ok=True)
    grid_path = work / "grid.parquet"
    if not grid_path.exists():
        atomic_parquet(build_source_grid(source, dates[0], names).table, grid_path)
    grid = load_grid(grid_path)
    if grid.cell_count > np.iinfo(np.int32).max:
        raise ValueError("Grid exceeds int32 cell-ID capacity")
    signature = hashlib.sha256(f"{request_hash}:{grid.fingerprint}".encode()).hexdigest()
    pair_count = len(names) * (len(names) - 1) // 2
    stride = 3 if sample_scope == "pairwise" else 2
    total = np.zeros((2 + stride * pair_count, grid.cell_count), dtype=np.uint64)
    progress_label = _progress_label(source)
    pending, provenance = [], {}
    reused = 0
    for day in dates:
        checkpoint = _read_checkpoint(
            work / f"{day.isoformat()}.npz", signature, day, total.shape, sample_scope,
        )
        if checkpoint is None:
            pending.append(day)
        else:
            counts, info = checkpoint
            total += counts
            provenance[day.isoformat()] = info
            reused += 1
    print(
        f"{progress_label}: {len(dates)} dates, {reused} cached, "
        f"{len(pending)} to scan",
        flush=True,
    )
    state = (source, names, thresholds, str(grid_path), batch_size, signature,
             str(work), sample_scope)
    if date_workers == 1:
        _initialize_worker(*state)
        for day in pending:
            _run_day(day)
            _reduce_day(
                work, day, signature, total, provenance, sample_scope, progress_label
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(date_workers, len(pending)), mp_context=get_context("spawn"),
            initializer=_initialize_worker, initargs=state,
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
                        provenance,
                        sample_scope,
                        progress_label,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    if current_source_fingerprint(source) != fingerprint:
        raise ValueError("Raw inputs changed during classification; outputs not published")
    _validate_counts(total, len(dates) * 24, sample_scope)
    _write_outputs(outputs, grid, total, names, thresholds, sample_scope)
    artifacts = {path.name: _sha256(path) for path in (
        outputs.cell_pair_differences, outputs.cellwise_non_consensus,
        outputs.cell_lookup, outputs.classes,
    )}
    atomic_json({
        **request, "complete": True, "grid_fingerprint": grid.fingerprint,
        "sample_scope": sample_scope, "non_consensus_support_scope": SUPPORT_SCOPE,
        "implementation_ids": {name: index for index, name in enumerate(names)},
        "implementation_pair_count": pair_count, "cell_count": grid.cell_count,
        "row_count": grid.cell_count * pair_count,
        "cells_with_joint_observations": int(np.count_nonzero(total[0])),
        "joint_observation_count": int(total[0].sum(dtype=np.uint64)),
        "non_consensus_observation_count": int(total[1].sum(dtype=np.uint64)),
        "pair_observation_count": sum(
            int(compared.sum(dtype=np.uint64))
            for compared, _, _ in _pair_counts(total, sample_scope)
        ),
        "direction_definition": "higher/lower means class(A) >/< class(B)",
        "artifact_sha256": artifacts, "git_commit": git_commit(),
        "execution_engine": "daily-wbgt-class-arrays-v2", "date_workers": date_workers,
        "batch_size": batch_size, "arrow_use_threads": False,
        "elapsed_seconds_this_invocation": monotonic() - started,
        "dates_reused": reused, "dates_scanned": len(pending),
        "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "daily_provenance": provenance,
    }, outputs.metadata)
    return outputs


def _progress_label(source: CellwiseSource) -> str:
    metadata = source_metadata(source)
    selection = metadata.get("country_selection")
    if isinstance(selection, dict):
        country_name = selection.get("country_name")
        if isinstance(country_name, str) and country_name:
            return f"{country_name} WBGT classification"
    return "ISO 7243"


def _initialize_worker(source, names, thresholds, grid_path, batch_size, signature,
                       work, sample_scope):
    global _WORKER
    grid = load_grid(Path(grid_path))
    _WORKER = dict(source=source, names=names, thresholds=thresholds,
                   source_ids=grid.table["source_cell_id"].combine_chunks(),
                   cell_count=grid.cell_count, batch_size=batch_size,
                   signature=signature, work=Path(work), sample_scope=sample_scope)


def _run_day(day: date) -> None:
    started = monotonic()
    counts, rows = _scan_day(day, **{
        key: _WORKER[key] for key in (
            "source", "names", "thresholds", "source_ids", "cell_count",
            "batch_size", "sample_scope",
        )
    })
    info = {"elapsed_seconds": monotonic() - started, "rows_decoded": rows,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    path = _WORKER["work"] / f"{day.isoformat()}.npz"
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, counts=counts, signature=_WORKER["signature"],
                                day=day.isoformat(), info=json.dumps(info))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _scan_day(day, *, source: CellwiseSource, names, thresholds, source_ids,
              cell_count, batch_size,
              sample_scope: SampleScope = "all-implementations"):
    size = cell_count * 24
    class_dtype = np.min_scalar_type(len(thresholds))
    classes = np.zeros((len(names), size), dtype=class_dtype)
    validity = np.zeros((len(names), size), dtype=bool) if sample_scope == "pairwise" else None
    joint = np.ones(size, dtype=bool)
    reference_seen = np.zeros(size, dtype=bool)
    reference = np.zeros((3, size), dtype=np.uint32)
    scopes: dict[str, int] = {}
    rows = 0
    for index, name in enumerate(names):
        seen = np.zeros(size, dtype=bool)
        for item in iter_source_batches(
            source, day, name, _COLUMNS, batch_size=batch_size
        ):
            batch = item.batch
            scope = scopes.setdefault(item.run_id, len(scopes) + 1)
            cells, hours, cycles, ordinals = identity_batch(
                batch, source_cell_ids=source_ids, current_date=day,
            )
            flat = cells * 24 + hours
            if len(np.unique(flat)) != len(flat) or seen[flat].any():
                raise ValueError(f"Duplicate cell/hour identity for {name} on {day}")
            seen[flat] = True
            known = reference_seen[flat]
            if (np.any(reference[0, flat[known]] != scope)
                or np.any(reference[1, flat[known]] != cycles[known])
                or np.any(reference[2, flat[known]] != ordinals[known])):
                raise ValueError(f"Conflicting result provenance for {name} on {day}")
            fresh = flat[~known]
            reference[0, fresh] = scope
            reference[1, fresh] = cycles[~known]
            reference[2, fresh] = ordinals[~known]
            reference_seen[flat] = True
            values = batch.column("wbgt_c")
            valid = pc.fill_null(pc.and_(
                pc.equal(batch.column("wbgt_status"), "ok"), pc.is_finite(values),
            ), False).to_numpy(zero_copy_only=False)
            joint[flat] &= valid
            if validity is not None:
                validity[index, flat] = valid
            classes[index, flat[valid]] = np.searchsorted(
                thresholds, values.to_numpy(zero_copy_only=False)[valid], side="right",
            ).astype(class_dtype)
            rows += batch.num_rows
        joint &= seen
    pairs = tuple(combinations(range(len(names)), 2))
    stride = 3 if sample_scope == "pairwise" else 2
    counts = np.empty((2 + stride * len(pairs), cell_count), dtype=np.uint8)
    counts[0] = joint.reshape(cell_count, 24).sum(axis=1, dtype=np.uint8)
    non_consensus = np.zeros(size, dtype=bool)
    for index, (a, b) in enumerate(pairs):
        supported = joint if validity is None else validity[a] & validity[b]
        offset = 2 + stride * index
        if validity is not None:
            counts[offset] = supported.reshape(cell_count, 24).sum(axis=1, dtype=np.uint8)
            offset += 1
        higher = supported & (classes[a] > classes[b])
        lower = supported & (classes[a] < classes[b])
        counts[offset] = higher.reshape(cell_count, 24).sum(axis=1, dtype=np.uint8)
        counts[offset + 1] = lower.reshape(cell_count, 24).sum(axis=1, dtype=np.uint8)
        non_consensus |= joint & (higher | lower)
    counts[1] = non_consensus.reshape(cell_count, 24).sum(axis=1, dtype=np.uint8)
    _validate_counts(counts, 24, sample_scope)
    return counts, rows


def _pair_counts(counts: np.ndarray, sample_scope: SampleScope):
    """Decode the common prefix and scope-dependent pair-count stride."""
    stride = 3 if sample_scope == "pairwise" else 2
    for index in range(2, len(counts), stride):
        if sample_scope == "pairwise":
            yield counts[index], counts[index + 1], counts[index + 2]
        else:
            yield counts[0], counts[index], counts[index + 1]


def _validate_counts(counts: np.ndarray, maximum: int,
                     sample_scope: SampleScope = "all-implementations") -> None:
    if np.any(counts[0] > maximum) or np.any(counts[1] > counts[0]):
        raise ValueError("Invalid joint/non-consensus counts")
    changed_sum = np.zeros(counts.shape[1], dtype=np.uint64)
    for compared, higher, lower in _pair_counts(counts, sample_scope):
        changed = higher.astype(np.uint64) + lower
        if (np.any(compared > maximum) or np.any(compared < counts[0])
            or np.any(changed > compared)):
            raise ValueError("Pair disagreement exceeds valid pair support")
        if sample_scope == "all-implementations" and np.any(changed > counts[1]):
            raise ValueError("Pair disagreement exceeds non-consensus support")
        changed_sum += changed
    if np.any(counts[1] > changed_sum):
        raise ValueError("Non-consensus exceeds all pair disagreements")


def _read_checkpoint(path, signature, day, shape, sample_scope="all-implementations"):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive["signature"].item() != signature or archive["day"].item() != day.isoformat():
                return None
            counts = archive["counts"]
            if counts.shape != shape or counts.dtype != np.uint8:
                return None
            _validate_counts(counts, 24, sample_scope)
            return counts, json.loads(archive["info"].item())
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return None


def _reduce_day(
    work, day, signature, total, provenance, sample_scope, progress_label
):
    checkpoint = _read_checkpoint(
        work / f"{day.isoformat()}.npz", signature, day, total.shape, sample_scope,
    )
    if checkpoint is None:
        raise ValueError(f"Invalid completed checkpoint for {day}")
    counts, info = checkpoint
    total += counts
    provenance[day.isoformat()] = info
    print(f"{progress_label} {day}: joint={int(counts[0].sum()):,}, "
          f"non-consensus={int(counts[1].sum()):,}, "
          f"scan={info['elapsed_seconds']:.1f}s, RSS={info['peak_rss_mib']:.0f} MiB", flush=True)


def _write_outputs(outputs, grid, counts, names, thresholds, sample_scope="all-implementations"):
    if np.any(counts > np.iinfo(np.uint32).max):
        raise ValueError("Per-cell counts overflow uint32")
    ids = np.arange(grid.cell_count, dtype=np.int32)
    schema = pa.schema([
        ("cell_id", pa.int32()), ("implementation_a_id", pa.uint8()),
        ("implementation_b_id", pa.uint8()), ("n_compared", pa.uint32()),
        ("n_higher", pa.uint32()), ("n_lower", pa.uint32()),
    ])
    path = outputs.cell_pair_differences
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for (a, b), (compared, higher, lower) in zip(
                combinations(range(len(names)), 2), _pair_counts(counts, sample_scope), strict=True,
            ):
                writer.write_table(pa.Table.from_arrays([
                    pa.array(ids), pa.array(np.full(len(ids), a, dtype=np.uint8)),
                    pa.array(np.full(len(ids), b, dtype=np.uint8)),
                    pa.array(compared, type=pa.uint32()),
                    pa.array(higher, type=pa.uint32()),
                    pa.array(lower, type=pa.uint32()),
                ], schema=schema))
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    percent = np.divide(100.0 * counts[1], counts[0],
                        out=np.full(len(ids), np.nan), where=counts[0] > 0)
    atomic_parquet(pa.table({
        "cell_id": ids, "n_joint": pa.array(counts[0], type=pa.uint32()),
        "n_non_consensus": pa.array(counts[1], type=pa.uint32()),
        "non_consensus_percent": percent,
    }), outputs.cellwise_non_consensus)
    atomic_parquet(grid.table.set_column(0, "cell_id", pa.array(ids)), outputs.cell_lookup)
    bounds = (-np.inf, *thresholds, np.inf)
    lines = ["class_id,class_label,threshold_definition"]
    for index, (lower, upper) in enumerate(zip(bounds, bounds[1:])):
        if index == 0:
            label, definition = f"< {upper:g} C", f"wbgt_c < {upper:g}"
        elif index == len(thresholds):
            label, definition = f">= {lower:g} C", f"wbgt_c >= {lower:g}"
        else:
            label, definition = f"[{lower:g}, {upper:g}) C", f"{lower:g} <= wbgt_c < {upper:g}"
        lines.append(f'{index},"{label}","{definition}"')
    descriptor, temporary = tempfile.mkstemp(dir=outputs.classes.parent, prefix=".classes.")
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, outputs.classes)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hourly_metadata(root: Path) -> dict[str, object]:
    """Reject incomplete, incompatible, or modified products before reuse."""
    path = root / "metadata.json"
    if not path.is_file():
        raise ValueError(f"Missing completion metadata: {path}")
    metadata = json.loads(path.read_text())
    expected = dict(schema_version=ISO7243_SCHEMA_VERSION, analysis=ANALYSIS,
                    complete=True, observation_basis="hourly",
                    classification_quantity="wbgt")
    sample_scope = metadata.get("sample_scope")
    expected_support = PAIRWISE_SUPPORT_SCOPE if sample_scope == "pairwise" else SUPPORT_SCOPE
    if (sample_scope not in SAMPLE_SCOPES or metadata.get("support_scope") != expected_support
        or metadata.get("non_consensus_support_scope") != SUPPORT_SCOPE):
        raise ValueError("Invalid hourly WBGT sample scope")
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Input is not a completed hourly WBGT-only ISO 7243 product")
    artifacts = metadata.get("artifact_sha256", {})
    for name in ("cell_pair_differences.parquet", "cellwise_non_consensus.parquet",
                 "cell_lookup.parquet", "classes.csv"):
        artifact = root / name
        if not artifact.is_file() or artifacts.get(name) != _sha256(artifact):
            raise ValueError(f"Incomplete or modified ISO 7243 artifact: {artifact}")
    return metadata
