"""Annual cellwise ISO 7243 non-consensus rates from raw HSB Parquet shards."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..classification import class_definitions, validate_thresholds
from ..comparison.sql import EXPECTED_IMPLEMENTATIONS
from ..results import RESULT_QUANTITIES, RESULT_QUANTITY_SPECS
from .grid import CellGrid, grid_from_source_cell_ids
from .pipeline import (
    _atomic_json,
    _atomic_parquet,
    _date_range,
    _git_commit,
    _hsb_run_paths,
    _input_files,
    _input_fingerprint,
)


NON_CONSENSUS_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path("data/cellwise_iso7243_non_consensus")
BATCH_SIZE = 250_000
_COLUMNS = (
    "cycle_ordinal",
    "row_id",
    "row_ordinal",
    *(column for spec in RESULT_QUANTITY_SPECS.values() for column in (spec.status_column, spec.value_column)),
)


@dataclass(frozen=True)
class NonConsensusOutputs:
    """Artifacts written by :func:`analyze_cellwise_iso7243_non_consensus`."""

    cellwise_non_consensus: Path
    cell_lookup: Path
    classes: Path
    metadata: Path


def analyze_cellwise_iso7243_non_consensus(
    hsb_run_paths: str | Path | Sequence[str | Path],
    *,
    start_date: str | date,
    end_date: str | date,
    thresholds_c: Sequence[float],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> NonConsensusOutputs:
    """Write annual per-cell non-unanimous ISO-class percentages.

    Every source Parquet file is decoded in aligned PyArrow batches. NumPy holds
    only the per-cell counters and current seven-implementation class batch.
    """

    thresholds = validate_thresholds(thresholds_c)
    hsb_runs = _hsb_run_paths(hsb_run_paths)
    dates = _date_range(start_date, end_date)
    root = Path(output_root).expanduser().resolve()
    outputs = NonConsensusOutputs(
        root / "cellwise_non_consensus.parquet",
        root / "cell_lookup.parquet",
        root / "classes.csv",
        root / "metadata.json",
    )
    if not force and any(path.exists() for path in outputs.__dict__.values()):
        raise FileExistsError(f"Non-consensus output already exists at {root}; pass force=True")
    files = {
        implementation: _input_files(hsb_runs, dates, implementation)
        for implementation in EXPECTED_IMPLEMENTATIONS
    }
    input_fingerprint = _input_fingerprint(
        tuple(sorted(path for paths in files.values() for path in paths))
    )
    files_by_date = _files_by_date(files, dates)
    # Result cell identities define a static ERA5 grid.  The first complete
    # day's reference shards therefore establish the output grid without a
    # redundant full-year pre-scan.
    grid = _build_grid(files_by_date[dates[0]][EXPECTED_IMPLEMENTATIONS[0]])
    if grid.cell_count > np.iinfo(np.int32).max:
        raise ValueError("Grid cell_id values exceed int32 output range")
    source_ids = grid.table["source_cell_id"].combine_chunks()
    joint_counts = np.zeros(grid.cell_count, dtype=np.uint32)
    non_consensus_counts = np.zeros(grid.cell_count, dtype=np.uint32)
    for current_date in dates:
        counts, non_consensus = _scan_day(
            files_by_date[current_date], current_date, source_ids, grid.cell_count, thresholds
        )
        joint_counts += counts
        non_consensus_counts += non_consensus
    _write_outputs(
        outputs,
        grid,
        joint_counts,
        non_consensus_counts,
        thresholds,
        dates,
        hsb_runs,
        input_fingerprint,
    )
    return outputs


def _build_grid(reference_files: tuple[Path, ...]) -> CellGrid:
    source_ids: set[str] = set()
    for path in reference_files:
        for batch in pq.ParquetFile(path).iter_batches(columns=["row_id"], batch_size=BATCH_SIZE):
            source_ids.update(pc.utf8_slice_codeunits(batch.column(0), 0, 28).to_pylist())
    return grid_from_source_cell_ids(sorted(source_ids))


def _files_by_date(
    files: dict[str, tuple[Path, ...]], dates: tuple[date, ...]
) -> dict[date, dict[str, tuple[Path, ...]]]:
    """Group already-validated input paths by date and aligned shard scope."""

    result: dict[date, dict[str, tuple[Path, ...]]] = {}
    for current_date in dates:
        date_token = current_date.isoformat()
        grouped = {
            implementation: tuple(
                path
                for path in paths
                if path.parents[2].name == date_token
            )
            for implementation, paths in files.items()
        }
        reference_scopes = tuple(path.parents[1].name for path in grouped[EXPECTED_IMPLEMENTATIONS[0]])
        if not reference_scopes or len(set(reference_scopes)) != len(reference_scopes):
            raise ValueError(f"Invalid implementation shard scopes on {current_date}")
        for implementation, paths in grouped.items():
            scopes = tuple(path.parents[1].name for path in paths)
            if scopes != reference_scopes:
                raise ValueError(
                    f"Implementation shard scopes differ for {implementation} on {current_date}"
                )
        result[current_date] = grouped
    return result


def _scan_day(
    files: dict[str, tuple[Path, ...]],
    current_date: date,
    source_ids: pa.Array,
    cell_count: int,
    thresholds: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    seen = np.zeros(cell_count * 24, dtype=bool)
    joint_counts = np.zeros(cell_count, dtype=np.uint32)
    non_consensus_counts = np.zeros(cell_count, dtype=np.uint32)
    shard_count = len(files[EXPECTED_IMPLEMENTATIONS[0]])
    for shard_index in range(shard_count):
        streams = {
            implementation: pq.ParquetFile(paths[shard_index]).iter_batches(
                columns=_COLUMNS, batch_size=BATCH_SIZE
            )
            for implementation, paths in files.items()
        }
        sentinel = object()
        for batches in zip_longest(*streams.values(), fillvalue=sentinel):
            if any(batch is sentinel for batch in batches):
                raise ValueError(f"Implementation row counts differ on {current_date}")
            aligned = dict(zip(EXPECTED_IMPLEMENTATIONS, batches, strict=True))
            reference = aligned[EXPECTED_IMPLEMENTATIONS[0]]
            for implementation, batch in aligned.items():
                if batch.num_rows != reference.num_rows:
                    raise ValueError(f"Implementation batch sizes differ on {current_date}")
                for column in ("cycle_ordinal", "row_id", "row_ordinal"):
                    if not batch.column(column).equals(reference.column(column)):
                        raise ValueError(
                            f"Implementation identity order differs for {implementation} on {current_date}"
                        )
            cells, valid_identity = _cell_ids(reference.column("row_id"), source_ids, current_date)
            if not valid_identity:
                raise ValueError(f"Malformed or unknown result identity on {current_date}")
            hours = _hours(reference.column("row_id"))
            if (hours > 23).any():
                raise ValueError(f"Invalid hour identity on {current_date}")
            flat = cells * 24 + hours
            if seen[flat].any():
                raise ValueError(f"Duplicate cell/hour identity on {current_date}")
            seen[flat] = True
            valid = np.ones(reference.num_rows, dtype=bool)
            classes = np.empty((len(EXPECTED_IMPLEMENTATIONS), reference.num_rows), dtype=np.uint8)
            for index, implementation in enumerate(EXPECTED_IMPLEMENTATIONS):
                batch = aligned[implementation]
                for quantity in RESULT_QUANTITIES:
                    spec = RESULT_QUANTITY_SPECS[quantity]
                    valid &= _valid_values(batch.column(spec.status_column), batch.column(spec.value_column))
                values = _floats(batch.column("wbgt_c"))
                classes[index] = np.searchsorted(
                    thresholds, np.nan_to_num(values, nan=thresholds[0]), side="right"
                ).astype(np.uint8)
            non_consensus = np.any(classes != classes[0], axis=0)
            np.add.at(joint_counts, cells[valid], 1)
            np.add.at(non_consensus_counts, cells[valid & non_consensus], 1)
    return joint_counts, non_consensus_counts


def _cell_ids(row_ids: pa.Array, source_ids: pa.Array, current_date: date) -> tuple[np.ndarray, bool]:
    if not pc.all(pc.equal(pc.utf8_length(row_ids), 39)).as_py():
        return np.empty(0, dtype=np.int64), False
    date_token = current_date.strftime("%Y%m%d")
    if not pc.all(pc.equal(pc.utf8_slice_codeunits(row_ids, 29, 37), date_token)).as_py():
        return np.empty(0, dtype=np.int64), False
    indices = pc.index_in(pc.utf8_slice_codeunits(row_ids, 0, 28), value_set=source_ids)
    if indices.null_count:
        return np.empty(0, dtype=np.int64), False
    return _integers(indices), True


def _hours(row_ids: pa.Array) -> np.ndarray:
    return _integers(pc.cast(pc.utf8_slice_codeunits(row_ids, 37, 39), pa.uint8()))


def _valid_values(status: pa.Array, values: pa.Array) -> np.ndarray:
    valid = pc.fill_null(pc.and_(pc.equal(status, "ok"), pc.is_finite(values)), False)
    return np.asarray(valid.to_numpy(zero_copy_only=False), dtype=bool)


def _write_outputs(
    outputs: NonConsensusOutputs,
    grid: CellGrid,
    joint_counts: np.ndarray,
    non_consensus_counts: np.ndarray,
    thresholds: tuple[float, ...],
    dates: tuple[date, ...],
    hsb_runs: tuple[Path, ...],
    input_fingerprint: str,
) -> None:
    if (non_consensus_counts > joint_counts).any():
        raise ValueError("Non-consensus count exceeds joint-valid count")
    cell_ids = np.arange(grid.cell_count, dtype=np.int32)
    percent = np.divide(
        100.0 * non_consensus_counts,
        joint_counts,
        out=np.full(grid.cell_count, np.nan, dtype=np.float64),
        where=joint_counts > 0,
    )
    _atomic_parquet(
        pa.table(
            {
                "cell_id": pa.array(cell_ids, type=pa.int32()),
                "n_joint": pa.array(joint_counts, type=pa.uint32()),
                "n_non_consensus": pa.array(non_consensus_counts, type=pa.uint32()),
                "non_consensus_percent": pa.array(percent, type=pa.float64()),
            }
        ),
        outputs.cellwise_non_consensus,
    )
    _atomic_parquet(
        grid.table.set_column(0, "cell_id", pa.array(cell_ids, type=pa.int32())),
        outputs.cell_lookup,
    )
    _atomic_text(class_definitions(thresholds).to_csv(index=False), outputs.classes)
    _atomic_json(
        {
            "schema_version": NON_CONSENSUS_SCHEMA_VERSION,
            "analysis": "iso7243-cellwise-non-consensus-percent-v1",
            "start_date": dates[0].isoformat(),
            "end_date": dates[-1].isoformat(),
            "thresholds_c": list(thresholds),
            "implementations": list(EXPECTED_IMPLEMENTATIONS),
            "support_scope": "all-implementations-all-quantities",
            "non_consensus_definition": "non-unanimous-iso7243-wbgt-class-v1",
            "cell_count": grid.cell_count,
            "cells_with_joint_observations": int((joint_counts > 0).sum()),
            "joint_observation_count": int(joint_counts.sum(dtype=np.uint64)),
            "non_consensus_observation_count": int(
                non_consensus_counts.sum(dtype=np.uint64)
            ),
            "grid_fingerprint": grid.fingerprint,
            "input_runs": [str(path) for path in hsb_runs],
            "input_fingerprint": input_fingerprint,
            "git_commit": _git_commit(),
            "execution_engine": "pyarrow-parquet-batches-and-numpy-v1",
        },
        outputs.metadata,
    )


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _integers(values: pa.Array) -> np.ndarray:
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64)


def _floats(values: pa.Array) -> np.ndarray:
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)
