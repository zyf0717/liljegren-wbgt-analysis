"""Annual ISO 7243 classification differences from cellwise WBGT quantiles."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..classification import class_definitions, validate_thresholds
from ..io.artifacts import atomic_json as _atomic_json
from ..io.artifacts import atomic_parquet as _atomic_parquet
from .paths import DEFAULT_ISO7243_INPUT_ROOT, DEFAULT_ISO7243_OUTPUT_ROOT

ISO7243_SCHEMA_VERSION = 1
PERCENTILES = ("p50", "p90", "p95", "p99")
DEFAULT_INPUT_ROOT = DEFAULT_ISO7243_INPUT_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_ISO7243_OUTPUT_ROOT


@dataclass(frozen=True)
class ISO7243ClassificationOutputs:
    """Artifacts written by :func:`analyze_cellwise_iso7243_classification`."""

    cell_pair_differences: Path
    cell_lookup: Path
    classes: Path
    metadata: Path


def analyze_cellwise_iso7243_classification(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    *,
    thresholds_c: Sequence[float],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> ISO7243ClassificationOutputs:
    """Write unweighted directional ISO 7243 differences for all cell pairs.

    Input implementations are read sequentially and reduced to four ``uint8``
    class-ID arrays. Each implementation pair is then emitted as one Parquet
    row group, avoiding a materialized cross join.
    """

    source_root = Path(input_root).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    paths = ISO7243ClassificationOutputs(
        root / "cell_pair_differences.parquet",
        root / "cell_lookup.parquet",
        root / "classes.csv",
        root / "metadata.json",
    )
    if not force and any(
        path.exists()
        for path in (
            paths.cell_pair_differences,
            paths.cell_lookup,
            paths.classes,
            paths.metadata,
        )
    ):
        raise FileExistsError(
            f"ISO 7243 output already exists at {root}; pass force=True"
        )
    thresholds = validate_thresholds(thresholds_c)
    metadata = _load_input_metadata(source_root)
    implementations = tuple(metadata["implementations"])
    grid = pq.read_table(source_root / "grid.parquet")
    cell_ids = _integer_column(grid, "cell_id")
    if not len(cell_ids):
        raise ValueError("Input grid is empty")
    if not np.array_equal(cell_ids, np.arange(len(cell_ids), dtype=np.int64)):
        raise ValueError("Input grid cell_id values are not contiguous")
    if cell_ids[-1] > np.iinfo(np.int32).max:
        raise ValueError("Input grid cell_id values exceed int32 output range")

    class_ids, complete = _load_class_ids(
        source_root, cell_ids, thresholds, metadata, implementations
    )
    if not complete.any():
        raise ValueError("No cells have complete valid WBGT quantiles")
    retained_ids = cell_ids[complete].astype(np.int32, copy=False)
    class_ids = class_ids[:, :, complete]
    _write_pair_differences(
        class_ids, retained_ids, implementations, paths.cell_pair_differences
    )
    lookup = grid.select(["cell_id", "source_cell_id", "latitude", "longitude"]).filter(
        pa.array(complete)
    )
    lookup = lookup.set_column(0, "cell_id", pa.array(retained_ids, type=pa.int32()))
    _atomic_parquet(lookup, paths.cell_lookup)
    _atomic_csv(_classes_csv(thresholds), paths.classes)
    _atomic_json(
        {
            "schema_version": ISO7243_SCHEMA_VERSION,
            "input_root": str(source_root),
            "input_fingerprints": {
                "metadata_json_sha256": _sha256(source_root / "metadata.json"),
                "grid_parquet_sha256": _sha256(source_root / "grid.parquet"),
                "implementation_parquet_sha256": {
                    implementation: _sha256(
                        source_root / "implementations" / f"{implementation}.parquet"
                    )
                    for implementation in implementations
                },
            },
            "thresholds_c": list(thresholds),
            "quantiles": list(PERCENTILES),
            "implementation_ids": {
                implementation: index
                for index, implementation in enumerate(implementations)
            },
            "canonical_cell_count": len(cell_ids),
            "complete_case_cell_count": len(retained_ids),
            "excluded_cell_count": int((~complete).sum()),
            "implementation_pair_count": len(implementations)
            * (len(implementations) - 1)
            // 2,
            "row_count": len(retained_ids)
            * len(implementations)
            * (len(implementations) - 1)
            // 2,
            "weighting": "none",
        },
        paths.metadata,
    )
    return paths


def _load_input_metadata(root: Path) -> dict[str, object]:
    path = root / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cellwise metadata: {path}")
    metadata = json.loads(path.read_text())
    if metadata.get("schema_version") != 2:
        raise ValueError("ISO 7243 analysis requires a joint cellwise quantile product")
    implementations = metadata.get("implementations")
    if (
        not isinstance(implementations, list)
        or len(implementations) < 2
        or not all(isinstance(value, str) and value for value in implementations)
        or len(set(implementations)) != len(implementations)
    ):
        raise ValueError("Input metadata does not contain valid implementations")
    if metadata.get("probabilities") != [0.5, 0.9, 0.95, 0.99]:
        raise ValueError("Input metadata does not contain the required quantiles")
    for key in ("temperature_bin_min_c", "temperature_bin_max_c"):
        if key not in metadata:
            raise ValueError(f"Input metadata is missing {key}")
    return metadata


def _load_class_ids(
    root: Path,
    cell_ids: np.ndarray,
    thresholds: tuple[float, ...],
    metadata: dict[str, object],
    implementations: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(cell_ids)
    classes = np.empty((len(implementations), len(PERCENTILES), count), dtype=np.uint8)
    complete = np.ones(count, dtype=bool)
    minimum = float(metadata["temperature_bin_min_c"])
    maximum = float(metadata["temperature_bin_max_c"])
    columns = ["cell_id", "n_joint"]
    for percentile in PERCENTILES:
        columns.extend(
            (
                f"wbgt_{percentile}_c",
                f"wbgt_{percentile}_lower_c",
                f"wbgt_{percentile}_upper_c",
            )
        )
    for implementation_index, implementation in enumerate(implementations):
        table = pq.read_table(
            root / "implementations" / f"{implementation}.parquet", columns=columns
        )
        if not np.array_equal(_integer_column(table, "cell_id"), cell_ids):
            raise ValueError(f"Cell universe/order differs for {implementation}")
        valid = _integer_column(table, "n_joint") > 0
        for percentile_index, percentile in enumerate(PERCENTILES):
            estimate = _float_column(table, f"wbgt_{percentile}_c")
            lower = _float_column(table, f"wbgt_{percentile}_lower_c")
            upper = _float_column(table, f"wbgt_{percentile}_upper_c")
            valid &= (
                np.isfinite(estimate)
                & np.isfinite(lower)
                & np.isfinite(upper)
                & (estimate >= minimum)
                & (estimate < maximum)
                & (lower <= estimate)
                & (estimate < upper)
            )
            classes[implementation_index, percentile_index] = np.searchsorted(
                thresholds, np.nan_to_num(estimate, nan=minimum), side="right"
            ).astype(np.uint8)
        complete &= valid
        del table, estimate, lower, upper
    return classes, complete


def _write_pair_differences(
    classes: np.ndarray,
    cell_ids: np.ndarray,
    implementations: tuple[str, ...],
    path: Path,
) -> None:
    schema = pa.schema(
        [
            ("cell_id", pa.int32()),
            ("implementation_a_id", pa.uint8()),
            ("implementation_b_id", pa.uint8()),
            ("n_compared", pa.uint8()),
            ("n_higher", pa.uint8()),
            ("n_lower", pa.uint8()),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with pq.ParquetWriter(temporary_path, schema, compression="zstd") as writer:
            for implementation_a in range(len(implementations)):
                classes_a = classes[implementation_a]
                for implementation_b in range(
                    implementation_a + 1, len(implementations)
                ):
                    classes_b = classes[implementation_b]
                    writer.write_table(
                        pa.table(
                            {
                                "cell_id": pa.array(cell_ids, type=pa.int32()),
                                "implementation_a_id": pa.array(
                                    np.full(
                                        len(cell_ids), implementation_a, dtype=np.uint8
                                    )
                                ),
                                "implementation_b_id": pa.array(
                                    np.full(
                                        len(cell_ids), implementation_b, dtype=np.uint8
                                    )
                                ),
                                "n_compared": pa.array(
                                    np.full(
                                        len(cell_ids), len(PERCENTILES), dtype=np.uint8
                                    )
                                ),
                                "n_higher": pa.array(
                                    (classes_a > classes_b).sum(axis=0).astype(np.uint8)
                                ),
                                "n_lower": pa.array(
                                    (classes_a < classes_b).sum(axis=0).astype(np.uint8)
                                ),
                            },
                            schema=schema,
                        )
                    )
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _integer_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64
    )


def _float_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    )


def _atomic_csv(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _classes_csv(thresholds: tuple[float, ...]) -> str:
    """Return the small, stable threshold sidecar without a dataframe dependency."""

    definitions = class_definitions(thresholds)
    lines = ["class_id,class_label,threshold_definition"]
    for class_id, label in zip(
        definitions["class_id"], definitions["class_label"], strict=True
    ):
        if class_id == 0:
            definition = f"wbgt_c < {thresholds[0]:g}"
        elif class_id == len(thresholds):
            definition = f"wbgt_c >= {thresholds[-1]:g}"
        else:
            definition = (
                f"{thresholds[class_id - 1]:g} <= wbgt_c < {thresholds[class_id]:g}"
            )
        lines.append(f'{class_id},"{label}","{definition}"')
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
