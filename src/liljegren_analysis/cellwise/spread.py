"""Reference-free spread analysis for persisted cellwise temperature quantiles."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..comparison.area_weighted import _ERA5_LATITUDE_WEIGHTS
from ..comparison.sql import EXPECTED_IMPLEMENTATIONS
from ..results import RESULT_QUANTITIES, RESULT_QUANTITY_SPECS, ResultQuantity
from .population import DEFAULT_OUTPUT_ROOT as DEFAULT_POPULATION_OUTPUT_ROOT

PERCENTILES = ("p50", "p90", "p95", "p99")
SPREAD_METRICS = (
    "spread_p50_c",
    "spread_p90_c",
    "spread_p95_c",
    "spread_p99_c",
)
ANALYSIS_SCHEMA_VERSION = 4
SPREAD_MAP_COLOR_LIMITS_C = (0.0, 8.0)
Weighting = Literal["none", "geometry", "population"]


@dataclass(frozen=True)
class QuantileSpreadOutputs:
    """Paths written by :func:`analyze_cellwise_quantile_spread`."""

    cellwise_spread: Path
    global_summary: Path
    c_anchor_differences: Path
    extrema_frequency: Path
    metadata: Path
    figure_directory: Path


def analyze_cellwise_quantile_spread(
    input_root: str | Path,
    *,
    output_root: str | Path | None = None,
    quantity: ResultQuantity = "wbgt",
    sample_size: int = 100,
    sample_seed: int = 0,
    weighting: Weighting = "geometry",
    population_weights_path: str | Path = DEFAULT_POPULATION_OUTPUT_ROOT / "population.parquet",
) -> QuantileSpreadOutputs:
    """Analyze one temperature's cross-implementation spread from Parquet only."""

    source_root = Path(input_root).expanduser().resolve()
    target_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else source_root.parent / "analysis"
    )
    if quantity not in RESULT_QUANTITIES:
        raise ValueError(f"Unsupported result quantity: {quantity}")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if weighting not in ("none", "geometry", "population"):
        raise ValueError(f"Unsupported weighting: {weighting}")
    source_metadata = _load_source_metadata(source_root)
    joint_schema = source_metadata.get("schema_version") == 2
    if not joint_schema and quantity != "wbgt":
        raise ValueError("Legacy cellwise products contain WBGT only")
    minimum_c, maximum_c, width_c = _histogram_bounds(source_metadata)
    grid = pq.read_table(source_root / "grid.parquet")
    cell_ids = _integer_column(grid, "cell_id")
    latitude = _float_column(grid, "latitude")
    longitude = _float_column(grid, "longitude")
    if not np.array_equal(cell_ids, np.arange(len(cell_ids), dtype=cell_ids.dtype)):
        raise ValueError("Canonical grid cell_id values are not contiguous")
    source_cell_ids = grid["source_cell_id"].combine_chunks().to_pylist()
    canonical_weights, weight_source = _weights(
        weighting, source_cell_ids, population_weights_path
    )
    source_tables, common = _load_quantile_tables(
        source_root,
        cell_ids,
        minimum_c=minimum_c,
        maximum_c=maximum_c,
        quantity=quantity,
        joint_schema=joint_schema,
    )
    if not common.any():
        raise ValueError("No complete-case cells across all implementations")
    if not joint_schema and not np.isclose(
        canonical_weights[common].sum(), canonical_weights.sum(), rtol=1e-12
    ):
        raise ValueError("Complete-case weight differs from the canonical retained grid")
    _spot_check_sources(
        source_root,
        source_tables,
        cell_ids,
        common,
        sample_size,
        sample_seed,
        quantity=quantity,
        joint_schema=joint_schema,
    )

    ids = cell_ids[common]
    latitudes = latitude[common]
    longitudes = longitude[common]
    weights = canonical_weights[common]
    if not weights.sum() > 0.0:
        raise ValueError("Complete-case cells have zero total weight")
    values = {
        implementation: {
            percentile: source_tables[implementation][percentile][common]
            for percentile in PERCENTILES
        }
        for implementation in EXPECTED_IMPLEMENTATIONS
    }
    cellwise, metric_values = _cellwise_table(ids, latitudes, longitudes, weights, values)
    summary = _global_summary(metric_values, weights)
    c_anchor = _c_anchor_summary(values, weights)
    extrema = _extrema_frequency(cellwise, weights)

    target_root.mkdir(parents=True, exist_ok=True)
    figure_directory = target_root / "figures"
    figure_directory.mkdir(exist_ok=True)
    _write_figures(summary, cellwise, figure_directory, quantity, weighting)
    cellwise_path = target_root / "cellwise_quantile_spread.parquet"
    summary_path = target_root / "global_quantile_spread_summary.parquet"
    anchor_path = target_root / "c_anchor_quantile_differences.parquet"
    extrema_path = target_root / "implementation_extrema_frequency.parquet"
    metadata_path = target_root / "metadata.json"
    _atomic_parquet(cellwise, cellwise_path)
    _atomic_parquet(summary, summary_path)
    _atomic_parquet(c_anchor, anchor_path)
    _atomic_parquet(extrema, extrema_path)
    _atomic_json(
        {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "input_root": str(source_root),
            "input_metadata_sha256": hashlib.sha256(
                (source_root / "metadata.json").read_bytes()
            ).hexdigest(),
            "input_histogram_version": source_metadata["histogram_version"],
            "temperature_bin_min_c": minimum_c,
            "temperature_bin_max_c": maximum_c,
            "temperature_bin_width_c": width_c,
            "quantity": quantity,
            "support_scope": source_metadata.get("support_scope", "implementation-native"),
            "percentiles": list(PERCENTILES),
            "implementations": list(EXPECTED_IMPLEMENTATIONS),
            "complete_case_cell_count": len(ids),
            "canonical_cell_count": len(cell_ids),
            "weighting": weighting,
            "weight_source": weight_source,
            "weight_sum": float(weights.sum()),
            "canonical_weight_sum": float(canonical_weights.sum()),
            "complete_case_weight_fraction": float(weights.sum() / canonical_weights.sum()),
            **(
                {
                    "area_weight_sum": float(weights.sum()),
                    "canonical_area_weight_sum": float(canonical_weights.sum()),
                    "complete_case_area_fraction": float(
                        weights.sum() / canonical_weights.sum()
                    ),
                }
                if weighting == "geometry"
                else {}
            ),
            "sample_size": min(sample_size, len(ids)),
            "sample_seed": sample_seed,
            "primary_map_metric": _primary_map_metric(summary),
            "spread_map_color_limits_c": list(SPREAD_MAP_COLOR_LIMITS_C),
            "git_commit": _git_commit(),
        },
        metadata_path,
    )
    return QuantileSpreadOutputs(
        cellwise_path, summary_path, anchor_path, extrema_path, metadata_path, figure_directory
    )


def _load_source_metadata(root: Path) -> dict[str, object]:
    path = root / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cellwise metadata: {path}")
    metadata = json.loads(path.read_text())
    expected = {
        "probabilities": [0.5, 0.9, 0.95, 0.99],
        "implementations": list(EXPECTED_IMPLEMENTATIONS),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Cellwise input metadata is not the required full implementation set")
    for key in ("histogram_version",):
        if key not in metadata:
            raise ValueError(f"Cellwise input metadata is missing {key}")
    _histogram_bounds(metadata)
    return metadata


def _histogram_bounds(metadata: dict[str, object]) -> tuple[float, float, float]:
    prefix = "temperature" if "temperature_bin_min_c" in metadata else "wbgt"
    keys = tuple(f"{prefix}_bin_{suffix}_c" for suffix in ("min", "max", "width"))
    if any(key not in metadata for key in keys):
        raise ValueError(f"Cellwise input metadata is missing histogram bounds: {keys}")
    minimum, maximum, width = (float(metadata[key]) for key in keys)
    if not minimum < maximum or width <= 0.0:
        raise ValueError("Cellwise input histogram bounds are invalid")
    return minimum, maximum, width


def _load_quantile_tables(
    root: Path,
    cell_ids: np.ndarray,
    *,
    minimum_c: float,
    maximum_c: float,
    quantity: ResultQuantity,
    joint_schema: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    tables: dict[str, dict[str, np.ndarray]] = {}
    common = np.ones(len(cell_ids), dtype=bool)
    prefix = f"{quantity}_" if joint_schema else ""
    count_column = "n_joint" if joint_schema else "n_valid"
    columns = [
        "cell_id",
        count_column,
        *(f"{prefix}{p}_c" for p in PERCENTILES),
        *(f"{prefix}{p}_lower_c" for p in PERCENTILES),
        *(f"{prefix}{p}_upper_c" for p in PERCENTILES),
    ]
    for implementation in EXPECTED_IMPLEMENTATIONS:
        path = root / "implementations" / f"{implementation}.parquet"
        table = pq.read_table(path, columns=columns)
        if not np.array_equal(_integer_column(table, "cell_id"), cell_ids):
            raise ValueError(f"Cell universe/order differs for {implementation}")
        values: dict[str, np.ndarray] = {
            "n_valid": _integer_column(table, count_column)
        }
        active = values["n_valid"] > 0
        for percentile in PERCENTILES:
            estimate = _float_column(table, f"{prefix}{percentile}_c")
            lower = _float_column(table, f"{prefix}{percentile}_lower_c")
            upper = _float_column(table, f"{prefix}{percentile}_upper_c")
            # Open-ended underflow/overflow quantiles have null midpoint or
            # bound values in the source product. Reject them rather than
            # silently excluding cells from the primary complete-case table.
            if not (
                np.isfinite(estimate[active]).all()
                and np.isfinite(lower[active]).all()
                and np.isfinite(upper[active]).all()
            ):
                raise ValueError(
                    f"{implementation} {percentile} is in an open histogram edge bin"
                )
            if not (
                (estimate[active] >= minimum_c).all()
                and (estimate[active] < maximum_c).all()
            ):
                raise ValueError(f"{implementation} {percentile} falls outside histogram bounds")
            if not (
                (lower[active] <= estimate[active]).all()
                and (estimate[active] < upper[active]).all()
            ):
                raise ValueError(f"{implementation} {percentile} has invalid containing bounds")
            values[percentile] = estimate
            common &= np.isfinite(estimate) & np.isfinite(lower) & np.isfinite(upper)
        common &= values["n_valid"] > 0
        tables[implementation] = values
    return tables, common


def _area_weights(source_cell_ids: list[str]) -> np.ndarray:
    weights = dict(
        zip(
            _ERA5_LATITUDE_WEIGHTS["latitude_code"].to_pylist(),
            _ERA5_LATITUDE_WEIGHTS["area_weight"].to_pylist(),
            strict=True,
        )
    )
    result = np.asarray([weights.get(value[12:19], np.nan) for value in source_cell_ids])
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Canonical grid has invalid ERA5 latitude area weights")
    return result


def _weights(
    weighting: Weighting,
    source_cell_ids: list[str],
    population_weights_path: str | Path,
) -> tuple[np.ndarray, str | None]:
    if weighting == "none":
        return np.ones(len(source_cell_ids), dtype=np.float64), None
    if weighting == "geometry":
        return _area_weights(source_cell_ids), "era5-cell-area-v1"
    path = Path(population_weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Population weights do not exist: {path}; run "
            "scripts/compute_era5_population_weights.py first"
        )
    table = pq.read_table(path, columns=["source_cell_id", "population"])
    population_ids = table["source_cell_id"].combine_chunks().to_pylist()
    population = _integer_column(table, "population").astype(np.float64)
    if len(population_ids) != len(set(population_ids)):
        raise ValueError("Population weights contain duplicate source_cell_id values")
    lookup = dict(zip(population_ids, population, strict=True))
    weights = np.asarray([lookup.get(cell_id, np.nan) for cell_id in source_cell_ids])
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("Population weights are missing or invalid for the analysis grid")
    return weights, str(path)


def _spot_check_sources(
    root: Path,
    tables: dict[str, dict[str, np.ndarray]],
    cell_ids: np.ndarray,
    common: np.ndarray,
    sample_size: int,
    seed: int,
    *,
    quantity: ResultQuantity,
    joint_schema: bool,
) -> None:
    prefix = f"{quantity}_" if joint_schema else ""
    selected = np.random.default_rng(seed).choice(np.flatnonzero(common), min(sample_size, int(common.sum())), replace=False)
    for implementation, table in tables.items():
        source = pq.read_table(
            root / "implementations" / f"{implementation}.parquet",
            columns=[
                "cell_id", *(f"{prefix}{percentile}_c" for percentile in PERCENTILES)
            ],
        )
        if not np.array_equal(_integer_column(source, "cell_id")[selected], cell_ids[selected]):
            raise ValueError(f"Source cell ID spot check failed for {implementation}")
        for percentile in PERCENTILES:
            if not np.array_equal(
                _float_column(source, f"{prefix}{percentile}_c")[selected],
                table[percentile][selected],
            ):
                raise ValueError(f"Source percentile spot check failed for {implementation} {percentile}")


def _cellwise_table(
    cell_ids: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, weights: np.ndarray,
    values: dict[str, dict[str, np.ndarray]],
) -> tuple[pa.Table, dict[str, np.ndarray]]:
    columns: dict[str, pa.Array] = {
        "cell_id": pa.array(cell_ids, type=pa.int64()), "latitude": pa.array(latitude),
        "longitude": pa.array(longitude), "weight": pa.array(weights),
    }
    metrics: dict[str, np.ndarray] = {}
    for implementation in EXPECTED_IMPLEMENTATIONS:
        prefix = implementation.replace("-", "_")
        for percentile in PERCENTILES:
            columns[f"q_{prefix}_{percentile}_c"] = pa.array(values[implementation][percentile])
    for percentile in PERCENTILES:
        matrix = np.column_stack([values[name][percentile] for name in EXPECTED_IMPLEMENTATIONS])
        minimum_index, maximum_index = matrix.argmin(axis=1), matrix.argmax(axis=1)
        metric = f"spread_{percentile}_c"
        spread = matrix[np.arange(len(matrix)), maximum_index] - matrix[np.arange(len(matrix)), minimum_index]
        if (spread < 0).any():
            raise ValueError(f"Negative {metric}")
        metrics[metric] = spread
        columns[metric] = pa.array(spread)
        columns[f"min_implementation_{percentile}"] = pa.array([EXPECTED_IMPLEMENTATIONS[index] for index in minimum_index])
        columns[f"max_implementation_{percentile}"] = pa.array([EXPECTED_IMPLEMENTATIONS[index] for index in maximum_index])
    return pa.table(columns), metrics


def _global_summary(metrics: dict[str, np.ndarray], weights: np.ndarray) -> pa.Table:
    rows: list[dict[str, object]] = []
    for metric, values in metrics.items():
        row: dict[str, object] = {"metric": metric, **_weighted_statistics(values, weights)}
        if metric.startswith("spread_"):
            for threshold in (0.1, 0.2, 0.5, 1.0):
                row[f"weight_fraction_gt_{str(threshold).replace('.', '_')}_c"] = float(weights[values > threshold].sum() / weights.sum())
        rows.append(row)
    return pa.Table.from_pylist(rows)


def _weighted_statistics(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "weighted_mean_c": float(np.average(values, weights=weights)),
        "weighted_median_c": _weighted_quantile(values, weights, 0.5),
        "weighted_p90_c": _weighted_quantile(values, weights, 0.9),
        "weighted_p95_c": _weighted_quantile(values, weights, 0.95),
        "weighted_p99_c": _weighted_quantile(values, weights, 0.99),
        "maximum_c": float(values.max()),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, probability * cumulative[-1], side="left")]])


def _c_anchor_summary(values: dict[str, dict[str, np.ndarray]], weights: np.ndarray) -> pa.Table:
    rows: list[dict[str, object]] = []
    for implementation in EXPECTED_IMPLEMENTATIONS:
        if implementation == "liljegren-c":
            continue
        for percentile in PERCENTILES:
            difference = values[implementation][percentile] - values["liljegren-c"][percentile]
            absolute = np.abs(difference)
            rows.append({
                "implementation": implementation, "anchor_implementation": "liljegren-c", "percentile": percentile,
                "weighted_bias_c": float(np.average(difference, weights=weights)),
                "weighted_mae_c": float(np.average(absolute, weights=weights)),
                "weighted_rmse_c": float(np.sqrt(np.average(np.square(difference), weights=weights))),
                "weighted_p95_absolute_difference_c": _weighted_quantile(absolute, weights, 0.95),
                "maximum_absolute_difference_c": float(absolute.max()),
            })
    return pa.Table.from_pylist(rows)


def _extrema_frequency(cellwise: pa.Table, weights: np.ndarray) -> pa.Table:
    rows: list[dict[str, object]] = []
    for percentile in PERCENTILES:
        minimum = cellwise[f"min_implementation_{percentile}"].to_pylist()
        maximum = cellwise[f"max_implementation_{percentile}"].to_pylist()
        for implementation in EXPECTED_IMPLEMENTATIONS:
            rows.append({"percentile": percentile, "implementation": implementation,
                         "weight_fraction_argmin": float(weights[np.asarray(minimum) == implementation].sum() / weights.sum()),
                         "weight_fraction_argmax": float(weights[np.asarray(maximum) == implementation].sum() / weights.sum())})
    return pa.Table.from_pylist(rows)


def _primary_map_metric(summary: pa.Table) -> str:
    metrics = summary["metric"].to_pylist()
    p95 = summary["weighted_p95_c"][metrics.index("spread_p95_c")].as_py()
    p99 = summary["weighted_p95_c"][metrics.index("spread_p99_c")].as_py()
    return "spread_p99_c" if p99 >= p95 else "spread_p95_c"


def _write_figures(
    summary: pa.Table,
    cellwise: pa.Table,
    directory: Path,
    quantity: ResultQuantity,
    weighting: Weighting,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = summary["metric"].to_pylist()
    percentiles = list(PERCENTILES)
    rows = [metric_names.index(f"spread_{p}_c") for p in percentiles]
    x = np.asarray([50, 90, 95, 99])
    figure, axis = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    weight_label = {
        "none": "Unweighted",
        "geometry": "Geometric-weighted",
        "population": "Population-weighted",
    }[weighting]
    for column, label in (
        ("weighted_median_c", f"{weight_label} median"),
        ("weighted_mean_c", f"{weight_label} mean"),
        ("weighted_p95_c", f"{weight_label} P95"),
    ):
        axis.plot(x, [summary[column][row].as_py() for row in rows], marker="o", label=label)
    label = RESULT_QUANTITY_SPECS[quantity].label
    axis.set(
        xlabel="Percentile",
        ylabel="Range of WBGT (°C) estimates",
        xticks=x,
    )
    axis.legend()
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"spread_by_percentile.{suffix}", dpi=200)
    plt.close(figure)
    for metric in SPREAD_METRICS:
        figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
        values = _float_column(cellwise, metric)
        image = axis.scatter(
            _float_column(cellwise, "longitude"),
            _float_column(cellwise, "latitude"),
            c=values,
            s=0.35,
            marker="s",
            linewidths=0,
            cmap="magma",
            vmin=SPREAD_MAP_COLOR_LIMITS_C[0],
            vmax=SPREAD_MAP_COLOR_LIMITS_C[1],
            rasterized=True,
        )
        axis.set(
            xlabel="Longitude",
            ylabel="Latitude",
            title=f"{metric.replace('_c', '').replace('_', ' ').upper()} (°C)",
            xlim=(-180, 180),
            ylim=(-90, 90),
        )
        figure.colorbar(image, ax=axis, label="Spread (°C)")
        for suffix in ("png", "pdf"):
            figure.savefig(directory / f"{metric.replace('_c', '')}_map.{suffix}", dpi=220)
        plt.close(figure)


def _integer_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)


def _float_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64)


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(value: dict[str, object], path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _git_commit() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None
