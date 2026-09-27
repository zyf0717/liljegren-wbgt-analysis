"""Reference-free spread calculations from persisted cellwise quantiles."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import (
    DEFAULT_PROBABILITIES,
    REFERENCE_IMPLEMENTATION,
)
from ..grid import area_weights
from ..io.artifacts import atomic_csv, atomic_json, atomic_parquet
from ..results import RESULT_QUANTITIES, ResultQuantity
from .population import load_population_weights

PERCENTILES = tuple(f"p{int(p * 100):02d}" for p in DEFAULT_PROBABILITIES)
SPREAD_METRICS = tuple(f"spread_{p}_c" for p in PERCENTILES)
Weighting = Literal["none", "geometry", "population"]


def analyze_cellwise_quantile_spread(
    input_root: str | Path,
    *,
    output_root: str | Path,
    quantity: ResultQuantity = "wbgt",
    weighting: Weighting = "geometry",
    population_weights_path: str | Path | None = None,
) -> Path:
    """Publish scientific spread tables; figures and coverage are separate steps."""
    tables, metadata = summarize_quantile_spread(
        input_root,
        quantity=quantity,
        weighting=weighting,
        population_weights_path=population_weights_path,
    )
    root = Path(output_root).expanduser().resolve()
    for name, table in tables.items():
        atomic_parquet(table, root / f"{name}.parquet")
    atomic_csv(
        _extrema_fractions(
            tables["implementation_extrema_frequency"], metadata["implementations"]
        ),
        root / "implementation_extrema_frequency.csv",
    )
    atomic_json(metadata, root / "metadata.json")
    return root


def summarize_quantile_spread(
    input_root: str | Path,
    *,
    quantity: ResultQuantity = "wbgt",
    weighting: Weighting = "geometry",
    population_weights_path: str | Path | None = None,
) -> tuple[dict[str, pa.Table], dict]:
    """Calculate named scientific tables without writing or rendering anything."""

    source_root = Path(input_root).expanduser().resolve()
    if quantity not in RESULT_QUANTITIES:
        raise ValueError(f"Unsupported result quantity: {quantity}")
    if weighting not in ("none", "geometry", "population"):
        raise ValueError(f"Unsupported weighting: {weighting}")
    source_metadata = _load_source_metadata(source_root)
    implementations = tuple(source_metadata["implementations"])
    if REFERENCE_IMPLEMENTATION not in implementations:
        raise ValueError(
            f"Reference implementation is absent: {REFERENCE_IMPLEMENTATION}"
        )
    grid = pq.read_table(source_root / "grid.parquet")
    cell_ids = np.asarray(grid["cell_id"], dtype=np.int64)
    latitude = np.asarray(grid["latitude"], dtype=np.float64)
    longitude = np.asarray(grid["longitude"], dtype=np.float64)
    source_cell_ids = grid["source_cell_id"].combine_chunks().to_pylist()
    cell_areas = area_weights(source_cell_ids)
    if weighting == "population":
        if population_weights_path is None:
            raise ValueError("Population weighting requires population_weights_path")
        canonical_weights, _ = load_population_weights(
            source_cell_ids, population_weights_path
        )
    elif weighting == "geometry":
        canonical_weights = cell_areas
    else:
        canonical_weights = np.ones(len(source_cell_ids))
    source_tables, common = _load_quantile_tables(
        source_root,
        cell_ids,
        quantity=quantity,
        implementations=implementations,
    )
    if not common.any():
        raise ValueError("No complete-case cells across all implementations")

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
        for implementation in implementations
    }
    cellwise, metric_values = _cellwise_table(
        ids, latitudes, longitudes, weights, values, implementations
    )
    summary = _global_summary(metric_values, weights)
    c_anchor = _c_anchor_summary(values, weights, implementations)
    extrema = _extrema_frequency(cellwise, weights, implementations)
    tables = {
        "cellwise_quantile_spread": cellwise,
        "global_quantile_spread_summary": summary,
        "c_anchor_quantile_differences": c_anchor,
        "implementation_extrema_frequency": extrema,
    }
    metadata = {
        "input_root": str(source_root),
        "quantity": quantity,
        "support_scope": source_metadata["support_scope"],
        "percentiles": list(PERCENTILES),
        "implementations": list(implementations),
        "weighting": weighting,
        "population_weights_path": str(Path(population_weights_path).resolve())
        if weighting == "population"
        else None,
        "complete_case_cell_count": len(ids),
        "canonical_cell_count": len(cell_ids),
        "weight_sum": float(weights.sum()),
        "canonical_weight_sum": float(canonical_weights.sum()),
        "complete_case_weight_fraction": float(weights.sum() / canonical_weights.sum()),
        "extrema_tie_policy": "first-in-implementation-order",
        **(
            {"country_selection": source_metadata["country_selection"]}
            if "country_selection" in source_metadata
            else {}
        ),
    }
    return tables, metadata


def _load_source_metadata(root: Path) -> dict:
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["schema_version"] != 2:
        raise ValueError("Unsupported cellwise input schema")
    if metadata["support_scope"] != "all-implementations-all-quantities":
        raise ValueError("Unsupported cellwise support scope")
    if metadata["probabilities"] != list(DEFAULT_PROBABILITIES):
        raise ValueError("Cellwise input metadata lacks the required quantiles")
    return metadata


def _load_quantile_tables(
    root: Path,
    cell_ids: np.ndarray,
    *,
    quantity: ResultQuantity,
    implementations: tuple[str, ...],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    tables: dict[str, dict[str, np.ndarray]] = {}
    common = np.ones(len(cell_ids), dtype=bool)
    prefix = f"{quantity}_"
    count_column = "n_joint"
    columns = [
        "cell_id",
        count_column,
        *(f"{prefix}{p}_c" for p in PERCENTILES),
        *(f"{prefix}{p}_lower_c" for p in PERCENTILES),
        *(f"{prefix}{p}_upper_c" for p in PERCENTILES),
    ]
    for implementation in implementations:
        path = root / "implementations" / f"{implementation}.parquet"
        table = pq.read_table(path, columns=columns)
        if not np.array_equal(np.asarray(table["cell_id"], dtype=np.int64), cell_ids):
            raise ValueError(f"Cell universe/order differs for {implementation}")
        values: dict[str, np.ndarray] = {
            "n_joint": np.asarray(table[count_column], dtype=np.int64)
        }
        active = values["n_joint"] > 0
        for percentile in PERCENTILES:
            estimate = np.asarray(table[f"{prefix}{percentile}_c"], dtype=np.float64)
            lower = np.asarray(table[f"{prefix}{percentile}_lower_c"], dtype=np.float64)
            upper = np.asarray(table[f"{prefix}{percentile}_upper_c"], dtype=np.float64)
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
            values[percentile] = estimate
            common &= np.isfinite(estimate) & np.isfinite(lower) & np.isfinite(upper)
        common &= values["n_joint"] > 0
        tables[implementation] = values
    return tables, common


def _cellwise_table(
    cell_ids: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    weights: np.ndarray,
    values: dict[str, dict[str, np.ndarray]],
    implementations: tuple[str, ...],
) -> tuple[pa.Table, dict[str, np.ndarray]]:
    columns: dict[str, pa.Array] = {
        "cell_id": pa.array(cell_ids, type=pa.int64()),
        "latitude": pa.array(latitude),
        "longitude": pa.array(longitude),
        "weight": pa.array(weights),
    }
    metrics: dict[str, np.ndarray] = {}
    for implementation in implementations:
        prefix = implementation.replace("-", "_")
        for percentile in PERCENTILES:
            columns[f"q_{prefix}_{percentile}_c"] = pa.array(
                values[implementation][percentile]
            )
    for percentile in PERCENTILES:
        matrix = np.column_stack([values[name][percentile] for name in implementations])
        minimum_index, maximum_index = matrix.argmin(axis=1), matrix.argmax(axis=1)
        metric = f"spread_{percentile}_c"
        spread = (
            matrix[np.arange(len(matrix)), maximum_index]
            - matrix[np.arange(len(matrix)), minimum_index]
        )
        metrics[metric] = spread
        columns[metric] = pa.array(spread)
        columns[f"min_implementation_{percentile}"] = pa.array(
            [implementations[index] for index in minimum_index]
        )
        columns[f"max_implementation_{percentile}"] = pa.array(
            [implementations[index] for index in maximum_index]
        )
    return pa.table(columns), metrics


def _global_summary(metrics: dict[str, np.ndarray], weights: np.ndarray) -> pa.Table:
    rows: list[dict[str, object]] = []
    for metric, values in metrics.items():
        row: dict[str, object] = {
            "metric": metric,
            **_weighted_statistics(values, weights),
        }
        if metric.startswith("spread_"):
            for threshold in (0.1, 0.2, 0.5, 1.0):
                row[f"weight_fraction_gt_{str(threshold).replace('.', '_')}_c"] = float(
                    weights[values > threshold].sum() / weights.sum()
                )
        rows.append(row)
    return pa.Table.from_pylist(rows)


def _weighted_statistics(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    quantiles = _weighted_quantiles(values, weights, (0.5, 0.9, 0.95, 0.99))
    return {
        "weighted_mean_c": float(np.average(values, weights=weights)),
        **{
            f"weighted_{label}_c": float(value)
            for label, value in zip(
                ("median", "p90", "p95", "p99"), quantiles, strict=True
            )
        },
        "maximum_c": float(values.max()),
    }


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probabilities
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    indices = np.searchsorted(
        cumulative, np.asarray(probabilities) * cumulative[-1], side="left"
    )
    return values[order[indices]]


def _c_anchor_summary(
    values: dict[str, dict[str, np.ndarray]],
    weights: np.ndarray,
    implementations: tuple[str, ...],
) -> pa.Table:
    rows: list[dict[str, object]] = []
    for implementation in implementations:
        if implementation == REFERENCE_IMPLEMENTATION:
            continue
        for percentile in PERCENTILES:
            difference = (
                values[implementation][percentile]
                - values[REFERENCE_IMPLEMENTATION][percentile]
            )
            absolute = np.abs(difference)
            rows.append(
                {
                    "implementation": implementation,
                    "anchor_implementation": REFERENCE_IMPLEMENTATION,
                    "percentile": percentile,
                    "weighted_bias_c": float(np.average(difference, weights=weights)),
                    "weighted_mae_c": float(np.average(absolute, weights=weights)),
                    "weighted_rmse_c": float(
                        np.sqrt(np.average(np.square(difference), weights=weights))
                    ),
                    "weighted_p95_absolute_difference_c": float(
                        _weighted_quantiles(absolute, weights, (0.95,))[0]
                    ),
                    "maximum_absolute_difference_c": float(absolute.max()),
                }
            )
    return pa.Table.from_pylist(rows)


def _extrema_frequency(
    cellwise: pa.Table, weights: np.ndarray, implementations: tuple[str, ...]
) -> pa.Table:
    rows: list[dict[str, object]] = []
    for percentile in PERCENTILES:
        minimum = cellwise[f"min_implementation_{percentile}"].to_pylist()
        maximum = cellwise[f"max_implementation_{percentile}"].to_pylist()
        for implementation in implementations:
            rows.append(
                {
                    "percentile": percentile,
                    "implementation": implementation,
                    "weight_fraction_argmin": float(
                        weights[np.asarray(minimum) == implementation].sum()
                        / weights.sum()
                    ),
                    "weight_fraction_argmax": float(
                        weights[np.asarray(maximum) == implementation].sum()
                        / weights.sum()
                    ),
                }
            )
    return pa.Table.from_pylist(rows)


def _extrema_fractions(extrema: pa.Table, implementations: tuple[str, ...]) -> pa.Table:
    """Pivot the existing extrema shares into fraction-valued percentile columns."""
    lookup = {
        (row["implementation"], row["percentile"]): row for row in extrema.to_pylist()
    }
    return pa.Table.from_pylist(
        [
            {
                "implementation": implementation,
                "extremum": extremum,
                **{
                    percentile: lookup[implementation, percentile][
                        f"weight_fraction_arg{extremum}"
                    ]
                    for percentile in PERCENTILES
                },
            }
            for implementation in implementations
            for extremum in ("min", "max")
        ]
    )
