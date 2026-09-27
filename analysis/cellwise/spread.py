"""Reference-free spread analysis for persisted cellwise temperature quantiles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import (
    DEFAULT_JOINT_OUTPUT_ROOT,
    DEFAULT_PROBABILITIES,
    DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT,
    REFERENCE_IMPLEMENTATION,
)
from ..grid import area_weights
from ..io.artifacts import atomic_csv, atomic_json, atomic_parquet, atomic_path
from ..results import RESULT_QUANTITIES, ResultQuantity
from ._spread_plots import (
    PERCENTILES,
    SPREAD_MAP_COLOR_LIMITS_C,
    SPREAD_METRICS,
    STACKED_QUANTITIES,
    write_figures,
    write_stacked_spread_maps,
)
from .population import DEFAULT_POPULATION_WEIGHTS_PATH, load_population_weights
from .projection import country_boundary_from_metadata

ANALYSIS_SCHEMA_VERSION = 9
TROPICAL_LATITUDE_LIMIT_DEGREES = 23.44
SPREAD_COVERAGE_THRESHOLDS_C = tuple(range(1, 9))
SPREAD_THRESHOLD_ATOL_C = 1e-12
Weighting = Literal["none", "geometry", "population"]
COVERAGE_WORKBOOK_ROWS = (
    ("affected_cell_count", "affected_cell_count", "integer"),
    ("affected_land_area_weight", "affected_land_area_weight", "weight"),
    ("land_area_fraction", "land_area_percent", "percentage"),
    ("affected_population", "affected_population", "integer"),
    ("population_fraction", "population_percent", "percentage"),
    ("total_land_area_weight", "total_land_area_weight", "weight"),
    ("total_population", "total_population", "integer"),
    (
        "complete_case_land_area_fraction",
        "complete_case_land_area_percent",
        "percentage",
    ),
    (
        "complete_case_population_fraction",
        "complete_case_population_percent",
        "percentage",
    ),
)


@dataclass(frozen=True)
class QuantileSpreadOutputs:
    """Paths written by :func:`analyze_cellwise_quantile_spread`."""

    cellwise_spread: Path
    global_summary: Path
    c_anchor_differences: Path
    extrema_frequency: Path
    metadata: Path
    figure_directory: Path
    extrema_frequency_csv: Path
    spread_coverage_xlsx: Path


def plot_stacked_cellwise_quantile_spread(
    output_root: str | Path = DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT,
) -> tuple[Path, ...]:
    """Plot separate P50/P95 WBGT, Tnwb, and Tg map stacks."""

    root = Path(output_root).expanduser().resolve()
    columns = ["latitude", "longitude", *SPREAD_METRICS]
    tables = {
        quantity: pq.read_table(
            root / quantity / "cellwise_quantile_spread.parquet",
            columns=columns,
        )
        for quantity, _ in STACKED_QUANTITIES
    }
    return write_stacked_spread_maps(tables, root / "figures")


def analyze_cellwise_quantile_spread(
    input_root: str | Path = DEFAULT_JOINT_OUTPUT_ROOT,
    *,
    output_root: str | Path | None = None,
    quantity: ResultQuantity = "wbgt",
    weighting: Weighting = "geometry",
    population_weights_path: str | Path = DEFAULT_POPULATION_WEIGHTS_PATH,
) -> QuantileSpreadOutputs:
    """Analyze spread and regional area/population coverage from Parquet only."""

    source_root = Path(input_root).expanduser().resolve()
    target_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT.resolve() / quantity
    )
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
    minimum_c, maximum_c, width_c = (
        source_metadata[f"temperature_bin_{key}_c"] for key in ("min", "max", "width")
    )
    grid = pq.read_table(source_root / "grid.parquet")
    cell_ids = np.asarray(grid["cell_id"], dtype=np.int64)
    latitude = np.asarray(grid["latitude"], dtype=np.float64)
    longitude = np.asarray(grid["longitude"], dtype=np.float64)
    source_cell_ids = grid["source_cell_id"].combine_chunks().to_pylist()
    cell_areas = area_weights(source_cell_ids)
    population_path = Path(population_weights_path).expanduser().resolve()
    population, _ = load_population_weights(source_cell_ids, population_path)
    if weighting == "geometry":
        canonical_weights = cell_areas
    elif weighting == "population":
        canonical_weights = population.astype(np.float64)
    else:
        canonical_weights = np.ones(len(source_cell_ids), dtype=np.float64)
    weight_source = {
        "none": None,
        "geometry": "era5-cell-area-v1",
        "population": str(population_path),
    }[weighting]
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
    coverage = _spread_coverage(metric_values, latitude, common, cell_areas, population)

    target_root.mkdir(parents=True, exist_ok=True)
    figure_directory = target_root / "figures"
    figure_directory.mkdir(exist_ok=True)
    country_selection = source_metadata.get("country_selection", {})
    region_label = country_selection.get("country_name")
    country_metadata = {
        key: source_metadata[key]
        for key in (
            "input_source",
            "country_cache",
            "country_manifest_fingerprint",
            "country_selection",
        )
        if key in source_metadata
    }
    write_figures(
        summary,
        cellwise,
        figure_directory,
        weighting,
        region_label=region_label,
        country_boundary=country_boundary_from_metadata(source_metadata),
    )
    cellwise_path = target_root / "cellwise_quantile_spread.parquet"
    summary_path = target_root / "global_quantile_spread_summary.parquet"
    anchor_path = target_root / "c_anchor_quantile_differences.parquet"
    extrema_path = target_root / "implementation_extrema_frequency.parquet"
    extrema_csv_path = target_root / "implementation_extrema_frequency.csv"
    coverage_xlsx_path = target_root / "spread_coverage.xlsx"
    metadata_path = target_root / "metadata.json"
    atomic_parquet(cellwise, cellwise_path)
    atomic_parquet(summary, summary_path)
    atomic_parquet(c_anchor, anchor_path)
    atomic_parquet(extrema, extrema_path)
    atomic_csv(_extrema_fractions(extrema, implementations), extrema_csv_path)
    _write_spread_coverage_xlsx(coverage, coverage_xlsx_path)
    atomic_json(
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
            "support_scope": source_metadata["support_scope"],
            "percentiles": list(PERCENTILES),
            "implementations": list(implementations),
            "complete_case_cell_count": len(ids),
            "canonical_cell_count": len(cell_ids),
            "weighting": weighting,
            "extrema_tie_policy": "first-in-implementation-order",
            "extrema_csv_units": "fraction-of-complete-case-weight",
            "spread_coverage": {
                "xlsx": coverage_xlsx_path.name,
                "layout": "transposed-one-sheet-per-region",
                "sheets": ["tropical", "non_tropical", "global"],
                "regions": {
                    "tropical": "abs(cell-center-latitude) <= 23.44 degrees",
                    "non_tropical": "abs(cell-center-latitude) > 23.44 degrees",
                    "global": "all-canonical-grid-cells",
                },
                "denominator": "all-canonical-grid-cells-in-each-region",
                "percentage_format": "numeric-adaptive-nonzero",
                "area_weight_format": "numeric-two-decimal-display",
                "count_format": "exact-integer",
                "thresholds_c": list(SPREAD_COVERAGE_THRESHOLDS_C),
                "threshold_operator": ">=",
                "threshold_equality_atol_c": SPREAD_THRESHOLD_ATOL_C,
                "area_weight_source": "era5-cell-area-v1",
                "population_source": str(population_path),
            },
            "weight_source": weight_source,
            "weight_sum": float(weights.sum()),
            "canonical_weight_sum": float(canonical_weights.sum()),
            "complete_case_weight_fraction": float(
                weights.sum() / canonical_weights.sum()
            ),
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
            "primary_map_metric": _primary_map_metric(summary),
            "spread_map_color_limits_c": list(SPREAD_MAP_COLOR_LIMITS_C),
            **(country_metadata if region_label is not None else {}),
        },
        metadata_path,
    )
    return QuantileSpreadOutputs(
        cellwise_path,
        summary_path,
        anchor_path,
        extrema_path,
        metadata_path,
        figure_directory,
        extrema_csv_path,
        coverage_xlsx_path,
    )


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


def _spread_coverage(
    metrics: dict[str, np.ndarray],
    latitude: np.ndarray,
    common: np.ndarray,
    area_weights: np.ndarray,
    population: np.ndarray,
) -> pa.Table:
    """Use each region's canonical totals; only complete cases can be affected.

    Metrics are aligned to ``common`` cells; coordinates and weights cover the
    canonical grid. Area weights are applied once; population is already a count.
    Empty regions and zero population have undefined (null) fractions.
    """
    tropical = np.abs(latitude) <= TROPICAL_LATITUDE_LIMIT_DEGREES
    rows: list[dict[str, object]] = []
    for region, region_mask in (
        ("tropical", tropical),
        ("non_tropical", ~tropical),
        ("global", np.ones(len(latitude), dtype=bool)),
    ):
        total_area = float(area_weights[region_mask].sum())
        total_population = int(population[region_mask].sum())
        region_complete = region_mask[common]
        area = area_weights[common][region_complete]
        people = population[common][region_complete]
        coverage = {
            "total_land_area_weight": total_area,
            "total_population": total_population,
            "complete_case_land_area_fraction": (
                float(area.sum() / total_area) if total_area > 0 else None
            ),
            "complete_case_population_fraction": (
                float(people.sum() / total_population) if total_population > 0 else None
            ),
        }
        for percentile in PERCENTILES:
            spread = metrics[f"spread_{percentile}_c"][region_complete]
            for threshold in SPREAD_COVERAGE_THRESHOLDS_C:
                # Histogram midpoint subtraction can put exact equality a few ULPs
                # below an integer threshold. This tolerance only absorbs roundoff.
                affected = (spread >= threshold) | np.isclose(
                    spread, threshold, rtol=0.0, atol=SPREAD_THRESHOLD_ATOL_C
                )
                affected_population = int(people[affected].sum())
                affected_area = float(area[affected].sum())
                rows.append(
                    {
                        "region": region,
                        "percentile": percentile,
                        "threshold_c": threshold,
                        "affected_cell_count": int(affected.sum()),
                        "affected_land_area_weight": affected_area,
                        "land_area_fraction": (
                            affected_area / total_area if total_area > 0 else None
                        ),
                        "affected_population": affected_population,
                        "population_fraction": (
                            affected_population / total_population
                            if total_population > 0
                            else None
                        ),
                        **coverage,
                    }
                )
    return pa.Table.from_pylist(rows)


def _write_spread_coverage_xlsx(coverage: pa.Table, path: Path) -> None:
    """Write transposed regional coverage sheets atomically."""

    from openpyxl import Workbook
    from openpyxl.styles import Font

    workbook = Workbook()
    workbook.remove(workbook.active)
    rows = coverage.to_pylist()
    for region in ("tropical", "non_tropical", "global"):
        sheet = workbook.create_sheet(region)
        region_rows = [row for row in rows if row["region"] == region]
        sheet.append(
            ["metric"]
            + [
                f"{str(row['percentile']).upper()} ≥ {row['threshold_c']}°C"
                for row in region_rows
            ]
        )
        for source, label, value_type in COVERAGE_WORKBOOK_ROWS:
            sheet.append([label] + [row[source] for row in region_rows])
            for cell in sheet[sheet.max_row][1:]:
                if value_type == "percentage" and cell.value is not None:
                    cell.number_format = _excel_percentage_format(float(cell.value))
                elif value_type == "integer":
                    cell.number_format = "#,##0"
                elif value_type == "weight":
                    cell.number_format = "#,##0.00"
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for cell in sheet["A"]:
            cell.font = Font(bold=True)
        sheet.freeze_panes = "B2"
        sheet.column_dimensions["A"].width = 40
        for column in sheet.iter_cols(min_col=2, max_col=sheet.max_column):
            sheet.column_dimensions[column[0].column_letter].width = 13

    with atomic_path(path) as temporary:
        workbook.save(temporary)


def _excel_percentage_format(value: float) -> str:
    """Show at least one decimal and enough precision to expose nonzero values."""

    percentage = 100.0 * abs(value)
    if percentage == 0.0 or percentage >= 0.1:
        decimals = 1
    else:
        decimals = min(12, math.ceil(-math.log10(percentage)) + 1)
    return f"0.{decimals * '0'}%"


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


def _primary_map_metric(summary: pa.Table) -> str:
    metrics = summary["metric"].to_pylist()
    candidates = tuple(f"spread_{percentile}_c" for percentile in PERCENTILES)
    return max(
        candidates,
        key=lambda metric: (
            summary["weighted_p95_c"][metrics.index(metric)].as_py(),
            int(metric.removeprefix("spread_p").removesuffix("_c")),
        ),
    )
