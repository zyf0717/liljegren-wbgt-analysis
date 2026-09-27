"""Area and population coverage of persisted cellwise quantile spreads."""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..grid import area_weights
from ..io.artifacts import atomic_path
from .population import load_population_weights
from .spread import PERCENTILES

TROPICAL_LATITUDE_LIMIT_DEGREES = 23.44
SPREAD_COVERAGE_THRESHOLDS_C = tuple(range(1, 9))
SPREAD_THRESHOLD_ATOL_C = 1e-12
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


def export_spread_coverage(input_root: Path, population_weights_path: Path) -> Path:
    """Export coverage using the full source grid as each region's denominator."""
    root = Path(input_root)
    metadata = json.loads((root / "metadata.json").read_text())
    grid = pq.read_table(Path(metadata["input_root"]) / "grid.parquet")
    cells = pq.read_table(root / "cellwise_quantile_spread.parquet")
    ids = np.asarray(grid["cell_id"])
    selected_ids = np.asarray(cells["cell_id"])
    common = np.isin(ids, selected_ids)
    if not np.array_equal(ids[common], selected_ids):
        raise ValueError("Spread cells do not align with the source grid")
    source_ids = grid["source_cell_id"].to_pylist()
    population, _ = load_population_weights(source_ids, population_weights_path)
    coverage = coverage_table(
        {f"spread_{p}_c": np.asarray(cells[f"spread_{p}_c"]) for p in PERCENTILES},
        np.asarray(grid["latitude"]),
        common,
        area_weights(source_ids),
        population,
    )
    path = root / "spread_coverage.xlsx"
    _write_spread_coverage_xlsx(coverage, path)
    return path


def coverage_table(
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
