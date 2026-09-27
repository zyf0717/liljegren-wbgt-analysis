from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from openpyxl import load_workbook

from analysis.cellwise.spread import (
    PERCENTILES,
    _cellwise_table,
    _extrema_fractions,
    _extrema_frequency,
    analyze_cellwise_quantile_spread,
    summarize_quantile_spread,
)
from analysis.cellwise.spread_coverage import (
    _excel_percentage_format,
    export_spread_coverage,
)
from analysis.cellwise.spread_coverage import coverage_table as _spread_coverage
from analysis.cellwise.spread_plots import (
    plot_quantile_spread,
    plot_stacked_cellwise_quantile_spread,
)
from analysis.config import (
    DEFAULT_IMPLEMENTATIONS,
    DEFAULT_PROBABILITIES,
    REFERENCE_IMPLEMENTATION,
)
from analysis.grid import grid_from_source_cell_ids

IMPLEMENTATIONS = DEFAULT_IMPLEMENTATIONS


class CellwiseQuantileSpreadTests(unittest.TestCase):
    def test_stacked_plot_combines_all_quantities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "quantile-spread"
            table = pa.table(
                {
                    "latitude": [0.0],
                    "longitude": [0.0],
                    "spread_p50_c": [1.0],
                    "spread_p95_c": [9.0],
                }
            )
            for quantity in ("wbgt", "tnwb", "tg"):
                directory = root / quantity
                directory.mkdir(parents=True)
                pq.write_table(table, directory / "cellwise_quantile_spread.parquet")

            outputs = plot_stacked_cellwise_quantile_spread(root)

            self.assertEqual(
                outputs,
                (
                    root.resolve() / "figures/stacked_spread_p50.png",
                    root.resolve() / "figures/stacked_spread_p95.png",
                ),
            )
            self.assertTrue(all(output.stat().st_size > 0 for output in outputs))

    def test_analysis_uses_upper_tail_complete_case_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            implementation_root = root / "implementations"
            implementation_root.mkdir(parents=True)
            grid = grid_from_source_cell_ids(
                ["era5-global-n000000-e0000000", "era5-global-s900000-e0000000"]
            ).table
            pq.write_table(grid, root / "grid.parquet")
            population_path = Path(temporary) / "population.parquet"
            pq.write_table(
                pa.table(
                    {
                        "source_cell_id": grid["source_cell_id"],
                        "population": pa.array([1, 3], type=pa.int64()),
                    }
                ),
                population_path,
            )
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "support_scope": "all-implementations-all-quantities",
                        "temperature_bin_min_c": -40.0,
                        "temperature_bin_max_c": 60.0,
                        "temperature_bin_width_c": 0.02,
                        "probabilities": list(DEFAULT_PROBABILITIES),
                        "implementations": list(IMPLEMENTATIONS),
                        "histogram_version": 3,
                    }
                )
            )
            for index, implementation in enumerate(IMPLEMENTATIONS):
                rows: dict[str, object] = {
                    "cell_id": [0, 1],
                    "n_joint": [10, 10],
                }
                for position, percentile in enumerate(PERCENTILES):
                    base = 20.0 + 2 * position
                    values = np.asarray([base + index, base + index + 0.5])
                    rows[f"wbgt_{percentile}_c"] = values
                    rows[f"wbgt_{percentile}_lower_c"] = values - 0.01
                    rows[f"wbgt_{percentile}_upper_c"] = values + 0.01
                pq.write_table(
                    pa.table(rows), implementation_root / f"{implementation}.parquet"
                )

            output = analyze_cellwise_quantile_spread(
                root,
                output_root=Path(temporary) / "analysis",
                population_weights_path=population_path,
            )
            tables, _ = summarize_quantile_spread(root)
            for name, table in tables.items():
                self.assertTrue(table.equals(pq.read_table(output / f"{name}.parquet")))
            self.assertFalse((output / "figures").exists())
            plot_quantile_spread(output)
            export_spread_coverage(output, population_path)
            self.assertFalse((output / "tropical_spread_coverage.csv").exists())
            self.assertFalse((output / "spread_coverage.csv").exists())

            self.assertTrue(
                ((output / "figures") / "spread_by_percentile.png").is_file()
            )
            self.assertFalse(
                ((output / "figures") / "spread_by_percentile.pdf").exists()
            )
            for percentile in PERCENTILES:
                self.assertTrue(
                    ((output / "figures") / f"spread_{percentile}_map.png").is_file()
                )
            self.assertEqual(list((output / "figures").glob("*.pdf")), [])

            cellwise = pq.read_table((output / "cellwise_quantile_spread.parquet"))
            expected_spread = float(len(IMPLEMENTATIONS) - 1)
            self.assertEqual(
                cellwise["spread_p50_c"].to_pylist(), [expected_spread] * 2
            )
            self.assertEqual(
                cellwise["spread_p95_c"].to_pylist(), [expected_spread] * 2
            )
            self.assertEqual(
                cellwise["min_implementation_p95"].to_pylist(),
                [IMPLEMENTATIONS[0]] * 2,
            )
            self.assertEqual(
                cellwise["max_implementation_p95"].to_pylist(),
                [IMPLEMENTATIONS[-1]] * 2,
            )
            summary = pq.read_table((output / "global_quantile_spread_summary.parquet"))
            self.assertEqual(
                summary["metric"].to_pylist(),
                [f"spread_{percentile}_c" for percentile in PERCENTILES],
            )
            anchor = pq.read_table((output / "c_anchor_quantile_differences.parquet"))
            self.assertEqual(
                anchor.num_rows, len(PERCENTILES) * (len(IMPLEMENTATIONS) - 1)
            )
            self.assertEqual(
                set(anchor["anchor_implementation"].to_pylist()),
                {REFERENCE_IMPLEMENTATION},
            )
            self.assertEqual(
                pq.read_table(
                    (output / "implementation_extrema_frequency.parquet")
                ).num_rows,
                len(PERCENTILES) * len(IMPLEMENTATIONS),
            )
            csv = pcsv.read_csv((output / "implementation_extrema_frequency.csv"))
            self.assertEqual(
                csv.column_names, ["implementation", "extremum", *PERCENTILES]
            )
            self.assertEqual(csv.num_rows, 2 * len(IMPLEMENTATIONS))
            self.assertEqual(
                csv["p50"].to_pylist(),
                [1.0] + [0.0] * (2 * len(IMPLEMENTATIONS) - 2) + [1.0],
            )
            self.assertEqual(
                csv.to_pylist(),
                _extrema_fractions(
                    pq.read_table(
                        (output / "implementation_extrema_frequency.parquet")
                    ),
                    IMPLEMENTATIONS,
                ).to_pylist(),
            )
            workbook = load_workbook((output / "spread_coverage.xlsx"), data_only=True)
            self.assertEqual(
                workbook.sheetnames, ["tropical", "non_tropical", "global"]
            )
            tropical_sheet = workbook["tropical"]
            self.assertEqual(tropical_sheet.max_column, 1 + 8 * len(PERCENTILES))
            self.assertEqual(
                [tropical_sheet.cell(1, column).value for column in range(2, 5)],
                ["P50 ≥ 1°C", "P50 ≥ 2°C", "P50 ≥ 3°C"],
            )
            tropical_rows = {
                tropical_sheet.cell(row, 1).value: [
                    tropical_sheet.cell(row, column).value
                    for column in range(2, tropical_sheet.max_column + 1)
                ]
                for row in range(2, tropical_sheet.max_row + 1)
            }
            expected_affected = [
                int(threshold <= expected_spread) for threshold in range(1, 9)
            ] * len(PERCENTILES)
            self.assertEqual(tropical_rows["affected_cell_count"], expected_affected)
            self.assertEqual(
                tropical_rows["affected_land_area_weight"], expected_affected
            )
            self.assertEqual(tropical_rows["land_area_percent"], expected_affected)
            self.assertEqual(tropical_rows["affected_population"], expected_affected)
            self.assertEqual(tropical_rows["population_percent"], expected_affected)
            self.assertEqual(tropical_rows["complete_case_land_area_percent"], [1] * 16)
            self.assertEqual(
                tropical_rows["complete_case_population_percent"], [1] * 16
            )
            self.assertEqual(tropical_rows["total_population"], [1] * 16)
            percentage_row = next(
                row
                for row in range(2, tropical_sheet.max_row + 1)
                if tropical_sheet.cell(row, 1).value == "land_area_percent"
            )
            for column in range(2, tropical_sheet.max_column + 1):
                self.assertEqual(
                    tropical_sheet.cell(percentage_row, column).number_format, "0.0%"
                )

            unweighted = analyze_cellwise_quantile_spread(
                root,
                output_root=Path(temporary) / "analysis-unweighted",
                weighting="none",
                population_weights_path=population_path,
            )
            self.assertEqual(
                pq.read_table((unweighted / "cellwise_quantile_spread.parquet"))[
                    "weight"
                ].to_pylist(),
                [1.0, 1.0],
            )

            population_weighted = analyze_cellwise_quantile_spread(
                root,
                output_root=Path(temporary) / "analysis-population",
                weighting="population",
                population_weights_path=population_path,
            )
            population_cellwise = pq.read_table(
                (population_weighted / "cellwise_quantile_spread.parquet")
            )
            self.assertEqual(population_cellwise["weight"].to_pylist(), [1.0, 3.0])
            population_metadata = json.loads(
                (population_weighted / "metadata.json").read_text()
            )
            self.assertEqual(population_metadata["weighting"], "population")
            self.assertEqual(
                population_metadata["extrema_tie_policy"],
                "first-in-implementation-order",
            )
            self.assertEqual(
                population_metadata["population_weights_path"],
                str(population_path),
            )
            for other in (unweighted, population_weighted):
                export_spread_coverage(other, population_path)
                other_workbook = load_workbook(
                    (other / "spread_coverage.xlsx"), data_only=True
                )
                for sheet_name in workbook.sheetnames:
                    self.assertEqual(
                        list(other_workbook[sheet_name].values),
                        list(workbook[sheet_name].values),
                    )

    def test_excel_percentage_format_exposes_small_nonzero_values(self) -> None:
        self.assertEqual(_excel_percentage_format(0.0), "0.0%")
        self.assertEqual(_excel_percentage_format(0.071), "0.0%")
        self.assertEqual(_excel_percentage_format(0.000022), "0.0000%")

    def test_regional_coverage_boundaries_weights_support_and_inclusive_thresholds(
        self,
    ) -> None:
        common = np.array([True, True, True, True, False])
        regional = _spread_coverage(
            {
                "spread_p50_c": np.array([1 - 4e-15, 8, 9, 9]),
                "spread_p95_c": np.array([2, 1 - 1e-8, 9, 9]),
            },
            np.array([-23.44, 23.44, -23.45, 23.45, 0]),
            common,
            np.array([1, 3, 50, 70, 4], dtype=float),
            np.array([10, 30, 1000, 2000, 60], dtype=np.int64),
        )
        self.assertEqual(regional.num_rows, 48)
        table = regional.filter(np.asarray(regional["region"]) == "tropical")
        rows = table.to_pylist()
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(row["total_land_area_weight"], 8)
            self.assertEqual(row["total_population"], 100)
            self.assertEqual(row["complete_case_land_area_fraction"], 0.5)
            self.assertEqual(row["complete_case_population_fraction"], 0.4)
        np.testing.assert_allclose(
            table["land_area_fraction"].to_numpy(),
            [0.5] + [3 / 8] * 7 + [1 / 8] * 2 + [0] * 6,
        )
        self.assertEqual(
            table["affected_population"].to_pylist(),
            [40] + [30] * 7 + [10] * 2 + [0] * 6,
        )
        np.testing.assert_allclose(
            table["affected_land_area_weight"].to_numpy(),
            [4] + [3] * 7 + [1, 1] + [0] * 6,
        )
        np.testing.assert_allclose(
            table["population_fraction"].to_numpy(),
            [0.4] + [0.3] * 7 + [0.1] * 2 + [0] * 6,
        )
        regions = {
            name: regional.filter(np.asarray(regional["region"]) == name).to_pylist()
            for name in ("tropical", "non_tropical", "global")
        }
        for tropical, non_tropical, global_row in zip(*regions.values(), strict=True):
            self.assertEqual(non_tropical["total_land_area_weight"], 120)
            self.assertEqual(non_tropical["total_population"], 3000)
            self.assertEqual(non_tropical["land_area_fraction"], 1.0)
            self.assertEqual(non_tropical["population_fraction"], 1.0)
            for column in (
                "affected_cell_count",
                "affected_population",
                "total_population",
            ):
                self.assertEqual(
                    global_row[column], tropical[column] + non_tropical[column]
                )
            for fraction, denominator in (
                ("land_area_fraction", "total_land_area_weight"),
                ("population_fraction", "total_population"),
                ("complete_case_land_area_fraction", "total_land_area_weight"),
                ("complete_case_population_fraction", "total_population"),
            ):
                self.assertAlmostEqual(
                    global_row[fraction] * global_row[denominator],
                    tropical[fraction] * tropical[denominator]
                    + non_tropical[fraction] * non_tropical[denominator],
                )

    def test_tropical_coverage_empty_region_zero_population_and_missing_support(
        self,
    ) -> None:
        for latitudes, population, common, area_fraction, population_fraction in (
            ([24, -24], [10, 20], [True, True], None, None),
            ([0, 20], [0, 0], [True, True], 1.0, None),
            ([0, 24], [10, 20], [False, True], 0.0, 0.0),
        ):
            with self.subTest(latitudes=latitudes, population=population):
                table = _spread_coverage(
                    {f"spread_{p}_c": np.full(sum(common), 8.0) for p in PERCENTILES},
                    np.array(latitudes),
                    np.array(common),
                    np.ones(2),
                    np.array(population, dtype=np.int64),
                )
                for row in table.to_pylist():
                    if row["region"] != "tropical":
                        continue
                    self.assertEqual(row["land_area_fraction"], area_fraction)
                    self.assertEqual(row["population_fraction"], population_fraction)
                    self.assertEqual(row["affected_population"], 0)

    def test_extrema_csv_fractions_preserve_weights_ties_and_percentiles(self) -> None:
        implementations = ("a", "b", "c")
        matrices = {
            "p50": [[1, 1, 3], [3, 2, 1]],
            "p95": [[2, 2, 2], [2, 2, 2]],
        }
        tied = [[2, 2, 2], [2, 2, 2]]
        values = {
            name: {
                percentile: np.asarray(matrices.get(percentile, tied), dtype=float)[
                    :, index
                ]
                for percentile in PERCENTILES
            }
            for index, name in enumerate(implementations)
        }
        for weights in (np.array([1.0, 3.0]), np.ones(2)):
            with self.subTest(weights=weights):
                table, _ = _cellwise_table(
                    np.arange(2),
                    np.zeros(2),
                    np.zeros(2),
                    weights,
                    values,
                    implementations,
                )
                result = _extrema_fractions(
                    _extrema_frequency(table, weights, implementations), implementations
                )
                first = weights[0] / weights.sum()
                remaining = 1.0 - first
                np.testing.assert_allclose(
                    result["p50"].to_numpy(), [first, remaining, 0, 0, remaining, first]
                )
                np.testing.assert_allclose(result["p95"].to_numpy(), [1, 1, 0, 0, 0, 0])
                for percentile in PERCENTILES:
                    fractions = result[percentile].to_numpy().reshape(3, 2)
                    np.testing.assert_allclose(fractions.sum(axis=0), [1, 1])

    def test_rejects_old_cellwise_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "Unsupported cellwise input schema"
            ):
                analyze_cellwise_quantile_spread(root, output_root=root / "out")


if __name__ == "__main__":
    unittest.main()
