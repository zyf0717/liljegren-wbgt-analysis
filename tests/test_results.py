from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


from liljegren_analysis import HSBRunResults


def _write_result(
    hsb_run: Path,
    implementation: str,
    value: float,
    result_date: str = "2025-01-01",
    run_id: str = "0001",
) -> None:
    path = (
        hsb_run
        / "results"
        / result_date
        / run_id
        / implementation
        / "results.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "cycle_ordinal": pa.array([1], type=pa.uint32()),
                "row_id": ["era5-global-n012500-e1037500-2025010100"],
                "row_ordinal": pa.array([1], type=pa.uint32()),
                "wbgt_status": ["ok"],
                "wbgt_c": [value],
                "tg_status": ["ok"],
                "tg_c": [value + 10.0],
                "tnwb_status": ["ok"],
                "tnwb_c": [value + 20.0],
                "unrelated_diagnostic": ["ignored"],
            }
        ),
        path,
    )


class HSBRunResultsTests(unittest.TestCase):
    def test_shared_relations_can_be_queried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"
            _write_result(first, "heatstress", 30.0)
            _write_result(second, "meteor", 32.0)

            with duckdb.connect() as connection:
                combined = HSBRunResults.from_hsb_run(
                    first, connection=connection
                ).scan(result_date="2025-01-01").union(
                    HSBRunResults.from_hsb_run(second, connection=connection).scan(
                        result_date="2025-01-01"
                    )
                )
                summary = combined.query(
                    "results",
                    """
                    SELECT implementation, count(*) AS observations, avg(wbgt_c) AS mean_c
                    FROM results
                    WHERE wbgt_status = 'ok'
                    GROUP BY implementation
                    ORDER BY implementation
                    """,
                ).fetchall()

            self.assertEqual(summary, [("heatstress", 1, 30.0), ("meteor", 1, 32.0)])

    def test_inclusive_date_range_prunes_by_exact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hsb_run = Path(temporary_directory) / "hsb_run"
            _write_result(hsb_run, "heatstress", 30.0, "2025-01-01")
            _write_result(hsb_run, "heatstress", 31.0, "2025-01-02")
            _write_result(hsb_run, "heatstress", 99.0, "2025-01-03")

            relation = HSBRunResults.from_hsb_run(hsb_run).scan(
                start_date="2025-01-01",
                end_date="2025-01-02",
            )

            self.assertEqual(
                relation.aggregate("count(*), avg(wbgt_c)").fetchone(),
                (2, 30.5),
            )

            explicit = HSBRunResults.from_hsb_run(hsb_run).scan(
                dates=["2025-01-01", "2025-01-03", "2025-01-03"]
            )
            self.assertEqual(
                explicit.aggregate("count(*), avg(wbgt_c)").fetchone(),
                (2, 64.5),
            )

    def test_multiple_scopes_and_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hsb_run = Path(temporary_directory) / "hsb_run"
            for run_id in ("0001", "0002"):
                _write_result(hsb_run, "heatstress", 30.0, run_id=run_id)
                _write_result(hsb_run, "meteor", 32.0, run_id=run_id)
                _write_result(hsb_run, "pywbgt", 99.0, run_id=run_id)

            relation = HSBRunResults.from_hsb_run(hsb_run).scan(
                run_ids=[1, 2, 2],
                implementations=["heatstress", "meteor"],
                include_filename=True,
            )
            rows = relation.project(
                "run_id, implementation, filename"
            ).order("run_id, implementation").fetchall()

            self.assertEqual(
                [(run_id, implementation) for run_id, implementation, _ in rows],
                [
                    ("0001", "heatstress"),
                    ("0001", "meteor"),
                    ("0002", "heatstress"),
                    ("0002", "meteor"),
                ],
            )
            self.assertTrue(all(filename.endswith("results.parquet") for _, _, filename in rows))

    def test_date_scope_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hsb_run = Path(temporary_directory) / "hsb_run"
            _write_result(hsb_run, "heatstress", 30.0)
            results = HSBRunResults.from_hsb_run(hsb_run)

            with self.assertRaisesRegex(ValueError, "only one"):
                results.scan(
                    result_date="2025-01-01",
                    start_date="2025-01-01",
                    end_date="2025-01-02",
                )
            with self.assertRaisesRegex(ValueError, "provided together"):
                results.scan(start_date="2025-01-01")
            with self.assertRaisesRegex(ValueError, "must not be after"):
                results.scan(start_date="2025-01-02", end_date="2025-01-01")
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                results.scan(run_id=1, run_ids=[1, 2])
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                results.scan(implementations=[])
            with self.assertRaisesRegex(TypeError, "not a string"):
                results.scan(dates="2025-01-01")  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                results.scan(quantity="tg", quantities=("wbgt", "tg"))
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                results.scan(quantities=[])
            with self.assertRaisesRegex(TypeError, "not a string"):
                results.scan(quantities="tg")  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "Unsupported result quantities"):
                results.scan(quantity="invalid")  # type: ignore[arg-type]

    def test_quantity_selection_is_qualified_and_column_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hsb_run = Path(temporary_directory) / "hsb_run"
            _write_result(hsb_run, "heatstress", 30.0)

            relation = HSBRunResults.from_hsb_run(hsb_run).scan(quantity="tg")

            self.assertEqual(
                relation.columns,
                [
                    "result_date",
                    "run_id",
                    "implementation",
                    "cycle_ordinal",
                    "row_id",
                    "row_ordinal",
                    "tg_status",
                    "tg_c",
                ],
            )
            plan = relation.explain()
            self.assertIn("tg_status", plan)
            self.assertIn("tg_c", plan)
            self.assertNotIn("wbgt_status", plan)
            self.assertNotIn("tnwb_status", plan)

            selected = HSBRunResults.from_hsb_run(hsb_run).scan(
                quantities=("tnwb", "wbgt", "tnwb")
            )
            self.assertEqual(
                selected.columns[-4:],
                ["wbgt_status", "wbgt_c", "tnwb_status", "tnwb_c"],
            )

    def test_rejects_legacy_generic_result_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hsb_run = Path(temporary_directory) / "hsb_run"
            path = (
                hsb_run
                / "results"
                / "2025-01-01"
                / "0001"
                / "heatstress"
                / "results.parquet"
            )
            path.parent.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "cycle_ordinal": pa.array([1], type=pa.uint32()),
                        "row_id": ["era5-global-n012500-e1037500-2025010100"],
                        "row_ordinal": pa.array([1], type=pa.uint32()),
                        "status": ["ok"],
                        "value_c": [30.0],
                    }
                ),
                path,
            )

            with self.assertRaisesRegex(ValueError, "legacy status/value_c"):
                HSBRunResults.from_hsb_run(hsb_run).scan()

    def test_rejects_incompatible_quantity_column_types(self) -> None:
        cases = (
            ("wbgt_status", pa.array([1], type=pa.int32()), "status types"),
            ("tg_c", pa.array(["invalid"]), "value types"),
        )
        for column, invalid_values, message in cases:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as root:
                hsb_run = Path(root) / "hsb_run"
                _write_result(hsb_run, "heatstress", 30.0)
                path = next(hsb_run.rglob("results.parquet"))
                table = pq.read_table(path)
                index = table.column_names.index(column)
                pq.write_table(table.set_column(index, column, invalid_values), path)

                with self.assertRaisesRegex(ValueError, message):
                    HSBRunResults.from_hsb_run(hsb_run).scan()


if __name__ == "__main__":
    unittest.main()
