from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis import HSBRunResults, RESULT_SCHEMA_ID
from liljegren_analysis.io.results import iter_result_batches


def _write_result(hsb_run: Path, implementation: str, value: float, result_date: str = "2025-01-01", run_id: str = "0001") -> None:
    path = hsb_run / "results" / result_date / run_id / implementation / "results.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "cycle_ordinal": pa.array([1], type=pa.uint32()),
        "row_id": ["era5-global-n012500-e1037500-2025010100"],
        "row_ordinal": pa.array([1], type=pa.uint32()),
        "wbgt_status": ["ok"], "wbgt_c": pa.array([value], type=pa.float32()),
        "tg_status": ["ok"], "tg_c": pa.array([value + 10.0], type=pa.float32()),
        "tnwb_status": ["ok"], "tnwb_c": pa.array([value + 20.0], type=pa.float32()),
        "unrelated_diagnostic": ["ignored"],
    }), path)


class HSBRunResultsTests(unittest.TestCase):
    def test_result_schema_id_is_float32_parquet_contract(self) -> None:
        self.assertEqual(
            RESULT_SCHEMA_ID,
            "hsb-wbgt-comparison-results-parquet-float32-v1",
        )

    def test_run_manifest_must_declare_supported_result_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "hsb_run"
            _write_result(root, "heatstress", 30.0)
            (root / "run-manifest.json").write_text(
                json.dumps(
                    {
                        "execution": {
                            "result_format": "parquet",
                            "result_schema_id": "hsb-wbgt-comparison-results-csv-v3",
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Unsupported HSB run result"):
                HSBRunResults.from_hsb_run(root)

    def test_files_preserve_provenance_and_deterministic_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "hsb_run"
            _write_result(root, "meteor", 32.0, run_id="0002")
            _write_result(root, "heatstress", 30.0, run_id="0001")
            files = HSBRunResults.from_hsb_run(root).files()
            self.assertEqual(
                [(item.result_date.isoformat(), item.run_id, item.implementation) for item in files],
                [("2025-01-01", "0001", "heatstress"), ("2025-01-01", "0002", "meteor")],
            )
            self.assertTrue(all(item.path.name == "results.parquet" for item in files))

    def test_exact_date_scopes_and_arrow_batch_column_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "hsb_run"
            _write_result(root, "heatstress", 30.0, "2025-01-01")
            _write_result(root, "heatstress", 31.0, "2025-01-02")
            _write_result(root, "heatstress", 99.0, "2025-01-03")
            files = HSBRunResults.from_hsb_run(root).files(start_date="2025-01-01", end_date="2025-01-02")
            batches = list(iter_result_batches(files, ("row_id", "tg_status", "tg_c"), batch_size=1))
            self.assertEqual([batch.schema.names for batch in batches], [["row_id", "tg_status", "tg_c"]] * 2)
            self.assertEqual([batch.column(2)[0].as_py() for batch in batches], [40.0, 41.0])
            explicit = HSBRunResults.from_hsb_run(root).files(dates=["2025-01-01", "2025-01-03", "2025-01-03"])
            self.assertEqual([item.result_date.isoformat() for item in explicit], ["2025-01-01", "2025-01-03"])

    def test_scope_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "hsb_run"
            _write_result(root, "heatstress", 30.0)
            results = HSBRunResults.from_hsb_run(root)
            with self.assertRaisesRegex(ValueError, "only one"):
                results.files(result_date="2025-01-01", start_date="2025-01-01", end_date="2025-01-02")
            with self.assertRaisesRegex(ValueError, "provided together"):
                results.files(start_date="2025-01-01")
            with self.assertRaisesRegex(ValueError, "must not be after"):
                results.files(start_date="2025-01-02", end_date="2025-01-01")
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                results.files(run_id=1, run_ids=[1, 2])
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                results.files(implementations=[])
            with self.assertRaisesRegex(TypeError, "not a string"):
                results.files(dates="2025-01-01")  # type: ignore[arg-type]

    def test_rejects_legacy_and_incompatible_schemas(self) -> None:
        cases = (
            ({"status": ["ok"], "value_c": [30.0]}, "legacy status/value_c"),
            ({"wbgt_status": pa.array([1], type=pa.int32())}, "status types"),
            ({"tg_c": pa.array(["invalid"])}, "value types"),
        )
        for replacement, message in cases:
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory) / "hsb_run"
                _write_result(root, "heatstress", 30.0)
                path = next(root.rglob("results.parquet"))
                if set(replacement) == {"status", "value_c"}:
                    table = pa.table({"cycle_ordinal": pa.array([1], type=pa.uint32()), "row_id": ["era5-global-n012500-e1037500-2025010100"], "row_ordinal": pa.array([1], type=pa.uint32()), **replacement})
                else:
                    table = pq.read_table(path)
                    for column, value in replacement.items():
                        table = table.set_column(table.column_names.index(column), column, value)
                pq.write_table(table, path)
                with self.assertRaisesRegex(ValueError, message):
                    HSBRunResults.from_hsb_run(root).files()


if __name__ == "__main__":
    unittest.main()
