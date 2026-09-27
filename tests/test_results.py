from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.results import date_range, discover_results, iter_result_batches


def _write_result(
    hsb_run: Path,
    implementation: str,
    value: float,
    result_date: str = "2025-01-01",
    run_id: str = "0001",
) -> None:
    path = (
        hsb_run / "results" / result_date / run_id / implementation / "results.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "cycle_ordinal": pa.array([1], type=pa.uint32()),
                "row_id": ["era5-global-n012500-e1037500-2025010100"],
                "row_ordinal": pa.array([1], type=pa.uint32()),
                "wbgt_status": ["ok"],
                "wbgt_c": pa.array([value], type=pa.float32()),
                "tg_status": ["ok"],
                "tg_c": pa.array([value + 10.0], type=pa.float32()),
                "tnwb_status": ["ok"],
                "tnwb_c": pa.array([value + 20.0], type=pa.float32()),
                "unrelated_diagnostic": ["ignored"],
            }
        ),
        path,
    )


class ResultReadingTests(unittest.TestCase):
    def test_files_preserve_provenance_and_deterministic_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "hsb_run"
            _write_result(root, "meteor", 32.0, run_id="0002")
            _write_result(root, "heatstress", 30.0, run_id="0001")
            files = discover_results((root,), (date(2025, 1, 1),))
            self.assertEqual(
                [
                    (item.result_date.isoformat(), item.run_id, item.implementation)
                    for item in files
                ],
                [
                    ("2025-01-01", "0001", "heatstress"),
                    ("2025-01-01", "0002", "meteor"),
                ],
            )
            self.assertTrue(all(item.path.name == "results.parquet" for item in files))

    def test_exact_date_scopes_and_arrow_batch_column_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "hsb_run"
            _write_result(root, "heatstress", 30.0, "2025-01-01")
            _write_result(root, "heatstress", 31.0, "2025-01-02")
            _write_result(root, "heatstress", 99.0, "2025-01-03")
            files = discover_results(
                (root,), date_range(date(2025, 1, 1), date(2025, 1, 2))
            )
            batches = list(
                iter_result_batches(
                    files, ("row_id", "tg_status", "tg_c"), batch_size=1
                )
            )
            self.assertEqual(
                [batch.schema.names for batch in batches],
                [["row_id", "tg_status", "tg_c"]] * 2,
            )
            self.assertEqual(
                [batch.column(2)[0].as_py() for batch in batches], [40.0, 41.0]
            )
            explicit = discover_results(
                (root,), [date(2025, 1, 1), date(2025, 1, 3), date(2025, 1, 3)]
            )
            self.assertEqual(
                [item.result_date.isoformat() for item in explicit],
                ["2025-01-01", "2025-01-03"],
            )


if __name__ == "__main__":
    unittest.main()
