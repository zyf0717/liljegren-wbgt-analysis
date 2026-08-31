from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis.cellwise import compute_cellwise_wbgt_quantiles
from liljegren_analysis.cellwise import pipeline as cellwise_pipeline


def _write_day(hsb_run: Path, result_date: str, values: list[float | None]) -> None:
    path = hsb_run / "results" / result_date / "0001" / "example" / "results.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = ["era5-global-n000000-e0000000", "era5-global-n002500-e0000000"]
    pq.write_table(
        pa.table(
            {
                "cycle_ordinal": pa.array([1, 1], type=pa.uint32()),
                "row_id": [f"{cell}-{result_date.replace('-', '')}00" for cell in cells],
                "row_ordinal": pa.array([0, 1], type=pa.uint32()),
                "wbgt_status": ["ok", "ok"],
                "wbgt_c": pa.array(values, type=pa.float64()),
                "tg_status": ["ok", "ok"],
                "tg_c": [30.0, 30.0],
                "tnwb_status": ["ok", "ok"],
                "tnwb_c": [20.0, 20.0],
            }
        ),
        path,
    )


class CellwiseShardMergeTests(unittest.TestCase):
    def test_pipeline_merges_daily_shards_and_reconciles_valid_counts(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            hsb_run = Path(root) / "hsb_run"
            output = Path(root) / "output"
            _write_day(hsb_run, "2025-01-01", [20.0, 30.0])
            _write_day(hsb_run, "2025-01-02", [21.0, None])
            output_path = compute_cellwise_wbgt_quantiles(
                hsb_run,
                implementation="example",
                start_date="2025-01-01",
                end_date="2025-01-02",
                output_root=output,
            )
            result = pq.read_table(output_path).sort_by("cell_id")
            self.assertEqual(result["n_valid"].to_pylist(), [2, 1])
            self.assertEqual(result["p50_lower_c"].to_pylist(), [20.0, 30.0])
            self.assertEqual(result["p90_lower_c"].to_pylist(), [21.0, 30.0])
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["implementations"], ["example"])
            self.assertEqual(metadata["cell_count"], 2)
            self.assertEqual(
                metadata["implementation_provenance"]["example"]
                ["exact_quantile_validation"]["checked_quantile_count"],
                8,
            )
            self.assertEqual(list((output / ".work").glob("*.mmap")), [])

    def test_resume_keeps_only_completed_daily_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            hsb_run = Path(root) / "hsb_run"
            output = Path(root) / "output"
            _write_day(hsb_run, "2025-01-01", [20.0, 30.0])
            _write_day(hsb_run, "2025-01-02", [21.0, 31.0])
            original_scan = cellwise_pipeline._scan_valid_day

            def fail_second_day(*args: object, **kwargs: object) -> tuple[object, object]:
                if args[1].isoformat() == "2025-01-02":
                    raise RuntimeError("simulated scan interruption")
                return original_scan(*args, **kwargs)  # type: ignore[arg-type]

            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with mock.patch.object(
                    cellwise_pipeline, "_scan_valid_day", side_effect=fail_second_day
                ):
                    compute_cellwise_wbgt_quantiles(
                        hsb_run,
                        implementation="example",
                        start_date="2025-01-01",
                        end_date="2025-01-02",
                        output_root=output,
                    )
            self.assertTrue((output / ".work" / "example.progress.json").exists())
            output_path = compute_cellwise_wbgt_quantiles(
                hsb_run,
                implementation="example",
                start_date="2025-01-01",
                end_date="2025-01-02",
                output_root=output,
            )
            result = pq.read_table(output_path).sort_by("cell_id")
            self.assertEqual(result["n_valid"].to_pylist(), [2, 2])
