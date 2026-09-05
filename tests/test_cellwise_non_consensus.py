from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis.cellwise.non_consensus import (
    analyze_cellwise_iso7243_non_consensus,
)
from liljegren_analysis.cellwise.non_consensus_plots import (
    plot_cellwise_iso7243_non_consensus,
)
from liljegren_analysis.comparison.config import EXPECTED_IMPLEMENTATIONS


_CELLS = ("era5-global-n000000-e0000000", "era5-global-n002500-e0000000")


def _write_day(root: Path, result_date: str, day: int) -> None:
    for index, implementation in enumerate(EXPECTED_IMPLEMENTATIONS):
        path = root / "results" / result_date / "0001" / implementation / "results.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = [20.0, 20.0]
        if day == 1:
            if index == 0:
                values[0] = 30.0
            if index in (0, 1):
                values[1] = 30.0
        tg_status = ["ok", "ok"]
        if day == 2 and implementation == "meteor":
            tg_status[1] = "invalid"
        pq.write_table(
            pa.table(
                {
                    "cycle_ordinal": pa.array([1, 1], type=pa.uint32()),
                    "row_id": [
                        f"{cell}-{result_date.replace('-', '')}00" for cell in _CELLS
                    ],
                    "row_ordinal": pa.array([0, 1], type=pa.uint32()),
                    "wbgt_status": ["ok", "ok"],
                    "wbgt_c": values,
                    "tg_status": tg_status,
                    "tg_c": [30.0, 30.0],
                    "tnwb_status": ["ok", "ok"],
                    "tnwb_c": [20.0, 20.0],
                }
            ),
            path,
        )


class CellwiseNonConsensusTests(unittest.TestCase):
    def test_joint_valid_annual_percentages_and_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "hsb-run"
            _write_day(run, "2025-01-01", 1)
            _write_day(run, "2025-01-02", 2)
            output = analyze_cellwise_iso7243_non_consensus(
                run,
                start_date="2025-01-01",
                end_date="2025-01-02",
                thresholds_c=[25.0],
                output_root=root / "output",
            )
            table = pq.read_table(output.cellwise_non_consensus)
            self.assertEqual(
                table.schema,
                pa.schema(
                    [
                        ("cell_id", pa.int32()),
                        ("n_joint", pa.uint32()),
                        ("n_non_consensus", pa.uint32()),
                        ("non_consensus_percent", pa.float64()),
                    ]
                ),
            )
            self.assertEqual(table["n_joint"].to_pylist(), [2, 1])
            self.assertEqual(table["n_non_consensus"].to_pylist(), [1, 1])
            self.assertEqual(table["non_consensus_percent"].to_pylist(), [50.0, 100.0])
            metadata = json.loads(output.metadata.read_text())
            self.assertEqual(metadata["support_scope"], "all-implementations-all-quantities")
            self.assertEqual(metadata["joint_observation_count"], 3)
            self.assertEqual(metadata["non_consensus_observation_count"], 2)
            with self.assertRaises(FileExistsError):
                analyze_cellwise_iso7243_non_consensus(
                    run,
                    start_date="2025-01-01",
                    end_date="2025-01-02",
                    thresholds_c=[25.0],
                    output_root=root / "output",
                )
            figure = plot_cellwise_iso7243_non_consensus(output.metadata.parent)
            self.assertGreater(figure.png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
