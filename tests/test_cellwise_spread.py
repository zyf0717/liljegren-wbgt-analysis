from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis.cellwise.grid import grid_from_source_cell_ids
from liljegren_analysis.cellwise.spread import analyze_cellwise_quantile_spread
from liljegren_analysis.comparison.sql import EXPECTED_IMPLEMENTATIONS


class CellwiseQuantileSpreadTests(unittest.TestCase):
    def test_analysis_uses_upper_tail_complete_case_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            implementation_root = root / "implementations"
            implementation_root.mkdir(parents=True)
            grid = grid_from_source_cell_ids(
                ["era5-global-n000000-e0000000", "era5-global-s900000-e0000000"]
            ).table
            pq.write_table(grid, root / "grid.parquet")
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "wbgt_bin_min_c": -40.0,
                        "wbgt_bin_max_c": 60.0,
                        "wbgt_bin_width_c": 0.02,
                        "probabilities": [0.5, 0.9, 0.95, 0.99],
                        "implementations": list(EXPECTED_IMPLEMENTATIONS),
                        "histogram_version": 3,
                    }
                )
            )
            for index, implementation in enumerate(EXPECTED_IMPLEMENTATIONS):
                rows: dict[str, object] = {
                    "cell_id": [0, 1],
                    "n_valid": [10, 10],
                }
                for percentile, base in (
                    ("p50", 25.0),
                    ("p90", 30.0),
                    ("p95", 32.0),
                    ("p99", 34.0),
                ):
                    values = np.asarray([base + index, base + index + 0.5])
                    rows[f"{percentile}_c"] = values
                    rows[f"{percentile}_lower_c"] = values - 0.01
                    rows[f"{percentile}_upper_c"] = values + 0.01
                pq.write_table(
                    pa.table(rows), implementation_root / f"{implementation}.parquet"
                )

            output = analyze_cellwise_quantile_spread(
                root, output_root=Path(temporary) / "analysis", sample_size=2
            )

            cellwise = pq.read_table(output.cellwise_spread)
            self.assertEqual(cellwise["spread_p50_c"].to_pylist(), [6.0, 6.0])
            self.assertEqual(cellwise["spread_p90_c"].to_pylist(), [6.0, 6.0])
            self.assertEqual(
                cellwise["min_implementation_p95"].to_pylist(), ["heatstress"] * 2
            )
            self.assertEqual(
                cellwise["max_implementation_p99"].to_pylist(), ["thermofeel"] * 2
            )
            summary = pq.read_table(output.global_summary)
            self.assertEqual(
                summary["metric"].to_pylist(),
                ["spread_p50_c", "spread_p90_c", "spread_p95_c", "spread_p99_c"],
            )
            self.assertEqual(pq.read_table(output.c_anchor_differences).num_rows, 24)
            self.assertEqual(pq.read_table(output.extrema_frequency).num_rows, 28)


if __name__ == "__main__":
    unittest.main()
