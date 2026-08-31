from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis.cellwise.grid import grid_from_source_cell_ids
from liljegren_analysis.cellwise.iso7243 import (
    PERCENTILES,
    analyze_cellwise_iso7243_classification,
)
from liljegren_analysis.comparison.sql import EXPECTED_IMPLEMENTATIONS


class CellwiseISO7243Tests(unittest.TestCase):
    def test_complete_case_directional_pair_product(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            implementations = source / "implementations"
            implementations.mkdir(parents=True)
            grid = grid_from_source_cell_ids(
                [
                    "era5-global-n000000-e0000000",
                    "era5-global-n002500-e0000000",
                    "era5-global-s002500-e0000000",
                    "era5-global-s900000-e0000000",
                ]
            ).table
            pq.write_table(grid, source / "grid.parquet")
            (source / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "implementations": list(EXPECTED_IMPLEMENTATIONS),
                        "probabilities": [0.5, 0.9, 0.95, 0.99],
                        "temperature_bin_min_c": -40.0,
                        "temperature_bin_max_c": 60.0,
                    }
                )
            )
            for index, implementation in enumerate(EXPECTED_IMPLEMENTATIONS):
                values = np.asarray(
                    [
                        [25.0, 30.0, 33.0, 34.0],
                        [24.0, 31.0, 32.0, 34.0],
                        [24.0, 31.0, 32.0, 34.0],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                )
                if index == 1:
                    values[0] = [24.0, 30.0, 34.0, 33.0]
                    values[1] = [25.0, 30.0, 33.0, 32.0]
                if index == 2:
                    values[2, 2] = np.nan
                rows: dict[str, object] = {
                    "cell_id": grid["cell_id"],
                    "n_joint": pa.array([10, 10, 10, 0], type=pa.uint32()),
                }
                for percentile_index, percentile in enumerate(PERCENTILES):
                    estimate = values[:, percentile_index]
                    rows[f"wbgt_{percentile}_c"] = estimate
                    rows[f"wbgt_{percentile}_lower_c"] = estimate - 0.01
                    rows[f"wbgt_{percentile}_upper_c"] = estimate + 0.01
                pq.write_table(pa.table(rows), implementations / f"{implementation}.parquet")

            output = analyze_cellwise_iso7243_classification(
                source,
                thresholds_c=[25.0, 30.0, 33.0],
                output_root=Path(temporary) / "output",
            )

            pairs = pq.read_table(output.cell_pair_differences)
            self.assertEqual(pairs.num_rows, 2 * 21)
            self.assertEqual(
                pairs.schema,
                pa.schema(
                    [
                        ("cell_id", pa.int32()),
                        ("implementation_a_id", pa.uint8()),
                        ("implementation_b_id", pa.uint8()),
                        ("n_compared", pa.uint8()),
                        ("n_higher", pa.uint8()),
                        ("n_lower", pa.uint8()),
                    ]
                ),
            )
            first_pair = pairs.slice(0, 2)
            self.assertEqual(first_pair["cell_id"].to_pylist(), [0, 1])
            self.assertEqual(first_pair["implementation_a_id"].to_pylist(), [0, 0])
            self.assertEqual(first_pair["implementation_b_id"].to_pylist(), [1, 1])
            self.assertEqual(first_pair["n_compared"].to_pylist(), [4, 4])
            self.assertEqual(first_pair["n_higher"].to_pylist(), [1, 1])
            self.assertEqual(first_pair["n_lower"].to_pylist(), [0, 2])
            self.assertEqual(pq.ParquetFile(output.cell_pair_differences).num_row_groups, 21)

            lookup = pq.read_table(output.cell_lookup)
            self.assertEqual(lookup["cell_id"].to_pylist(), [0, 1])
            self.assertEqual(lookup.schema.field("cell_id").type, pa.int32())
            self.assertEqual(
                output.classes.read_text().splitlines(),
                [
                    "class_id,class_label,threshold_definition",
                    '0,"< 25 C","wbgt_c < 25"',
                    '1,"[25, 30) C","25 <= wbgt_c < 30"',
                    '2,"[30, 33) C","30 <= wbgt_c < 33"',
                    '3,">= 33 C","wbgt_c >= 33"',
                ],
            )
            metadata = json.loads(output.metadata.read_text())
            self.assertEqual(metadata["complete_case_cell_count"], 2)
            self.assertEqual(metadata["excluded_cell_count"], 2)
            self.assertEqual(metadata["implementation_pair_count"], 21)
            self.assertEqual(metadata["row_count"], 42)
            self.assertEqual(metadata["implementation_ids"]["heatstress"], 0)
            self.assertEqual(metadata["implementation_ids"]["thermofeel"], 6)

            with self.assertRaises(FileExistsError):
                analyze_cellwise_iso7243_classification(
                    source,
                    thresholds_c=[25.0, 30.0, 33.0],
                    output_root=Path(temporary) / "output",
                )
            analyze_cellwise_iso7243_classification(
                source,
                thresholds_c=[25.0, 30.0, 33.0],
                output_root=Path(temporary) / "output",
                force=True,
            )


if __name__ == "__main__":
    unittest.main()
