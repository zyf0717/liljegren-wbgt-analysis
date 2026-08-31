from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis.cellwise.iso7243_plots import (
    plot_cellwise_iso7243_classification,
    summarize_cellwise_iso7243_classification,
)


class CellwiseISO7243PlotTests(unittest.TestCase):
    def test_streamed_summary_and_figure_weightings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "changes"
            root.mkdir()
            source_ids = [
                "era5-global-n000000-e0000000",
                "era5-global-n002500-e0000000",
            ]
            pq.write_table(
                pa.table(
                    {
                        "cell_id": pa.array([0, 2], type=pa.int32()),
                        "source_cell_id": source_ids,
                        "latitude": [0.0, 0.25],
                        "longitude": [0.0, 0.0],
                    }
                ),
                root / "cell_lookup.parquet",
            )
            rows: list[dict[str, int]] = []
            for implementation_a in range(7):
                for implementation_b in range(implementation_a + 1, 7):
                    rows.extend(
                        (
                            {
                                "cell_id": 0,
                                "implementation_a_id": implementation_a,
                                "implementation_b_id": implementation_b,
                                "n_compared": 4,
                                "n_higher": 1 if (implementation_a, implementation_b) == (0, 1) else 0,
                                "n_lower": 1 if (implementation_a, implementation_b) == (0, 1) else 0,
                            },
                            {
                                "cell_id": 2,
                                "implementation_a_id": implementation_a,
                                "implementation_b_id": implementation_b,
                                "n_compared": 4,
                                "n_higher": 0,
                                "n_lower": 0,
                            },
                        )
                    )
            pq.write_table(pa.Table.from_pylist(rows), root / "cell_pair_differences.parquet")
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "row_count": 42,
                        "implementation_pair_count": 21,
                        "quantiles": ["p50", "p90", "p95", "p99"],
                        "implementation_ids": {f"implementation-{index}": index for index in range(7)},
                    }
                )
            )
            population_path = Path(temporary) / "population.parquet"
            pq.write_table(
                pa.table(
                    {
                        "source_cell_id": source_ids,
                        "population": pa.array([10, 0], type=pa.int64()),
                    }
                ),
                population_path,
            )

            summary = summarize_cellwise_iso7243_classification(
                root, population_weights_path=population_path
            )
            np.testing.assert_allclose(summary.changed_fraction, [1.0 / 42.0, 0.0])
            np.testing.assert_allclose(
                summary.histogram_shares[0], [41.0 / 42.0, 0.0, 1.0 / 42.0, 0.0, 0.0]
            )
            np.testing.assert_allclose(
                summary.histogram_shares[2], [20.0 / 21.0, 0.0, 1.0 / 21.0, 0.0, 0.0]
            )
            np.testing.assert_allclose(summary.pair_changed_fraction[0], [0.5, 0.0])

            outputs = plot_cellwise_iso7243_classification(
                root,
                output_directory=Path(temporary) / "figures",
                population_weights_path=population_path,
            )
            self.assertGreater(outputs.png.stat().st_size, 0)
            self.assertGreater(outputs.pairwise_png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
