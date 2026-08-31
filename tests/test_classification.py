from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from liljegren_analysis.classification import (
    analyze_era5_country_cache,
    analyze_nasa_power_run,
    analyze_wbgt_classification,
)


class ClassificationTests(unittest.TestCase):
    def test_shared_engine_classifies_and_summarizes_pairs(self) -> None:
        observations = pd.DataFrame(
            {
                "sample_id": [1, 1, 2, 2],
                "timestamp": pd.to_datetime(
                    ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"] * 2
                ),
                "implementation": ["a", "b", "a", "b"],
                "wbgt_c": [24.9, 25.0, 31.1, 30.9],
            }
        )

        analysis = analyze_wbgt_classification(
            observations,
            (25.0, 31.0),
            pair_keys=["sample_id", "timestamp"],
            month_column="timestamp",
        )

        summary = analysis.outputs["pair_change_summary.csv"].iloc[0]
        self.assertEqual(summary["n_paired"], 2)
        self.assertEqual(summary["n_changed"], 2)
        self.assertEqual(summary["a_higher_class"], 1)
        self.assertEqual(summary["b_higher_class"], 1)
        self.assertEqual(analysis.classified["class_id"].tolist(), [0, 1, 2, 1])

    def test_era5_adapter_reads_only_preselected_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "observations.parquet"
            pd.DataFrame(
                {
                    "result_date": [date(2025, 1, 1)] * 4,
                    "run_id": [1] * 4,
                    "implementation": ["a", "b", "a", "b"],
                    "cycle_ordinal": [1] * 4,
                    "row_id": ["one", "one", "two", "two"],
                    "cell_id": ["cell", "cell", "cell", "cell"],
                    "wbgt_status": ["ok"] * 4,
                    "wbgt_c": [24.0, 26.0, 30.0, 30.0],
                    "tg_status": ["ok"] * 4,
                    "tg_c": [30.0] * 4,
                    "tnwb_status": ["ok"] * 4,
                    "tnwb_c": [20.0] * 4,
                }
            ).to_parquet(cache)

            analysis = analyze_era5_country_cache(
                cache, date(2025, 1, 1), date(2025, 1, 1), (25.0,)
            )

            self.assertEqual(analysis.observation_summary.loc[0, "result_rows"], 4)
            self.assertEqual(
                analysis.classification.outputs["pair_change_summary.csv"].loc[
                    0, "n_changed"
                ],
                1,
            )

    def test_nasa_adapter_loads_matched_result_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dataset").mkdir()
            pd.DataFrame(
                {
                    "row_id": ["one", "two"],
                    "location_id": ["alpha", "alpha"],
                    "interval_start_utc": [
                        "2025-01-01T00:00:00Z",
                        "2025-01-01T01:00:00Z",
                    ],
                    "longitude_deg_east": [0.0, 0.0],
                    "latitude_deg_north": [0.0, 0.0],
                }
            ).to_csv(root / "dataset" / "workload.csv", index=False)
            for implementation, values in (("a", [24.0, 30.0]), ("b", [26.0, 30.0])):
                directory = root / "results" / "2025" / "0001" / implementation
                directory.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "cycle_ordinal": [1, 1],
                        "row_id": ["one", "two"],
                        "row_ordinal": [1, 2],
                        "wbgt_status": ["ok", "ok"],
                        "wbgt_c": values,
                    }
                ).to_parquet(directory / "results.parquet")

            analysis = analyze_nasa_power_run(
                root,
                date(2025, 1, 1),
                date(2025, 1, 1),
                (25.0,),
                location_ids=["alpha"],
            )

            self.assertEqual(analysis.workload_summary.loc[0, "workload_rows"], 2)
            self.assertEqual(
                analysis.outputs["location_pair_change_summary.csv"].loc[0, "n_changed"], 1
            )


if __name__ == "__main__":
    unittest.main()
