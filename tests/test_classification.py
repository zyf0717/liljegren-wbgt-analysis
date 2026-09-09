from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from liljegren_analysis.cellwise import DEFAULT_JOINT_IMPLEMENTATIONS
from liljegren_analysis.classification import (
    DEFAULT_CLASSIFICATION_IMPLEMENTATIONS,
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
        non_consensus = analysis.outputs["non_consensus_summary.csv"].iloc[0]
        self.assertEqual(non_consensus["n_joint"], 2)
        self.assertEqual(non_consensus["n_non_consensus"], 2)
        self.assertEqual(non_consensus["non_consensus_percent"], 100.0)

    def test_shared_engine_non_consensus_requires_all_target_support(self) -> None:
        observations = pd.DataFrame(
            {
                "sample_id": [1, 1, 1, 2, 2],
                "timestamp": pd.to_datetime(["2025-01-01T00:00:00Z"] * 5),
                "implementation": ["a", "b", "c", "a", "b"],
                "wbgt_c": [24.0, 26.0, 24.0, 24.0, 26.0],
            }
        )

        analysis = analyze_wbgt_classification(
            observations,
            (25.0,),
            pair_keys=["sample_id", "timestamp"],
            month_column="timestamp",
            implementations=("a", "b", "c"),
        )

        summary = analysis.outputs["non_consensus_summary.csv"].iloc[0]
        self.assertEqual(summary["n_joint"], 1)
        self.assertEqual(summary["n_non_consensus"], 1)

    def test_era5_adapter_reads_only_preselected_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "observations.parquet"
            rows = []
            for implementation in (*DEFAULT_CLASSIFICATION_IMPLEMENTATIONS, "extra"):
                rows.extend(
                    {
                        "result_date": date(2025, 1, 1),
                        "run_id": 1,
                        "implementation": implementation,
                        "cycle_ordinal": 1,
                        "row_id": row_id,
                        "cell_id": "cell",
                        "wbgt_status": "ok",
                        "wbgt_c": value,
                        "tg_status": "ok",
                        "tg_c": 30.0,
                        "tnwb_status": "ok",
                        "tnwb_c": 20.0,
                    }
                    for row_id, value in (
                        (
                            "one",
                            24.0
                            if implementation == "argonne-liljegren-c"
                            else 26.0,
                        ),
                        ("two", 30.0),
                    )
                )
            source = pd.DataFrame(rows)
            source.to_parquet(cache)

            analysis = analyze_era5_country_cache(
                cache, date(2025, 1, 1), date(2025, 1, 1), (25.0,)
            )

            self.assertEqual(analysis.observation_summary.loc[0, "result_rows"], 12)
            self.assertEqual(
                analysis.observation_summary.loc[0, "implementations"],
                len(DEFAULT_CLASSIFICATION_IMPLEMENTATIONS),
            )
            self.assertIsNone(analysis.classification.classified)
            self.assertIsNone(analysis.classification.paired)
            pair_summary = analysis.classification.outputs[
                "pair_change_summary.csv"
            ]
            pair = pair_summary.loc[
                pair_summary["implementation_a"].eq("argonne-liljegren-c")
                & pair_summary["implementation_b"].eq("heatstress")
            ].iloc[0]
            self.assertEqual(
                pair["n_changed"], 1
            )
            self.assertEqual(
                analysis.outputs["non_consensus_summary.csv"].loc[
                    0, "n_non_consensus"
                ],
                1,
            )
            selected = source.loc[
                source["implementation"].isin(DEFAULT_CLASSIFICATION_IMPLEMENTATIONS)
            ].copy()
            expected = analyze_wbgt_classification(
                selected,
                (25.0,),
                pair_keys=[
                    "result_date",
                    "run_id",
                    "cycle_ordinal",
                    "row_id",
                    "cell_id",
                ],
                month_column="result_date",
                implementations=DEFAULT_CLASSIFICATION_IMPLEMENTATIONS,
            )
            for filename, expected_frame in expected.outputs.items():
                pd.testing.assert_frame_equal(
                    analysis.classification.outputs[filename],
                    expected_frame,
                    check_dtype=False,
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
            for implementation in (*DEFAULT_CLASSIFICATION_IMPLEMENTATIONS, "extra"):
                values = (
                    [24.0, 30.0]
                    if implementation == "argonne-liljegren-c"
                    else [26.0, 30.0]
                )
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
                set(analysis.status_summary["implementation"]),
                set(DEFAULT_CLASSIFICATION_IMPLEMENTATIONS),
            )
            pair_summary = analysis.outputs["location_pair_change_summary.csv"]
            pair = pair_summary.loc[
                pair_summary["implementation_a"].eq("argonne-liljegren-c")
                & pair_summary["implementation_b"].eq("heatstress")
            ].iloc[0]
            self.assertEqual(
                pair["n_changed"], 1
            )
            self.assertEqual(
                analysis.outputs["location_non_consensus_summary.csv"].loc[
                    0, "n_non_consensus"
                ],
                1,
            )

    def test_era5_streaming_matches_shared_engine_across_months(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "observations.parquet"
            rows = []
            for result_date, samples in (
                (date(2025, 1, 31), (("one", (24.0, 26.0, 24.0)), ("two", (30.0,) * 3))),
                (date(2025, 2, 1), (("three", (24.0, 26.0, None)), ("four", (30.0,) * 3))),
            ):
                for implementation_index, implementation in enumerate(("a", "b", "c")):
                    for row_id, values in samples:
                        value = values[implementation_index]
                        rows.append(
                            {
                                "result_date": result_date,
                                "run_id": "0001",
                                "implementation": implementation,
                                "cycle_ordinal": 1,
                                "row_id": row_id,
                                "cell_id": "cell",
                                "wbgt_status": "ok" if value is not None else "failed",
                                "wbgt_c": value,
                                "tg_status": "ok",
                                "tg_c": 30.0,
                                "tnwb_status": "ok",
                                "tnwb_c": 20.0,
                            }
                        )
            source = pd.DataFrame(rows)
            source.to_parquet(cache)

            analysis = analyze_era5_country_cache(
                cache,
                date(2025, 1, 1),
                date(2025, 2, 28),
                (25.0,),
                implementations=("a", "b", "c"),
            )
            valid = source["wbgt_status"].eq("ok") & source["wbgt_c"].notna()
            expected = analyze_wbgt_classification(
                source.loc[valid].copy(),
                (25.0,),
                pair_keys=[
                    "result_date",
                    "run_id",
                    "cycle_ordinal",
                    "row_id",
                    "cell_id",
                ],
                month_column="result_date",
                implementations=("a", "b", "c"),
            )

            for filename, expected_frame in expected.outputs.items():
                pd.testing.assert_frame_equal(
                    analysis.classification.outputs[filename],
                    expected_frame,
                    check_dtype=False,
                )
            coverage = pd.concat(
                [
                    source.assign(
                        quantity=quantity,
                        quantity_status=source[f"{quantity}_status"],
                        quantity_value_present=(
                            source[f"{quantity}_status"].eq("ok")
                            & source[f"{quantity}_c"].notna()
                        ),
                    )[
                        [
                            "quantity",
                            "implementation",
                            "quantity_status",
                            "quantity_value_present",
                        ]
                    ]
                    for quantity in ("wbgt", "tg", "tnwb")
                ]
            )
            coverage = (
                coverage.groupby(
                    [
                        "quantity",
                        "implementation",
                        "quantity_status",
                        "quantity_value_present",
                    ],
                    as_index=False,
                    sort=True,
                )
                .size()
                .rename(columns={"size": "n"})
            )
            pd.testing.assert_frame_equal(
                analysis.quantity_coverage, coverage, check_dtype=False
            )
            self.assertEqual(
                analysis.outputs["monthly_non_consensus_summary.csv"][
                    "n_joint"
                ].tolist(),
                [2, 1],
            )

    def test_country_classification_default_matches_iso7243_targets(self) -> None:
        self.assertEqual(
            DEFAULT_CLASSIFICATION_IMPLEMENTATIONS,
            DEFAULT_JOINT_IMPLEMENTATIONS,
        )


if __name__ == "__main__":
    unittest.main()
