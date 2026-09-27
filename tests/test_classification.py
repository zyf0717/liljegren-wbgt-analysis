from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from reference_classification import reference_classification
from sample_settings import DEFAULT_IMPLEMENTATIONS

from analysis.classification.era5 import analyze_era5_country_cache


def write_cache(source, path):
    from test_cellwise_country_source import cache_scopes

    source = source.copy()
    cell = "era5-global-n000000-e0000000"
    source["cell_id"] = cell
    source["run_id"] = source["run_id"].astype(str)
    hours = source.groupby("result_date")["row_id"].transform(
        lambda ids: pd.factorize(ids)[0]
    )
    source["row_ordinal"] = hours.astype("uint32")
    source["row_id"] = [
        f"{cell}-{day:%Y%m%d}{hour:02}" for day, hour in zip(source.result_date, hours)
    ]
    source.to_parquet(path)
    path.with_name("manifest.json").write_text(
        json.dumps({"scopes": cache_scopes(path)})
    )
    return source


class ClassificationTests(unittest.TestCase):
    def test_era5_adapter_reads_only_preselected_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "observations.parquet"
            rows = []
            for implementation in (*DEFAULT_IMPLEMENTATIONS, "extra"):
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
                            24.0 if implementation == "argonne-liljegren-c" else 26.0,
                        ),
                        ("two", 30.0),
                    )
                )
            source = pd.DataFrame(rows)
            source = write_cache(source, cache)

            analysis = analyze_era5_country_cache(
                cache,
                date(2025, 1, 1),
                date(2025, 1, 1),
                (25.0,),
                implementations=DEFAULT_IMPLEMENTATIONS,
            )

            self.assertEqual(
                analysis["observation_summary.csv"].loc[0, "result_rows"], 12
            )
            self.assertEqual(
                analysis["observation_summary.csv"].loc[0, "implementations"],
                len(DEFAULT_IMPLEMENTATIONS),
            )
            pair_summary = analysis["pair_change_summary.csv"]
            pair = pair_summary.loc[
                pair_summary["implementation_a"].eq("argonne-liljegren-c")
                & pair_summary["implementation_b"].eq("heatstress")
            ].iloc[0]
            self.assertEqual(pair["n_changed"], 1)
            self.assertEqual(
                analysis["non_consensus_summary.csv"].loc[0, "n_non_consensus"],
                1,
            )
            selected = source.loc[
                source["implementation"].isin(DEFAULT_IMPLEMENTATIONS)
            ].copy()
            expected = reference_classification(
                selected,
                (25.0,),
                implementations=DEFAULT_IMPLEMENTATIONS,
            )
            for filename, expected_frame in expected.items():
                pd.testing.assert_frame_equal(
                    analysis[filename],
                    expected_frame,
                    check_dtype=False,
                )

    def test_era5_streaming_matches_reference_across_months(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "observations.parquet"
            rows = []
            for result_date, samples in (
                (
                    date(2025, 1, 31),
                    (("one", (24.0, 26.0, 24.0)), ("two", (30.0,) * 3)),
                ),
                (
                    date(2025, 2, 1),
                    (("three", (24.0, 26.0, None)), ("four", (30.0,) * 3)),
                ),
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
            source = write_cache(source, cache)

            analysis = analyze_era5_country_cache(
                cache,
                date(2025, 1, 1),
                date(2025, 2, 28),
                (25.0,),
                implementations=("a", "b", "c"),
            )
            valid = source["wbgt_status"].eq("ok") & source["wbgt_c"].notna()
            expected = reference_classification(
                source.loc[valid].copy(),
                (25.0,),
                implementations=("a", "b", "c"),
            )

            for filename, expected_frame in expected.items():
                pd.testing.assert_frame_equal(
                    analysis[filename],
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
                analysis["quantity_coverage.csv"], coverage, check_dtype=False
            )
            self.assertEqual(
                analysis["monthly_non_consensus_summary.csv"]["n_joint"].tolist(),
                [2, 1],
            )


if __name__ == "__main__":
    unittest.main()


def test_shuffled_cache_missing_implementation_and_month_boundary(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from iso7243_fixtures import write_day
    from test_cellwise_country_source import _country_cache, _rewrite_cache

    run = tmp_path / "run"
    for day in ("2025-01-31", "2025-02-01"):
        write_day(run, day)
    cache = _country_cache(run, tmp_path / "country/observations.parquet")
    rows = pq.read_table(cache).to_pandas()
    rows = rows.loc[
        ~(rows.result_date.eq(date(2025, 1, 31)) & rows.implementation.eq("thermofeel"))
    ]
    _rewrite_cache(cache, pa.Table.from_pandas(rows, preserve_index=False))
    actual = analyze_era5_country_cache(
        cache,
        date(2025, 1, 1),
        date(2025, 2, 28),
        [25, 30],
        implementations=DEFAULT_IMPLEMENTATIONS,
    )
    valid = rows.wbgt_status.eq("ok") & np.isfinite(rows.wbgt_c)
    expected = reference_classification(
        rows.loc[valid],
        [25, 30],
        implementations=DEFAULT_IMPLEMENTATIONS,
    )
    for name, table in expected.items():
        pd.testing.assert_frame_equal(actual[name], table, check_dtype=False)


def test_threshold_equality_and_incomplete_support(tmp_path):
    rows = []
    for hour, values in enumerate(((24.9, 25.0, 24.9), (31.1, 30.9, None))):
        for name, value in zip(("a", "b", "c"), values):
            rows.append(
                {
                    "result_date": date(2025, 1, 1),
                    "run_id": "1",
                    "cycle_ordinal": 1,
                    "row_id": str(hour),
                    "implementation": name,
                    "wbgt_status": "ok" if value is not None else "failed",
                    "wbgt_c": value,
                    "tg_status": "ok",
                    "tg_c": 35.0,
                    "tnwb_status": "ok",
                    "tnwb_c": 20.0,
                }
            )
    cache = tmp_path / "observations.parquet"
    source = write_cache(pd.DataFrame(rows), cache)
    result = analyze_era5_country_cache(
        cache,
        date(2025, 1, 1),
        date(2025, 1, 1),
        (25.0, 31.0),
        implementations=("a", "b", "c"),
    )
    summary = result["pair_change_summary.csv"].iloc[0]
    assert (summary.n_paired, summary.n_changed) == (2, 2)
    assert (summary.a_higher_class, summary.b_higher_class) == (1, 1)
    transitions = result["pair_transition_counts.csv"]
    ab = transitions.loc[transitions.implementation_b.eq("b")]
    assert list(zip(ab.class_id_a, ab.class_id_b)) == [(0, 1), (2, 1)]
    consensus = result["non_consensus_summary.csv"].iloc[0]
    assert (consensus.n_joint, consensus.n_non_consensus) == (1, 1)
    for name, table in reference_classification(
        source, (25.0, 31.0), ("a", "b", "c")
    ).items():
        pd.testing.assert_frame_equal(result[name], table, check_dtype=False)
