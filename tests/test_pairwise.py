from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis import compute_daily_pairwise, persist_daily_pairwise
from liljegren_analysis.comparison.config import EXPECTED_IMPLEMENTATIONS
from liljegren_analysis.comparison.pairwise import DailyPairwiseSummaries


def _row_id(index: int) -> str:
    return f"era5-global-n000000-e0000000-20250101{index:02d}"


def _write_results(hsb_run: Path, implementation: str, values: list[float | None], *, statuses: list[str] | None = None, tg_statuses: list[str] | None = None, cycle_ordinals: list[int] | None = None, row_ids: list[str] | None = None) -> None:
    path = hsb_run / "results" / "2025-01-01" / "0001" / implementation / "results.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    statuses = statuses or ["ok"] * len(values)
    pq.write_table(pa.table({
        "cycle_ordinal": pa.array(cycle_ordinals or [1] * len(values), type=pa.uint32()),
        "row_id": row_ids or [_row_id(index) for index in range(len(values))],
        "row_ordinal": pa.array(range(len(values)), type=pa.uint32()),
        "wbgt_status": statuses, "wbgt_c": values,
        "tg_status": tg_statuses or statuses, "tg_c": [None if value is None else value + 100 for value in values],
        "tnwb_status": statuses, "tnwb_c": [None if value is None else value - 100 for value in values],
    }), path)


class PairwiseTests(unittest.TestCase):
    def test_pairwise_totals_and_parquet_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_results(root / "first", "alpha", [10.0, 20.0, None])
            _write_results(root / "first", "beta", [8.0, 99.0, 4.0], statuses=["ok", "failed", "ok"])
            _write_results(root / "second", "gamma", [7.0, 15.0, 1.0])
            result_date, summaries, _, _ = compute_daily_pairwise(date(2025, 1, 1), (root / "first", root / "second"))
            rows = {(row["implementation_a"], row["implementation_b"]): row for row in summaries.pairwise.to_pylist() if row["quantity"] == "wbgt"}
            self.assertEqual(rows[("alpha", "beta")]["n"], 1)
            self.assertEqual(rows[("alpha", "gamma")]["sum_difference_c"], 8.0)
            histogram_counts = {}
            for row in summaries.abs_histogram.to_pylist():
                key = row["quantity"], row["implementation_a"], row["implementation_b"]
                histogram_counts[key] = histogram_counts.get(key, 0) + row["n"]
            self.assertEqual(histogram_counts[("wbgt", "alpha", "gamma")], 2)
            published = persist_daily_pairwise(root / "output", result_date, summaries)
            self.assertTrue((published / "manifest.json").exists())
            self.assertEqual(pq.read_table(published / "pairwise.parquet").num_rows, 9)

    def test_all_implementations_requires_exact_complete_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hsb"
            for index, implementation in enumerate(EXPECTED_IMPLEMENTATIONS):
                _write_results(root, implementation, [float(index), float(index + 1)])
            _, summaries, _, _ = compute_daily_pairwise(date(2025, 1, 1), (root,), "all-implementations")
            self.assertEqual(summaries.pairwise.num_rows, 63)
            _write_results(root, "heatstress", [1.0, 2.0], cycle_ordinals=[1, 1], row_ids=[_row_id(0), _row_id(0)])
            with self.assertRaisesRegex(ValueError, "no complete"):
                compute_daily_pairwise(date(2025, 1, 1), (root,), "all-implementations")

    def test_invalid_histogram_is_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_results(root, "alpha", [1.0])
            _write_results(root, "beta", [2.0])
            result_date, summaries, _, _ = compute_daily_pairwise(date(2025, 1, 1), (root,))
            invalid = DailyPairwiseSummaries(summaries.pairwise, summaries.abs_histogram.slice(1), summaries.quantity_status)
            with self.assertRaisesRegex(ValueError, "histogram totals"):
                persist_daily_pairwise(root / "output", result_date, invalid)


if __name__ == "__main__":
    unittest.main()
