from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from polars.testing import assert_frame_equal

from liljegren_analysis import (
    completed_pairwise_dates,
    compute_daily_pairwise,
    initialize_pairwise_output,
    pairwise_metadata,
    persist_daily_pairwise,
)
from liljegren_analysis.comparison.config import (
    DEFAULT_PAIRWISE_IMPLEMENTATIONS,
)
from liljegren_analysis.comparison.pairwise import (
    PAIRWISE_COLUMNS,
    PAIRWISE_GROUP_COLUMNS,
    DailyPairwiseSummaries,
    _enriched_histogram_summary,
    _pairwise_aggregations,
    _pairwise_summary,
)


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
        "wbgt_status": statuses, "wbgt_c": pa.array(values, type=pa.float32()),
        "tg_status": tg_statuses or statuses, "tg_c": pa.array([None if value is None else value + 100 for value in values], type=pa.float32()),
        "tnwb_status": statuses, "tnwb_c": pa.array([None if value is None else value - 100 for value in values], type=pa.float32()),
    }), path)


class PairwiseTests(unittest.TestCase):
    def test_enriched_histogram_preserves_direct_pairwise_statistics(self) -> None:
        values = pl.DataFrame({
            "result_date": [date(2025, 1, 1)] * 6,
            "quantity": ["wbgt"] * 6,
            "implementation_a": ["alpha"] * 6,
            "implementation_b": ["beta"] * 6,
            "area_weight": [1.0, 0.5, 1.5, 2.0, 0.25, 0.75],
            "difference_c": [-5.0, -0.5, -0.1, 0.0, 0.2, 2.0],
        })
        direct = values.group_by(*PAIRWISE_GROUP_COLUMNS).agg(
            *_pairwise_aggregations()
        ).select(PAIRWISE_COLUMNS)
        enriched = _enriched_histogram_summary(values.lazy()).collect(
            engine="streaming"
        )
        derived = _pairwise_summary(enriched)

        self.assertGreater(enriched.height, 1)
        assert_frame_equal(derived, direct, check_exact=False, rel_tol=1e-14)

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
            for name in (
                "sum_difference_c",
                "sum_squared_difference_c2",
                "sum_absolute_difference_c",
            ):
                self.assertTrue(pa.types.is_float64(summaries.pairwise.schema.field(name).type))
            published = persist_daily_pairwise(root / "output", result_date, summaries)
            self.assertTrue((published / "manifest.json").exists())
            self.assertEqual(pq.read_table(published / "pairwise.parquet").num_rows, 9)

    def test_all_implementations_uses_configured_complete_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hsb"
            implementations = DEFAULT_PAIRWISE_IMPLEMENTATIONS
            for index, implementation in enumerate(implementations):
                _write_results(root, implementation, [float(index), float(index + 1)])
            _, summaries, _, _ = compute_daily_pairwise(date(2025, 1, 1), (root,), "all-implementations")
            self.assertEqual(
                summaries.pairwise.num_rows,
                3 * len(implementations) * (len(implementations) - 1) // 2,
            )
            _write_results(root, implementations[0], [1.0, 2.0], cycle_ordinals=[1, 1], row_ids=[_row_id(0), _row_id(0)])
            with self.assertRaisesRegex(ValueError, "no complete"):
                compute_daily_pairwise(date(2025, 1, 1), (root,), "all-implementations")

    def test_all_implementations_excludes_unconfigured_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hsb"
            implementations = (*DEFAULT_PAIRWISE_IMPLEMENTATIONS, "extra")
            for index, implementation in enumerate(implementations):
                _write_results(root, implementation, [float(index)])
            _, summaries, _, _ = compute_daily_pairwise(
                date(2025, 1, 1), (root,), "all-implementations"
            )
            self.assertEqual(
                summaries.pairwise.num_rows,
                3
                * len(DEFAULT_PAIRWISE_IMPLEMENTATIONS)
                * (len(DEFAULT_PAIRWISE_IMPLEMENTATIONS) - 1)
                // 2,
            )
            compared = {
                value
                for column in ("implementation_a", "implementation_b")
                for value in summaries.pairwise[column].to_pylist()
            }
            self.assertEqual(compared, set(DEFAULT_PAIRWISE_IMPLEMENTATIONS))

    def test_all_implementations_requires_every_configured_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hsb"
            for implementation in DEFAULT_PAIRWISE_IMPLEMENTATIONS[:-1]:
                _write_results(root, implementation, [1.0])
            with self.assertRaisesRegex(ValueError, "implementations are missing"):
                compute_daily_pairwise(
                    date(2025, 1, 1), (root,), "all-implementations"
                )

    def test_pairwise_output_requires_explicit_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            root = output / "all-implementations" / "v1"
            root.mkdir(parents=True)
            (root / "metadata.json").write_text('{"schema_version": 1}')
            stale = root / "dates" / "stale"
            stale.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                initialize_pairwise_output(output, "all-implementations")
            initialized = initialize_pairwise_output(
                output, "all-implementations", overwrite=True
            )
            self.assertEqual(initialized, root.resolve())
            self.assertFalse(stale.exists())
            self.assertEqual(
                json.loads((initialized / "metadata.json").read_text()),
                pairwise_metadata("all-implementations"),
            )
            self.assertEqual(
                pairwise_metadata("all-implementations")["implementations"],
                list(DEFAULT_PAIRWISE_IMPLEMENTATIONS),
            )

    def test_completed_dates_require_manifest_and_all_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            root = initialize_pairwise_output(output, "all-implementations")
            complete = root / "dates" / "2025-01-01"
            complete.mkdir(parents=True)
            filenames = ["pairwise.parquet", "abs_histogram.parquet", "quantity_status.parquet"]
            for filename in filenames:
                (complete / filename).touch()
            (complete / "manifest.json").write_text(
                json.dumps({"result_date": "2025-01-01", "files": filenames})
            )
            incomplete = root / "dates" / "2025-01-02"
            incomplete.mkdir()
            (incomplete / "manifest.json").write_text(
                json.dumps({"result_date": "2025-01-02", "files": filenames})
            )
            self.assertEqual(
                completed_pairwise_dates(output, "all-implementations"),
                {date(2025, 1, 1)},
            )

    def test_area_weight_validation_tolerates_reduction_order_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hsb = root / "hsb"
            for index, implementation in enumerate(DEFAULT_PAIRWISE_IMPLEMENTATIONS):
                _write_results(hsb, implementation, [float(index)])
            result_date, summaries, _, _ = compute_daily_pairwise(
                date(2025, 1, 1), (hsb,), "all-implementations"
            )
            status = summaries.quantity_status
            offsets = {"wbgt": 0.0, "tg": 3e-6, "tnwb": 5e-6}
            weights = pa.array(
                [
                    4_568_836.0 + offsets[quantity.as_py()]
                    for quantity in status["quantity"]
                ],
                type=pa.float64(),
            )
            noisy = DailyPairwiseSummaries(
                summaries.pairwise,
                summaries.abs_histogram,
                status.set_column(
                    status.schema.get_field_index("area_weight_sum"),
                    "area_weight_sum",
                    weights,
                ),
            )
            persist_daily_pairwise(
                root / "accepted", result_date, noisy, "all-implementations"
            )

            material_weights = pa.array(
                [
                    4_568_836.0 + ({"wbgt": 0.0, "tg": 0.01, "tnwb": 0.0})[
                        quantity.as_py()
                    ]
                    for quantity in status["quantity"]
                ],
                type=pa.float64(),
            )
            material = DailyPairwiseSummaries(
                summaries.pairwise,
                summaries.abs_histogram,
                status.set_column(
                    status.schema.get_field_index("area_weight_sum"),
                    "area_weight_sum",
                    material_weights,
                ),
            )
            with self.assertRaisesRegex(ValueError, "Quantity status totals"):
                persist_daily_pairwise(
                    root / "rejected", result_date, material, "all-implementations"
                )

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
