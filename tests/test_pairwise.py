from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis import compute_daily_pairwise, persist_daily_pairwise
from liljegren_analysis.area_weighted import _ERA5_LATITUDE_WEIGHTS
from liljegren_analysis.pairwise import (
    DailyPairwiseSummaries,
    _combined_results,
)
from liljegren_analysis.pairwise_sql import (
    EXPECTED_IMPLEMENTATIONS,
    build_daily_pairwise_sql,
)


def _row_id(index: int) -> str:
    return f"era5-global-n000000-e0000000-20250101{index:02d}"


def _write_results(
    campaign: Path,
    implementation: str,
    values: list[float | None],
    *,
    statuses: list[str] | None = None,
    cycle_ordinals: list[int] | None = None,
    row_ids: list[str] | None = None,
) -> None:
    path = (
        campaign
        / "results"
        / "2025-01-01"
        / "0001"
        / implementation
        / "results.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(values)
    pq.write_table(
        pa.table(
            {
                "cycle_ordinal": pa.array(
                    cycle_ordinals or [1] * count,
                    type=pa.uint32(),
                ),
                "row_id": row_ids or [_row_id(index) for index in range(count)],
                "row_ordinal": pa.array(range(count), type=pa.uint32()),
                "status": statuses or ["ok"] * count,
                "value_c": values,
            }
        ),
        path,
    )


class PairwiseTests(unittest.TestCase):
    def _compute(self, root: Path) -> DailyPairwiseSummaries:
        first = root / "first"
        second = root / "second"
        _write_results(first, "alpha", [10.0, 20.0, None])
        _write_results(
            first,
            "beta",
            [8.0, 99.0, 4.0],
            statuses=["ok", "failed", "ok"],
        )
        _write_results(second, "gamma", [7.0, 15.0, 1.0])

        result_date, summaries, worker_pid, compute_seconds = (
            compute_daily_pairwise(
                date(2025, 1, 1),
                (first, second),
                threads=1,
            )
        )
        self.assertEqual(result_date, date(2025, 1, 1))
        self.assertEqual(worker_pid, os.getpid())
        self.assertGreaterEqual(compute_seconds, 0.0)
        return summaries

    def test_compute_emits_exact_totals_and_mergeable_distributions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            summaries = self._compute(Path(temporary_directory))

        rows = {
            (row["implementation_a"], row["implementation_b"]): row
            for row in summaries.pairwise.to_pylist()
        }
        self.assertEqual(
            list(rows),
            [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")],
        )
        self.assertEqual(rows[("alpha", "beta")]["n"], 1)
        self.assertEqual(rows[("alpha", "beta")]["sum_difference_c"], 2.0)

        alpha_gamma = rows[("alpha", "gamma")]
        self.assertEqual(alpha_gamma["n"], 2)
        self.assertEqual(alpha_gamma["sum_difference_c"], 8.0)
        self.assertEqual(alpha_gamma["sum_squared_difference_c2"], 34.0)
        self.assertEqual(alpha_gamma["sum_absolute_difference_c"], 8.0)
        self.assertEqual(alpha_gamma["min_difference_c"], 3.0)
        self.assertEqual(alpha_gamma["max_difference_c"], 5.0)
        self.assertEqual(alpha_gamma["max_absolute_difference_c"], 5.0)

        for table in (summaries.abs_histogram,):
            with duckdb.connect() as connection:
                connection.register("summary", table)
                grouped = {
                    (row[0], row[1]): row[2:]
                    for row in connection.execute(
                    """
                    SELECT implementation_a, implementation_b,
                           sum(n), sum(area_weight_sum)
                    FROM summary
                    GROUP BY implementation_a, implementation_b
                    ORDER BY implementation_a, implementation_b
                    """
                    ).fetchall()
                }
            for pair, total in rows.items():
                self.assertEqual(grouped[pair][0], total["n"])
                self.assertAlmostEqual(
                    grouped[pair][1],
                    total["area_weight_sum"],
                )

    def test_all_implementations_uses_one_complete_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            campaign = Path(temporary_directory) / "campaign"
            for index, implementation in enumerate(EXPECTED_IMPLEMENTATIONS):
                values: list[float | None] = [
                    float(index),
                    float(index + 10),
                    float(index + 20),
                ]
                arguments = {}
                if implementation == EXPECTED_IMPLEMENTATIONS[0]:
                    values.insert(1, float(index))
                    arguments = {
                        "cycle_ordinals": [1, 1, 1, 1],
                        "row_ids": [_row_id(0), _row_id(0), _row_id(1), _row_id(2)],
                    }
                if implementation == EXPECTED_IMPLEMENTATIONS[-1]:
                    values[1] = None
                _write_results(campaign, implementation, values, **arguments)
            _write_results(campaign, "unexpected", [1.0, 2.0, 3.0])

            _, summaries, _, _ = compute_daily_pairwise(
                date(2025, 1, 1),
                (campaign,),
                threads=1,
                sample_scope="all-implementations",
            )

        rows = summaries.pairwise.to_pylist()
        self.assertEqual(len(rows), 21)
        self.assertEqual({row["n"] for row in rows}, {1})
        observed = {
            row[key]
            for row in rows
            for key in ("implementation_a", "implementation_b")
        }
        self.assertEqual(observed, set(EXPECTED_IMPLEMENTATIONS))

    def test_all_implementations_rejects_empty_complete_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            campaign = Path(temporary_directory) / "campaign"
            for implementation in EXPECTED_IMPLEMENTATIONS[:-1]:
                _write_results(campaign, implementation, [1.0, 2.0, 3.0])
            with self.assertRaisesRegex(ValueError, "no complete"):
                compute_daily_pairwise(
                    date(2025, 1, 1),
                    (campaign,),
                    threads=1,
                    sample_scope="all-implementations",
                )

    def test_compute_validates_inputs_and_overlapping_scopes(self) -> None:
        with self.assertRaisesRegex(ValueError, "runs is empty"):
            compute_daily_pairwise(date(2025, 1, 1), (), threads=1)
        with self.assertRaisesRegex(ValueError, "threads must be positive"):
            compute_daily_pairwise(date(2025, 1, 1), (Path("unused"),), 0)
        with self.assertRaisesRegex(ValueError, "Unsupported sample scope"):
            compute_daily_pairwise(
                date(2025, 1, 1),
                (Path("unused"),),
                1,
                sample_scope="invalid",  # type: ignore[arg-type]
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            campaign = Path(temporary_directory) / "campaign"
            _write_results(campaign, "alpha", [1.0])
            _write_results(campaign, "beta", [2.0])
            with self.assertRaisesRegex(ValueError, "Overlapping result scope"):
                compute_daily_pairwise(
                    date(2025, 1, 1),
                    (campaign, campaign),
                    1,
                )

    def test_query_plan_has_one_source_scan_self_join_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            campaign = Path(temporary_directory) / "campaign"
            _write_results(campaign, "alpha", [1.0])
            _write_results(campaign, "beta", [2.0])
            with duckdb.connect() as connection:
                connection.register(
                    "_era5_latitude_weights",
                    _ERA5_LATITUDE_WEIGHTS,
                )
                results = _combined_results(
                    date(2025, 1, 1),
                    (campaign,),
                    connection,
                )
                plan = results.query(
                    "results",
                    build_daily_pairwise_sql("pairwise"),
                ).explain()

        # DuckDB prints PARQUET_SCAN twice inside one operator box: once as
        # its heading and once as its function name.
        self.assertEqual(plan.count("PARQUET_SCAN"), 2)
        self.assertEqual(plan.count("HASH_GROUP_BY"), 1)
        self.assertEqual(plan.count("HASH_JOIN"), 2)

    def test_all_implementations_sql_uses_wide_complete_rows(self) -> None:
        sql = build_daily_pairwise_sql("all-implementations")

        self.assertNotIn("sampled_results", sql)
        self.assertNotIn("_expected_implementations", sql)
        self.assertNotIn("INNER JOIN complete_observations", sql)
        self.assertNotIn("CROSS JOIN LATERAL", sql)
        self.assertEqual(sql.count("unnest(["), 4)
        self.assertEqual(sql.count("max(value_c) FILTER"), 7)
        self.assertEqual(sql.count("count(*) FILTER"), 10)

    def test_persistence_is_atomic_idempotent_and_mode_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summaries = self._compute(root)
            database = root / "pairwise.duckdb"
            with duckdb.connect(str(database)) as connection:
                persist_daily_pairwise(connection, date(2025, 1, 1), summaries)
                persist_daily_pairwise(connection, date(2025, 1, 1), summaries)
                self.assertEqual(
                    connection.execute("SELECT * FROM pairwise_schema").fetchall(),
                    [(4, "pairwise", 1, 1)],
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT count(*) FROM daily_pairwise"
                    ).fetchone()[0],
                    3,
                )

                invalid = DailyPairwiseSummaries(
                    summaries.pairwise,
                    summaries.abs_histogram.slice(1),
                )
                with self.assertRaisesRegex(ValueError, "totals differ"):
                    persist_daily_pairwise(
                        connection,
                        date(2025, 1, 1),
                        invalid,
                    )
                self.assertEqual(
                    connection.execute(
                        "SELECT count(*) FROM daily_pairwise"
                    ).fetchone()[0],
                    3,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "sample_scope=all-implementations",
                ):
                    persist_daily_pairwise(
                        connection,
                        date(2025, 1, 1),
                        summaries,
                        sample_scope="all-implementations",
                    )


if __name__ == "__main__":
    unittest.main()
