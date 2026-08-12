from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from liljegren_analysis import (  # noqa: E402
    compute_daily_pairwise,
    persist_daily_pairwise,
)


RESULT_COLUMNS = [
    "result_date",
    "implementation_a",
    "implementation_b",
    "n",
    "mean_difference_c",
    "variance_difference",
    "stddev_difference_c",
    "mean_absolute_difference_c",
    "rmse_c",
    "min_difference_c",
    "max_difference_c",
    "max_absolute_difference_c",
    "p05_difference_c",
    "p25_difference_c",
    "median_difference_c",
    "p75_difference_c",
    "p95_difference_c",
    "median_absolute_difference_c",
    "p95_absolute_difference_c",
]


def _write_results(
    campaign: Path,
    implementation: str,
    values: list[float | None],
    *,
    statuses: list[str] | None = None,
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
                "cycle_ordinal": pa.array([1, 1, 2], type=pa.uint32()),
                "row_id": ["row-1", "row-2", "row-3"],
                "row_ordinal": pa.array(range(count), type=pa.uint32()),
                "status": statuses or ["ok"] * count,
                "value_c": values,
            }
        ),
        path,
    )


class PairwiseTests(unittest.TestCase):
    def _compute(self, root: Path) -> pa.Table:
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

        result_date, result, worker_pid, compute_seconds = compute_daily_pairwise(
            date(2025, 1, 1),
            (first, second),
            threads=1,
        )
        self.assertEqual(result_date, date(2025, 1, 1))
        self.assertEqual(worker_pid, os.getpid())
        self.assertGreaterEqual(compute_seconds, 0.0)
        return result

    def test_compute_filters_and_aggregates_all_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self._compute(Path(temporary_directory))

        self.assertEqual(result.column_names, RESULT_COLUMNS)
        rows = {
            (row["implementation_a"], row["implementation_b"]): row
            for row in result.to_pylist()
        }
        self.assertEqual(
            list(rows),
            [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")],
        )

        self.assertEqual(rows[("alpha", "beta")]["n"], 1)
        self.assertEqual(rows[("alpha", "beta")]["mean_difference_c"], 2.0)

        alpha_gamma = rows[("alpha", "gamma")]
        self.assertEqual(alpha_gamma["n"], 2)
        self.assertEqual(alpha_gamma["mean_difference_c"], 4.0)
        self.assertEqual(alpha_gamma["variance_difference"], 2.0)
        self.assertEqual(alpha_gamma["mean_absolute_difference_c"], 4.0)
        self.assertAlmostEqual(alpha_gamma["rmse_c"], (17.0) ** 0.5)
        self.assertEqual(alpha_gamma["min_difference_c"], 3.0)
        self.assertEqual(alpha_gamma["max_difference_c"], 5.0)
        self.assertEqual(alpha_gamma["max_absolute_difference_c"], 5.0)

        beta_gamma = rows[("beta", "gamma")]
        self.assertEqual(beta_gamma["n"], 2)
        self.assertEqual(beta_gamma["mean_difference_c"], 2.0)
        self.assertEqual(beta_gamma["min_difference_c"], 1.0)
        self.assertEqual(beta_gamma["max_difference_c"], 3.0)

    def test_compute_validates_worker_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "runs is empty"):
            compute_daily_pairwise(date(2025, 1, 1), (), threads=1)
        with self.assertRaisesRegex(ValueError, "threads must be positive"):
            compute_daily_pairwise(date(2025, 1, 1), (Path("unused"),), threads=0)

    def test_persistence_is_idempotent_and_rolls_back_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = self._compute(root)
            database = root / "pairwise.duckdb"

            with duckdb.connect(str(database)) as connection:
                persist_daily_pairwise(connection, date(2025, 1, 1), result)
                persist_daily_pairwise(connection, date(2025, 1, 1), result)
                self.assertEqual(
                    connection.execute(
                        "SELECT count(*) FROM daily_pairwise"
                    ).fetchone()[0],
                    3,
                )

                invalid = pa.table(
                    {"result_date": pa.array([date(2025, 1, 1)])}
                )
                with self.assertRaises(duckdb.Error):
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
                persist_daily_pairwise(connection, date(2025, 1, 1), result)


if __name__ == "__main__":
    unittest.main()
