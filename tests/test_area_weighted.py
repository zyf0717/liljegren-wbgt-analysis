from __future__ import annotations

import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis import (
    compute_daily_area_weighted_pairwise,
    persist_daily_area_weighted_pairwise,
)


RESULT_COLUMNS = [
    "result_date",
    "implementation_a",
    "implementation_b",
    "n",
    "area_weight_sum",
    "area_weighted_mean_difference_c",
    "area_weighted_variance_difference",
    "area_weighted_stddev_difference_c",
    "area_weighted_mean_absolute_difference_c",
    "area_weighted_rmse_c",
]


def _row_id(latitude: float, hour: int) -> str:
    hemisphere = "n" if latitude >= 0 else "s"
    magnitude = round(abs(latitude) * 10_000)
    return (
        f"era5-global-{hemisphere}{magnitude:06d}-"
        f"e0000000-20250101{hour:02d}"
    )


def _area_weight(latitude: float) -> float:
    half_spacing = math.radians(0.125)
    lower = math.radians(max(-90.0, latitude - 0.125))
    upper = math.radians(min(90.0, latitude + 0.125))
    return (math.sin(upper) - math.sin(lower)) / (
        2.0 * math.sin(half_spacing)
    )


def _write_results(
    campaign: Path,
    implementation: str,
    row_ids: list[str],
    values: list[float],
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
    count = len(row_ids)
    pq.write_table(
        pa.table(
            {
                "cycle_ordinal": pa.array([1] * count, type=pa.uint32()),
                "row_id": row_ids,
                "row_ordinal": pa.array(range(count), type=pa.uint32()),
                "status": ["ok"] * count,
                "value_c": values,
            }
        ),
        path,
    )


class AreaWeightedPairwiseTests(unittest.TestCase):
    def test_compute_matches_direct_weighted_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"
            latitudes = [0.0, 60.0, -60.0, -90.0]
            row_ids = [
                _row_id(latitude, hour)
                for hour, latitude in enumerate(latitudes)
            ]
            left = [2.0, 4.0, 6.0, 8.0]
            right = [1.0, 1.0, 3.0, 4.0]
            _write_results(first, "alpha", row_ids, left)
            _write_results(second, "beta", row_ids, right)

            _, result, _, _ = compute_daily_area_weighted_pairwise(
                date(2025, 1, 1),
                (first, second),
                threads=1,
            )

        self.assertEqual(result.column_names, RESULT_COLUMNS)
        row = result.to_pylist()[0]
        differences = [x - y for x, y in zip(left, right, strict=True)]
        weights = [_area_weight(latitude) for latitude in latitudes]
        total_weight = sum(weights)
        mean = sum(
            weight * difference
            for weight, difference in zip(weights, differences, strict=True)
        ) / total_weight
        variance = sum(
            weight * (difference - mean) ** 2
            for weight, difference in zip(weights, differences, strict=True)
        ) / total_weight
        mean_absolute = sum(
            weight * abs(difference)
            for weight, difference in zip(weights, differences, strict=True)
        ) / total_weight
        rmse = math.sqrt(
            sum(
                weight * difference**2
                for weight, difference in zip(weights, differences, strict=True)
            )
            / total_weight
        )

        self.assertEqual(row["n"], 4)
        self.assertAlmostEqual(row["area_weight_sum"], total_weight)
        self.assertAlmostEqual(
            row["area_weighted_mean_difference_c"],
            mean,
        )
        self.assertAlmostEqual(
            row["area_weighted_variance_difference"],
            variance,
        )
        self.assertAlmostEqual(
            row["area_weighted_stddev_difference_c"],
            math.sqrt(variance),
        )
        self.assertAlmostEqual(
            row["area_weighted_mean_absolute_difference_c"],
            mean_absolute,
        )
        self.assertAlmostEqual(row["area_weighted_rmse_c"], rmse)
        self.assertAlmostEqual(_area_weight(60.0), _area_weight(-60.0))
        self.assertAlmostEqual(_area_weight(0.0), 1.0)
        self.assertGreater(_area_weight(-90.0), 0.0)

    def test_compute_rejects_non_era5_row_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"
            _write_results(first, "alpha", ["row-1"], [2.0])
            _write_results(second, "beta", ["row-1"], [1.0])

            with self.assertRaisesRegex(duckdb.Error, "Invalid ERA5 row_id"):
                compute_daily_area_weighted_pairwise(
                    date(2025, 1, 1),
                    (first, second),
                    threads=1,
                )

    def test_persistence_is_versioned_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"
            row_ids = [_row_id(0.0, 0), _row_id(60.0, 1)]
            _write_results(first, "alpha", row_ids, [2.0, 4.0])
            _write_results(second, "beta", row_ids, [1.0, 1.0])
            _, result, _, _ = compute_daily_area_weighted_pairwise(
                date(2025, 1, 1),
                (first, second),
                threads=1,
            )

            with duckdb.connect() as connection:
                persist_daily_area_weighted_pairwise(
                    connection,
                    date(2025, 1, 1),
                    result,
                )
                persist_daily_area_weighted_pairwise(
                    connection,
                    date(2025, 1, 1),
                    result,
                )

                self.assertEqual(
                    connection.execute(
                        "SELECT count(*) FROM daily_pairwise_area_weighted"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT table_name, schema_version FROM pairwise_schema"
                    ).fetchall(),
                    [("daily_pairwise_area_weighted", 1)],
                )


if __name__ == "__main__":
    unittest.main()
