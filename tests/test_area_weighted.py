from __future__ import annotations

import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.comparison.pairwise import compute_daily_pairwise
from analysis.grid import LATITUDE_WEIGHT_TABLE
from analysis.results import discover_results


def _row_id(latitude: float, hour: int) -> str:
    hemisphere = "n" if latitude >= 0 else "s"
    magnitude = round(abs(latitude) * 10000)
    return f"era5-global-{hemisphere}{magnitude:06d}-e0000000-20250101{hour:02d}"


def _weight(latitude: float) -> float:
    half_spacing = math.radians(0.125)
    lower = math.radians(max(-90.0, latitude - 0.125))
    upper = math.radians(min(90.0, latitude + 0.125))
    return (math.sin(upper) - math.sin(lower)) / (2 * math.sin(half_spacing))


def _write(
    hsb_run: Path, implementation: str, row_ids: list[str], values: list[float]
) -> None:
    path = (
        hsb_run / "results" / "2025-01-01" / "0001" / implementation / "results.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(values)
    pq.write_table(
        pa.table(
            {
                "cycle_ordinal": pa.array([1] * count, type=pa.uint32()),
                "row_id": row_ids,
                "row_ordinal": pa.array(range(count), type=pa.uint32()),
                "wbgt_status": ["ok"] * count,
                "wbgt_c": values,
                "tg_status": ["ok"] * count,
                "tg_c": values,
                "tnwb_status": ["ok"] * count,
                "tnwb_c": values,
            }
        ),
        path,
    )


class AreaWeightedTests(unittest.TestCase):
    def test_precomputed_weights_cover_era5_grid(self) -> None:
        weights = dict(
            zip(
                LATITUDE_WEIGHT_TABLE["latitude_code"].to_pylist(),
                LATITUDE_WEIGHT_TABLE["area_weight"].to_pylist(),
                strict=True,
            )
        )
        self.assertEqual(len(weights), 721)
        self.assertAlmostEqual(weights["n000000"], 1.0)
        self.assertAlmostEqual(weights["n600000"], math.cos(math.radians(60)))
        self.assertEqual(weights["n600000"], weights["s600000"])
        self.assertGreater(weights["n900000"], 0.0)

    def test_single_pass_area_sums_match_direct_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latitudes = [0.0, 60.0, -60.0, -90.0]
            row_ids = [_row_id(value, index) for index, value in enumerate(latitudes)]
            left = [2.0, 4.0, 6.0, 8.0]
            right = [1.0, 1.0, 3.0, 4.0]
            _write(root, "alpha", row_ids, left)
            _write(root, "beta", row_ids, right)
            _, summaries = (
                date(2025, 1, 1),
                compute_daily_pairwise(
                    date(2025, 1, 1),
                    discover_results((root,), (date(2025, 1, 1),)),
                    implementations=("alpha", "beta"),
                ),
            )
        row = next(
            (
                item
                for item in summaries.pairwise.to_pylist()
                if item["quantity"] == "wbgt"
            )
        )
        differences = [x - y for x, y in zip(left, right, strict=True)]
        weights = [_weight(latitude) for latitude in latitudes]
        self.assertEqual(row["n"], 4)
        self.assertAlmostEqual(row["area_weight_sum"], sum(weights))
        self.assertAlmostEqual(
            row["area_weighted_difference_sum_c"],
            sum((w * d for w, d in zip(weights, differences, strict=True))),
        )
        self.assertAlmostEqual(
            row["area_weighted_squared_difference_sum_c2"],
            sum((w * d * d for w, d in zip(weights, differences, strict=True))),
        )
        self.assertAlmostEqual(
            row["area_weighted_absolute_difference_sum_c"],
            sum((w * abs(d) for w, d in zip(weights, differences, strict=True))),
        )


if __name__ == "__main__":
    unittest.main()
