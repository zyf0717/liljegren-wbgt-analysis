from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from liljegren_analysis import CampaignResults  # noqa: E402


def _write_result(campaign: Path, implementation: str, value: float) -> None:
    path = (
        campaign
        / "results"
        / "2025-01-01"
        / "0001"
        / implementation
        / "results.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "cycle_ordinal": pa.array([1], type=pa.uint32()),
                "row_id": ["era5-global-n012500-e1037500-2025010100"],
                "row_ordinal": pa.array([1], type=pa.uint32()),
                "status": ["ok"],
                "value_c": [value],
            }
        ),
        path,
    )


class CampaignResultsTests(unittest.TestCase):
    def test_shared_relations_can_be_queried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"
            _write_result(first, "heatstress", 30.0)
            _write_result(second, "meteor", 32.0)

            with duckdb.connect() as connection:
                combined = CampaignResults.from_campaign(
                    first, connection=connection
                ).scan(result_date="2025-01-01").union(
                    CampaignResults.from_campaign(second, connection=connection).scan(
                        result_date="2025-01-01"
                    )
                )
                summary = combined.query(
                    "results",
                    """
                    SELECT implementation, count(*) AS observations, avg(value_c) AS mean_c
                    FROM results
                    WHERE status = 'ok'
                    GROUP BY implementation
                    ORDER BY implementation
                    """,
                ).fetchall()

            self.assertEqual(summary, [("heatstress", 1, 30.0), ("meteor", 1, 32.0)])


if __name__ == "__main__":
    unittest.main()
