from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


from liljegren_analysis import CampaignResults


def _write_result(
    campaign: Path,
    implementation: str,
    value: float,
    result_date: str = "2025-01-01",
    run_id: str = "0001",
) -> None:
    path = (
        campaign
        / "results"
        / result_date
        / run_id
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

    def test_inclusive_date_range_prunes_by_exact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            campaign = Path(temporary_directory) / "campaign"
            _write_result(campaign, "heatstress", 30.0, "2025-01-01")
            _write_result(campaign, "heatstress", 31.0, "2025-01-02")
            _write_result(campaign, "heatstress", 99.0, "2025-01-03")

            relation = CampaignResults.from_campaign(campaign).scan(
                start_date="2025-01-01",
                end_date="2025-01-02",
            )

            self.assertEqual(
                relation.aggregate("count(*), avg(value_c)").fetchone(),
                (2, 30.5),
            )

            explicit = CampaignResults.from_campaign(campaign).scan(
                dates=["2025-01-01", "2025-01-03", "2025-01-03"]
            )
            self.assertEqual(
                explicit.aggregate("count(*), avg(value_c)").fetchone(),
                (2, 64.5),
            )

    def test_multiple_scopes_and_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            campaign = Path(temporary_directory) / "campaign"
            for run_id in ("0001", "0002"):
                _write_result(campaign, "heatstress", 30.0, run_id=run_id)
                _write_result(campaign, "meteor", 32.0, run_id=run_id)
                _write_result(campaign, "pywbgt", 99.0, run_id=run_id)

            relation = CampaignResults.from_campaign(campaign).scan(
                run_ids=[1, 2, 2],
                implementations=["heatstress", "meteor"],
                include_filename=True,
            )
            rows = relation.project(
                "run_id, implementation, filename"
            ).order("run_id, implementation").fetchall()

            self.assertEqual(
                [(run_id, implementation) for run_id, implementation, _ in rows],
                [
                    ("0001", "heatstress"),
                    ("0001", "meteor"),
                    ("0002", "heatstress"),
                    ("0002", "meteor"),
                ],
            )
            self.assertTrue(all(filename.endswith("results.parquet") for _, _, filename in rows))

    def test_date_scope_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            campaign = Path(temporary_directory) / "campaign"
            _write_result(campaign, "heatstress", 30.0)
            results = CampaignResults.from_campaign(campaign)

            with self.assertRaisesRegex(ValueError, "only one"):
                results.scan(
                    result_date="2025-01-01",
                    start_date="2025-01-01",
                    end_date="2025-01-02",
                )
            with self.assertRaisesRegex(ValueError, "provided together"):
                results.scan(start_date="2025-01-01")
            with self.assertRaisesRegex(ValueError, "must not be after"):
                results.scan(start_date="2025-01-02", end_date="2025-01-01")
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                results.scan(run_id=1, run_ids=[1, 2])
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                results.scan(implementations=[])
            with self.assertRaisesRegex(TypeError, "not a string"):
                results.scan(dates="2025-01-01")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
