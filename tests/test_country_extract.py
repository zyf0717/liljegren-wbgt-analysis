from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis import extract_country_observations


def _write_boundaries(path: Path) -> None:
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "ADMIN": "Example",
                    "ISO_A2": "XX",
                    "ISO_A2_EH": "XX",
                    "ISO_A3": "XXX",
                    "ISO_A3_EH": "XXX",
                    "ADM0_A3": "XXX",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.13, 0.13],
                            [0.14, 0.13],
                            [0.14, 0.14],
                            [0.13, 0.14],
                            [0.13, 0.13],
                        ]
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_results(campaign: Path) -> None:
    path = (
        campaign
        / "results"
        / "2025-01-01"
        / "0001"
        / "alpha"
        / "results.parquet"
    )
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "cycle_ordinal": pa.array([1, 1, 1], type=pa.uint32()),
                "row_id": [
                    "era5-global-n002500-e0002500-2025010100",
                    "era5-global-n002500-e0002500-2025010101",
                    "era5-global-n002500-e0005000-2025010100",
                ],
                "row_ordinal": pa.array([1, 2, 3], type=pa.uint32()),
                "status": ["ok", "error", "ok"],
                "value_c": [31.0, None, 99.0],
            }
        ),
        path,
    )


class CountryExtractionTests(unittest.TestCase):
    def test_extracts_valid_country_rows_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            campaign = root / "campaign"
            boundaries = root / "boundaries.geojson"
            output = root / "data" / "country" / "xxx" / "observations.parquet"
            _write_boundaries(boundaries)
            _write_results(campaign)

            extraction = extract_country_observations(
                "xx",
                (campaign,),
                date(2025, 1, 1),
                date(2025, 1, 1),
                output,
                boundaries=boundaries,
            )

            with duckdb.connect() as connection:
                rows = connection.execute(
                    "SELECT implementation, cell_id, value_c FROM read_parquet(?)",
                    [str(output)],
                ).fetchall()
            manifest = json.loads(
                output.with_name("manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(
                rows,
                [("alpha", "era5-global-n002500-e0002500", 31.0)],
            )
            self.assertEqual(extraction.country_iso_a3, "XXX")
            self.assertEqual(extraction.observation_count, 1)
            self.assertEqual(manifest["observation_count"], 1)
            self.assertEqual(manifest["parquet_file"], "observations.parquet")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                extract_country_observations(
                    "XX",
                    (campaign,),
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                    output,
                    boundaries=boundaries,
                )

    def test_rejects_negative_threads(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            extract_country_observations(
                "XX",
                (Path("unused"),),
                date(2025, 1, 1),
                date(2025, 1, 1),
                Path("unused.parquet"),
                threads=-1,
            )


if __name__ == "__main__":
    unittest.main()
