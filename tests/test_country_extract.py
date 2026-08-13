from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis import (
    CountryExtractionProgress,
    extract_countries_observations,
    extract_country_observations,
)
from liljegren_analysis.country_extract import _parquet_may_overlap


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


def _write_multiple_boundaries(path: Path) -> None:
    features = []
    for name, iso_a2, iso_a3, longitude in (
        ("Example", "XX", "XXX", 0.13),
        ("Second", "YY", "YYY", 0.38),
        ("Missing", "ZZ", "ZZZ", 1.13),
        ("Overlap", "QQ", "QQQ", 0.13),
    ):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "ADMIN": name,
                    "ISO_A2": iso_a2,
                    "ISO_A2_EH": iso_a2,
                    "ISO_A3": iso_a3,
                    "ISO_A3_EH": iso_a3,
                    "ADM0_A3": iso_a3,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [longitude, 0.13],
                            [longitude + 0.01, 0.13],
                            [longitude + 0.01, 0.14],
                            [longitude, 0.14],
                            [longitude, 0.13],
                        ]
                    ],
                },
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


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


def _write_irrelevant_results(campaign: Path, *, statistics: bool = True) -> Path:
    path = (
        campaign
        / "results"
        / "2025-01-01"
        / "0002"
        / "alpha"
        / "results.parquet"
    )
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "cycle_ordinal": pa.array([1], type=pa.uint32()),
                "row_id": ["era5-global-s900000-w1800000-2025010100"],
                "row_ordinal": pa.array([1], type=pa.uint32()),
                "status": ["ok"],
                "value_c": [99.0],
            }
        ),
        path,
        write_statistics=statistics,
    )
    return path


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
                columns = [
                    description[0]
                    for description in connection.execute(
                        "SELECT * FROM read_parquet(?) LIMIT 0", [str(output)]
                    ).description
                ]
            manifest = json.loads(
                output.with_name("manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(
                rows,
                [("alpha", "era5-global-n002500-e0002500", 31.0)],
            )
            self.assertEqual(
                columns,
                [
                    "result_date",
                    "run_id",
                    "implementation",
                    "cycle_ordinal",
                    "row_id",
                    "cell_id",
                    "row_ordinal",
                    "value_c",
                ],
            )
            self.assertEqual(extraction.country_iso_a3, "XXX")
            self.assertEqual(extraction.observation_count, 1)
            self.assertEqual(manifest["observation_count"], 1)
            self.assertEqual(manifest["parquet_file"], "observations.parquet")
            status = json.loads(
                output.with_name("extraction-status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["selected_source_files"], 1)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                extract_country_observations(
                    "XX",
                    (campaign,),
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                    output,
                    boundaries=boundaries,
                )

            replacement = extract_country_observations(
                "XX",
                (campaign,),
                date(2025, 1, 1),
                date(2025, 1, 1),
                output,
                boundaries=boundaries,
                overwrite=True,
            )
            self.assertEqual(replacement.observation_count, 1)

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

    def test_extracts_multiple_countries_with_one_pruned_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            campaign = root / "campaign"
            boundaries = root / "boundaries.geojson"
            output_directory = root / "data" / "country"
            _write_multiple_boundaries(boundaries)
            _write_results(campaign)
            _write_irrelevant_results(campaign)
            progress: list[CountryExtractionProgress] = []

            with patch(
                "liljegren_analysis.country_extract._create_results_view",
                wraps=__import__(
                    "liljegren_analysis.country_extract", fromlist=["_create_results_view"]
                )._create_results_view,
            ) as create_results_view:
                extractions = extract_countries_observations(
                    ["xx", "XXX", "yy", "QQ"],
                    (campaign,),
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                    output_directory,
                    boundaries=boundaries,
                    progress=progress.append,
                )

            self.assertEqual(
                [extraction.country_iso_a3 for extraction in extractions],
                ["XXX", "YYY", "QQQ"],
            )
            create_results_view.assert_called_once()
            selected_paths = create_results_view.call_args.args[1]
            self.assertEqual(len(selected_paths), 1)

            with duckdb.connect() as connection:
                first = connection.execute(
                    "SELECT row_id, value_c FROM read_parquet(?)",
                    [str(output_directory / "xxx" / "observations.parquet")],
                ).fetchall()
                second = connection.execute(
                    "SELECT row_id, value_c FROM read_parquet(?)",
                    [str(output_directory / "yyy" / "observations.parquet")],
                ).fetchall()
            self.assertEqual(
                first,
                [("era5-global-n002500-e0002500-2025010100", 31.0)],
            )
            self.assertEqual(
                second,
                [("era5-global-n002500-e0005000-2025010100", 99.0)],
            )
            with duckdb.connect() as connection:
                overlap = connection.execute(
                    "SELECT row_id, value_c FROM read_parquet(?)",
                    [str(output_directory / "qqq" / "observations.parquet")],
                ).fetchall()
            self.assertEqual(overlap, first)
            self.assertEqual(
                [item.phase for item in progress],
                ["discovering", "scanning", "exporting", "publishing", "complete"],
            )
            self.assertEqual(progress[-1].total_source_files, 2)
            self.assertEqual(progress[-1].selected_source_files, 1)

    def test_validates_all_countries_before_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            campaign = root / "campaign"
            boundaries = root / "boundaries.geojson"
            output_directory = root / "data" / "country"
            _write_multiple_boundaries(boundaries)
            _write_results(campaign)

            with self.assertRaisesRegex(ValueError, "Missing"):
                extract_countries_observations(
                    ["XX", "ZZ"],
                    (campaign,),
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                    output_directory,
                    boundaries=boundaries,
                )

            self.assertFalse(
                (output_directory / "xxx" / "observations.parquet").exists()
            )
            status = json.loads(
                (output_directory / "xxx" / "extraction-status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["state"], "failed")

    def test_footer_pruning_falls_back_when_statistics_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            campaign = Path(temporary_directory) / "campaign"
            with_statistics = _write_irrelevant_results(campaign)
            without_statistics = _write_irrelevant_results(
                Path(temporary_directory) / "other", statistics=False
            )
            ranges = (
                (
                    "era5-global-n002500-e0002500-",
                    "era5-global-n002500-e0002500.",
                ),
            )

            self.assertFalse(_parquet_may_overlap(with_statistics, ranges))
            self.assertTrue(_parquet_may_overlap(without_statistics, ranges))

    def test_interruption_records_cancelled_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            campaign = root / "campaign"
            boundaries = root / "boundaries.geojson"
            output_directory = root / "data" / "country"
            _write_boundaries(boundaries)
            _write_results(campaign)

            with patch(
                "liljegren_analysis.country_extract._execute_scan",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    extract_countries_observations(
                        ["XX"],
                        (campaign,),
                        date(2025, 1, 1),
                        date(2025, 1, 1),
                        output_directory,
                        boundaries=boundaries,
                    )

            country_directory = output_directory / "xxx"
            self.assertFalse((country_directory / "observations.parquet").exists())
            status = json.loads(
                (country_directory / "extraction-status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["state"], "cancelled")
            self.assertFalse(any(output_directory.glob(".country-extract-*")))


class CountryExtractionCliTests(unittest.TestCase):
    def test_rejects_custom_output_for_multiple_countries(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "extract_country.py"
        specification = importlib.util.spec_from_file_location(
            "test_extract_country_script", script
        )
        assert specification is not None and specification.loader is not None
        extract_country = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(extract_country)

        with patch.object(
            sys,
            "argv",
            ["extract_country.py", "SG", "JPN", "--output", "output.parquet"],
        ):
            with self.assertRaises(SystemExit):
                extract_country.parse_args()


if __name__ == "__main__":
    unittest.main()
