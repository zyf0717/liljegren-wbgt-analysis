from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.country.extraction import (
    _BufferedParquetWriter,
    _overlapping_row_groups,
    extract_countries_observations,
)


def _boundaries(path: Path) -> None:
    features = []
    for name, a2, a3, longitude in (
        ("Example", "XX", "XXX", 0.13),
        ("Second", "YY", "YYY", 0.38),
        ("Missing", "ZZ", "ZZZ", 1.13),
    ):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "ADMIN": name,
                    "ISO_A2": a2,
                    "ISO_A2_EH": a2,
                    "ISO_A3": a3,
                    "ISO_A3_EH": a3,
                    "ADM0_A3": a3,
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
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def _results(root: Path, *, statistics: bool = True) -> Path:
    path = root / "results/2025-01-01/0001/alpha/results.parquet"
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
                "wbgt_status": ["ok", "error", "ok"],
                "wbgt_c": [31.0, None, 99.0],
                "tg_status": ["ok", "ok", "ok"],
                "tg_c": [30.0, 29.0, 98.0],
                "tnwb_status": ["ok", "non-convergent", "ok"],
                "tnwb_c": [29.0, None, 97.0],
            }
        ),
        path,
        write_statistics=statistics,
    )
    return path


class CountryExtractionTests(unittest.TestCase):
    def test_implementation_selection_excludes_unselected_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundaries = root / "boundaries.geojson"
            _boundaries(boundaries)
            first = _results(root / "hsb")
            names = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
            for name in (*names[1:], "excluded"):
                target = first.parent.parent / name / first.name
                target.parent.mkdir()
                shutil.copyfile(first, target)

            for label, selection, expected in (
                ("selected", names, set(names)),
                ("default", None, {*names, "excluded"}),
            ):
                kwargs = {} if selection is None else {"implementations": selection}
                (extraction,) = extract_countries_observations(
                    ["XX"], (root / "hsb",), date(2025, 1, 1), date(2025, 1, 1),
                    root / label, boundaries=boundaries, **kwargs,
                )
                table = pq.read_table(root / label / "xxx/observations.parquet")
                self.assertEqual(set(table["implementation"].to_pylist()), expected)
                self.assertEqual(table.num_rows, 2 * len(expected))
                self.assertEqual(
                    {scope["implementation"] for scope in extraction.scopes}, expected
                )

    def test_streams_country_rows_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundaries = root / "boundaries.geojson"
            _boundaries(boundaries)
            _results(root / "hsb")
            output = root / "out" / "xxx" / "observations.parquet"
            (extraction,) = extract_countries_observations(
                ["XX"],
                (root / "hsb",),
                date(2025, 1, 1),
                date(2025, 1, 1),
                output.parent.parent,
                boundaries=boundaries,
            )
            table = pq.read_table(output).sort_by("row_id")
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(
                table["cell_id"].to_pylist(), ["era5-global-n002500-e0002500"] * 2
            )
            self.assertEqual(
                extraction.valid_observation_counts, {"wbgt": 1, "tg": 2, "tnwb": 1}
            )
            self.assertEqual(
                json.loads(output.with_name("manifest.json").read_text())[
                    "result_row_count"
                ],
                2,
            )

    def test_clips_country_polygon_to_explicit_bounds_and_records_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundaries = root / "boundaries.geojson"
            _boundaries(boundaries)
            _results(root / "hsb")
            output = root / "out" / "xxx" / "observations.parquet"

            (extraction,) = extract_countries_observations(
                ["XX"],
                (root / "hsb",),
                date(2025, 1, 1),
                date(2025, 1, 1),
                output.parent.parent,
                bounds=(0.12, 0.12, 0.37, 0.15),
                boundaries=boundaries,
            )

            table = pq.read_table(output)
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(
                table["cell_id"].to_pylist(),
                ["era5-global-n002500-e0002500"] * 2,
            )
            self.assertEqual(extraction.selection_bounds, (0.12, 0.12, 0.37, 0.15))
            self.assertEqual(extraction.schema_version, 4)

    def test_routes_one_scan_to_multiple_countries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundaries = root / "boundaries.geojson"
            _boundaries(boundaries)
            _results(root / "hsb")
            extractions = extract_countries_observations(
                ["XX", "YY"],
                (root / "hsb",),
                date(2025, 1, 1),
                date(2025, 1, 1),
                root / "out",
                boundaries=boundaries,
            )
            self.assertEqual(
                [item.country_iso_a3 for item in extractions], ["XXX", "YYY"]
            )
            self.assertEqual([item.result_row_count for item in extractions], [2, 1])

    def test_fails_before_publication_when_any_country_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundaries = root / "boundaries.geojson"
            _boundaries(boundaries)
            _results(root / "hsb")
            with self.assertRaisesRegex(ValueError, "Missing"):
                extract_countries_observations(
                    ["XX", "ZZ"],
                    (root / "hsb",),
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                    root / "out",
                    boundaries=boundaries,
                )
            self.assertFalse((root / "out" / "xxx" / "observations.parquet").exists())

    def test_footer_pruning_falls_back_without_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with_stats = _results(root / "first")
            without_stats = _results(root / "second", statistics=False)
            ranges = (
                ("era5-global-n002500-e0002500-", "era5-global-n002500-e0002500."),
            )
            self.assertTrue(_overlapping_row_groups(with_stats, ranges))
            self.assertTrue(_overlapping_row_groups(without_stats, ranges))

    def test_footer_pruning_selects_only_overlapping_row_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.parquet"
            pq.write_table(
                pa.table({"row_id": ["a", "b", "c"]}),
                path,
                row_group_size=1,
            )

            self.assertEqual(_overlapping_row_groups(path, (("b", "c"),)), (1,))

    def test_buffered_writer_emits_bounded_row_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.parquet"
            table = pa.table(
                {
                    "value": range(600_000),
                    "result_date": [date(2025, 1, 1)] * 600_000,
                    "implementation": ["alpha"] * 600_000,
                    "run_id": ["run"] * 600_000,
                }
            )
            writer = _BufferedParquetWriter(path, table.schema)

            writer.write(table.slice(0, 200_000))
            writer.write(table.slice(200_000))
            writer.finish()

            metadata = pq.read_metadata(path)
            self.assertEqual(metadata.num_row_groups, 3)
            self.assertEqual(
                [metadata.row_group(index).num_rows for index in range(3)],
                [250_000, 250_000, 100_000],
            )


if __name__ == "__main__":
    unittest.main()


def test_buffered_routing_tracks_scopes_across_group_boundaries(tmp_path, monkeypatch):
    from analysis.country import extraction

    monkeypatch.setattr(extraction, "_OUTPUT_ROW_GROUP_SIZE", 4)
    path = tmp_path / "cache.parquet"

    def rows(day, name, count):
        return pa.table(
            {
                "result_date": [day] * count,
                "implementation": [name] * count,
                "run_id": ["run"] * count,
                "value": range(count),
            }
        )

    first = rows(date(2025, 1, 31), "a", 3)
    second = rows(date(2025, 2, 1), "b", 6)
    writer = _BufferedParquetWriter(path, first.schema)
    writer.write(first)
    writer.write(second)
    writer.finish()
    assert writer.routing() == [
        {
            "result_date": "2025-01-31",
            "implementation": "a",
            "run_id": "run",
            "row_groups": [0],
        },
        {
            "result_date": "2025-02-01",
            "implementation": "b",
            "run_id": "run",
            "row_groups": [0, 1, 2],
        },
    ]
    reader = pq.ParquetFile(path)
    assert [reader.metadata.row_group(i).num_rows for i in range(3)] == [4, 4, 1]
