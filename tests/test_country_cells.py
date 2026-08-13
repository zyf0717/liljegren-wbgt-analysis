from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from liljegren_analysis.country.cells import Country, country_cells, load_countries
from shapely.geometry import box


class CountryCellTests(unittest.TestCase):
    def test_selects_positive_area_cell_intersections(self) -> None:
        country = Country(
            iso_a2="XX",
            iso_a3="XXX",
            name="Example",
            geometry=box(0.13, 0.13, 0.14, 0.14),
        )

        cells = country_cells(country)

        self.assertEqual([cell.cell_id for cell in cells], ["era5-global-n002500-e0002500"])
        self.assertFalse(cells[0].center_in_country)
        self.assertEqual(cells[0].row_id_prefix, "era5-global-n002500-e0002500-")

    def test_excludes_zero_area_boundary_touches(self) -> None:
        country = Country(
            iso_a2="XX",
            iso_a3="XXX",
            name="Example",
            geometry=box(-0.125, -0.125, 0.125, 0.125),
        )

        cells = country_cells(country)

        self.assertEqual([cell.cell_id for cell in cells], ["era5-global-n000000-e0000000"])
        self.assertTrue(cells[0].center_in_country)

    def test_loads_alpha_2_and_alpha_3_aliases_without_duplicates(self) -> None:
        document = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "ADMIN": "Singapore",
                        "ISO_A2": "SG",
                        "ISO_A2_EH": "SG",
                        "ISO_A3": "SGP",
                        "ISO_A3_EH": "SGP",
                        "ADM0_A3": "SGP",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[103.0, 1.0], [104.0, 1.0], [104.0, 2.0], [103.0, 2.0], [103.0, 1.0]]],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "countries.geojson"
            path.write_text(json.dumps(document), encoding="utf-8")

            countries = load_countries(path, ["sg", "SGP"])

        self.assertEqual(len(countries), 1)
        self.assertEqual(countries[0].iso_a3, "SGP")

    def test_rejects_unknown_country_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "countries.geojson"
            path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ZZ"):
                load_countries(path, ["ZZ"])


if __name__ == "__main__":
    unittest.main()
