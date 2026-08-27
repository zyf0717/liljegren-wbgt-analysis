from __future__ import annotations

import json
import unittest
from pathlib import Path


NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


def _code(name: str) -> str:
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


class NotebookMigrationTests(unittest.TestCase):
    def test_migrated_notebooks_are_valid_python(self) -> None:
        for name in (
            "10-daily_pairwise.ipynb",
            "11-era5_country_classification_changes.ipynb",
            "12-nasa_power_classification_changes.ipynb",
            "20-yearly_pairwise.ipynb",
        ):
            with self.subTest(notebook=name):
                compile(_code(name), name, "exec")

    def test_era5_country_classification_uses_native_wbgt_columns(self) -> None:
        code = _code("11-era5_country_classification_changes.ipynb")

        self.assertIn("WHERE wbgt_status = 'ok' AND wbgt_c IS NOT NULL", code)
        self.assertIn("a.wbgt_c AS wbgt_c_a", code)
        self.assertIn("AS wbgt_difference_c", code)
        self.assertNotIn("value_c", code)
        self.assertNotIn(" AS difference_c", code)

    def test_nasa_power_classification_uses_native_wbgt_columns(self) -> None:
        code = _code("12-nasa_power_classification_changes.ipynb")

        self.assertIn('results["wbgt_status"].eq("ok")', code)
        self.assertIn('classified["wbgt_c"]', code)
        self.assertIn('paired_classifications["wbgt_c_a"]', code)
        self.assertIn('paired_classifications["wbgt_difference_c"]', code)
        self.assertNotIn("value_c", code)
        self.assertNotIn('results["status"]', code)


if __name__ == "__main__":
    unittest.main()
