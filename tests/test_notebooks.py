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
            "10-daily-pairwise.ipynb",
            "11-cellwise-temperature-quantiles.ipynb",
            "20-era5-jpn-classification-changes.ipynb",
            "21-nasa-power-sg-classification-changes.ipynb",
            "30-yearly-pairwise.ipynb",
            "31-cellwise-quantile-spread.ipynb",
        ):
            with self.subTest(notebook=name):
                compile(_code(name), name, "exec")

    def test_era5_country_classification_uses_shared_cache_analysis(self) -> None:
        code = _code("20-era5-jpn-classification-changes.ipynb")

        self.assertIn("analyze_era5_country_cache", code)
        self.assertIn("observations.parquet", code)
        self.assertNotIn("country_cells(", code)

    def test_nasa_power_classification_uses_shared_analysis(self) -> None:
        code = _code("21-nasa-power-sg-classification-changes.ipynb")

        self.assertIn("analyze_nasa_power_run", code)
        self.assertIn("write_outputs", code)

    def test_yearly_pairwise_reads_partitioned_parquet(self) -> None:
        code = _code("30-yearly-pairwise.ipynb")

        self.assertIn("pyarrow.parquet", code)
        self.assertIn("pairwise_root", code)
        self.assertNotIn("duckdb", code.casefold())

    def test_era5_notebooks_use_current_run_contract(self) -> None:
        code = "\n".join(
            _code(name)
            for name in (
                "10-daily-pairwise.ipynb",
                "11-cellwise-temperature-quantiles.ipynb",
                "30-yearly-pairwise.ipynb",
                "31-cellwise-quantile-spread.ipynb",
            )
        )
        self.assertIn("wbgt-comparison-20260902-150355", code)
        for obsolete in (
            "wbgt-comparison-20260823-181813",
            "hsb-wbgt-comparison-results-csv-v3",
            "'liljegren-c'",
            "'pywbgt'",
        ):
            self.assertNotIn(obsolete, code)


if __name__ == "__main__":
    unittest.main()
