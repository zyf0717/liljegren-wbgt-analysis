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
        self.assertIn("pairwise_product_root", code)
        self.assertIn("DEFAULT_PAIRWISE_OUTPUT_ROOT", code)
        self.assertIn("combinations(implementations, 2)", code)
        self.assertNotIn("EXPECTED_IMPLEMENTATION", code)
        self.assertNotIn("duckdb", code.casefold())

    def test_daily_pairwise_notebook_resumes_complete_case_product(self) -> None:
        code = _code("10-daily-pairwise.ipynb")

        self.assertIn('sample_scope = "all-implementations"', code)
        self.assertIn("completed_pairwise_dates", code)
        self.assertIn("pending_dates", code)
        self.assertIn("replace_existing_output = False", code)

    def test_joint_cellwise_notebook_uses_parallel_safe_defaults(self) -> None:
        code = _code("11-cellwise-temperature-quantiles.ipynb")

        self.assertIn("DATE_WORKERS = 4", code)
        self.assertIn("REPLAY_WORKERS = 2", code)
        self.assertIn("REPLACE_EXISTING_OUTPUT = False", code)
        self.assertIn("'kong-huber-pywbgt'", code)

    def test_interactive_global_map_uses_equal_earth(self) -> None:
        code = _code("31-cellwise-quantile-spread.ipynb")

        self.assertIn("equal_earth_coordinates", code)
        self.assertIn("go.Scattergl", code)
        self.assertNotIn("x=longitude", code)

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
