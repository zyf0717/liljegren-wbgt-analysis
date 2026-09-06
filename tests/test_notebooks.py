from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parents[1] / "notebooks"
WORKFLOW_NOTEBOOKS = {
    "10-daily-pairwise.ipynb": "run_daily_pairwise",
    "11-cellwise-temperature-quantiles.ipynb": "compute_joint_cellwise_temperature_quantiles",
    "20-era5-jpn-classification-changes.ipynb": "analyze_era5_country_cache",
    "21-nasa-power-sg-classification-changes.ipynb": "analyze_nasa_power_run",
    "30-yearly-pairwise.ipynb": "analyze_yearly_pairwise",
    "31-cellwise-quantile-spread.ipynb": "analyze_cellwise_quantile_spread",
}


def _code(name: str) -> str:
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebooks_are_valid_python() -> None:
    for path in NOTEBOOKS.glob("*.ipynb"):
        compile(_code(path.name), path.name, "exec")


def test_workflow_notebooks_use_canonical_apis_and_explicit_configuration() -> None:
    for name, entrypoint in WORKFLOW_NOTEBOOKS.items():
        code = _code(name)
        tree = ast.parse(code)
        configured = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id.isupper()
        }
        assert entrypoint in code
        assert {"REPOSITORY", "OUTPUT_ROOT"} <= configured
        assert "from liljegren_analysis import" not in code


def test_notebooks_do_not_embed_pipeline_or_annual_aggregation() -> None:
    daily = _code("10-daily-pairwise.ipynb")
    yearly = _code("30-yearly-pairwise.ipynb")
    assert "ProcessPoolExecutor" not in daily
    assert "completed_pairwise_dates" not in daily
    assert "pooled_sufficient_statistics" not in yearly
    assert "abs_histogram.parquet" not in yearly


def test_visualization_notebooks_keep_interactive_equal_earth_views() -> None:
    yearly = _code("30-yearly-pairwise.ipynb")
    cellwise = _code("31-cellwise-quantile-spread.ipynb")
    assert "go.Heatmap" in yearly
    assert "widgets.interactive_output" in yearly
    assert "equal_earth_coordinates" in cellwise
    assert "go.Scattergl" in cellwise


def test_import_smoke_notebook_tracks_only_used_dependencies() -> None:
    code = _code("00-test-imports.ipynb")
    assert "scipy" not in code
    assert "sklearn" not in code
