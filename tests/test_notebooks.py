"""Execute notebook front ends with small data and inspect their results."""

import json
from datetime import date
from pathlib import Path

import IPython.display
import pandas as pd
import plotly.graph_objects as go
import pytest
import yaml
from sample_settings import DEFAULT_IMPLEMENTATIONS, DEFAULT_PROBABILITIES
from test_pairwise import _write_results

from analysis import config

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def code_cells(name):
    notebook = json.loads((NOTEBOOKS / name).read_text())
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def execute(name, *, after_setup=None):
    namespace = {"__name__": "__main__"}
    for index, code in enumerate(code_cells(name)):
        exec(compile(code, name, "exec"), namespace)
        if index == 0 and after_setup:
            after_setup(namespace)
    return namespace


@pytest.mark.parametrize(
    "path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.stem
)
def test_notebook_setup_executes(path, monkeypatch):
    original = config.load_settings
    monkeypatch.setattr(
        config,
        "load_settings",
        lambda name: original(name, NOTEBOOKS.parent / "config.example.yaml"),
    )
    exec(compile(code_cells(path.name)[0], path.name, "exec"), {})


def test_daily_and_yearly_notebooks_run_export_and_record_overrides(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    run = tmp_path / "run"
    for name, values in (("alpha", [25.0, 30.0]), ("beta", [26.0, 32.0])):
        _write_results(run, name, values)
    shared = dict(
        start_date="2025-01-01",
        end_date="2025-01-01",
        implementations=["alpha", "beta"],
    )
    settings = {
        "daily_pairwise": dict(
            **shared, hsb_run_paths=[str(run)], output_root="daily", workers=2
        ),
        "yearly_pairwise": dict(
            start_date=shared["start_date"],
            end_date=shared["end_date"],
            input_root="daily",
            output_root="yearly",
            reference_implementation="alpha",
            export_implementations=["beta"],
        ),
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(settings))
    execute(
        "10-daily-pairwise.ipynb",
        after_setup=lambda ns: ns["SETTINGS"].update(workers=1),
    )
    assert json.loads((tmp_path / "daily/settings.json").read_text())["workers"] == 1
    figures = []
    monkeypatch.setattr(IPython.display, "display", lambda obj: figures.append(obj))
    namespace = execute("30-yearly-pairwise.ipynb")
    matrix = namespace["current_matrix"]("wbgt", "mean_difference_c")
    assert matrix.loc["alpha", "beta"] == -1.5
    namespace["export_current_matrix"](None)
    exported = pd.read_csv(
        tmp_path / "yearly/pairwise_matrix_wbgt_mean_difference_c.csv", index_col=0
    )
    assert exported.loc["alpha", "beta"] == -1.5
    assert any(isinstance(item, go.Figure) for item in figures)


def test_quantile_and_map_notebooks_execute_with_small_grid(tmp_path, monkeypatch):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_joint_cellwise_temperature_pipeline import _CELLS, _write_joint_day

    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    run = tmp_path / "run"
    _write_joint_day(run, "2025-01-01", 1)
    population = tmp_path / "population.parquet"
    pq.write_table(
        pa.table({"source_cell_id": _CELLS, "population": [10, 20, 30]}), population
    )
    settings = {
        "quantiles": dict(
            implementations=list(DEFAULT_IMPLEMENTATIONS),
            probabilities=list(DEFAULT_PROBABILITIES),
            hsb_run_paths=[str(run)],
            start_date="2025-01-01",
            end_date="2025-01-01",
            output_root="quantiles",
            date_workers=1,
            replay_workers=1,
            exact_validation_cells=3,
        ),
        "quantile_spread": dict(
            input_root="quantiles",
            output_root="spread",
            quantities=["wbgt"],
            population_weights_path=str(population),
        ),
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(settings))
    execute("11-cellwise-temperature-quantiles.ipynb")
    figures = []
    monkeypatch.setattr(go.Figure, "show", lambda self, **kwargs: figures.append(self))
    namespace = execute("31-cellwise-quantile-spread.ipynb")
    namespace["show_pairwise_map"]("heatstress", "argonne-liljegren-c", "p50")
    assert figures
    colors = np.asarray(figures[-1].data[0].marker.color)
    assert np.all(colors[np.isfinite(colors)] > 0)
    assert (tmp_path / "spread/wbgt/settings.json").is_file()


def test_country_classification_notebook_executes(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    from iso7243_fixtures import write_day
    from test_country_extract import _boundaries

    from analysis.country.extraction import extract_countries_observations

    run = tmp_path / "run"
    write_day(run)
    boundary = tmp_path / "boundary.geojson"
    _boundaries(boundary)
    # The fixture country intersects the cell at 0.25N, 0E after this shift.
    document = json.loads(boundary.read_text())
    for feature in document["features"]:
        for ring in feature["geometry"]["coordinates"]:
            for point in ring:
                point[0] -= 0.13
    boundary.write_text(json.dumps(document))
    extract_countries_observations(
        ("XX",),
        (run,),
        date(2025, 1, 1),
        date(2025, 1, 1),
        tmp_path / "country",
        boundaries=boundary,
    )
    cache = tmp_path / "country/xxx/observations.parquet"
    assert pq.read_table(cache).num_rows > 0
    settings = {
        "country_classification": dict(
            implementations=list(DEFAULT_IMPLEMENTATIONS),
            cache_path=str(cache),
            start_date="2025-01-01",
            end_date="2025-01-01",
            thresholds_c=[25, 30],
            output_root="classification",
        )
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    execute("20-era5-jpn-classification-changes.ipynb")
    assert (tmp_path / "classification/monthly_pair_transition_counts.csv").is_file()
    assert (tmp_path / "classification/settings.json").is_file()
