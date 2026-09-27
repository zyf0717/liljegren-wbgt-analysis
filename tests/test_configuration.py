"""Configuration and real script execution on small local data."""

import inspect
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest
import yaml
from test_pairwise import _write_results

from analysis import config
from analysis.io.artifacts import save_settings
from scripts import daily_pairwise

REPOSITORY = Path(__file__).resolve().parents[1]


def test_missing_configuration_and_section(tmp_path):
    with pytest.raises(FileNotFoundError, match="Copy config.example.yaml"):
        config.load_settings("daily_pairwise", tmp_path / "absent.yaml")
    path = tmp_path / "config.yaml"
    path.write_text("other: {}\n")
    with pytest.raises(ValueError, match="daily_pairwise"):
        config.load_settings("daily_pairwise", path)


def test_yaml_anchors_paths_dates_and_independent_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text("""shared: &shared
  start_date: 2025-01-01
  end_date: '2025-01-02'
  hsb_run_paths: [inputs/run]
  output_root: data/output
  thresholds_c: [25, 30]
first:
  <<: *shared
second:
  <<: *shared
""")
    first = config.load_settings("first")
    second = config.load_settings("second")
    assert first["start_date"] == date(2025, 1, 1)
    assert first["end_date"] == date(2025, 1, 2)
    assert first["hsb_run_paths"] == (tmp_path / "inputs/run",)
    assert first["output_root"] == tmp_path / "data/output"
    first["thresholds_c"].append(33)
    save_settings(first, first["output_root"])
    assert second["thresholds_c"] == [25, 30]
    assert json.loads((first["output_root"] / "settings.json").read_text())[
        "thresholds_c"
    ] == [25, 30, 33]


def test_sample_configuration_matches_function_arguments():
    example = REPOSITORY / "config.example.yaml"
    from analysis.cellwise.iso7243 import analyze_cellwise_iso7243_classification
    from analysis.cellwise.joint_pipeline import (
        compute_joint_cellwise_temperature_quantiles,
    )
    from analysis.cellwise.population import compute_era5_population_weights
    from analysis.cellwise.spread import analyze_cellwise_quantile_spread
    from analysis.classification.era5 import analyze_era5_country_cache
    from analysis.comparison.workflows import run_daily_pairwise
    from analysis.comparison.yearly import analyze_yearly_pairwise
    from analysis.country.cells import export_country_cells
    from analysis.country.extraction import extract_countries_observations

    functions = {
        "daily_pairwise": run_daily_pairwise,
        "yearly_pairwise": analyze_yearly_pairwise,
        "quantiles": compute_joint_cellwise_temperature_quantiles,
        "population": compute_era5_population_weights,
        "iso7243": analyze_cellwise_iso7243_classification,
        "extract_country": extract_countries_observations,
        "country_cells": export_country_cells,
        "country_classification": analyze_era5_country_cache,
        "quantile_spread": analyze_cellwise_quantile_spread,
    }
    for section, function in functions.items():
        settings = config.load_settings(section, example)
        if section == "country_classification":
            settings.pop("output_root")
        if section == "quantile_spread":
            settings["quantity"] = settings.pop("quantities")[0]
        inspect.signature(function).bind(**settings)


def test_scripts_import_without_installation_or_config():
    modules = [f"scripts.{path.stem}" for path in (REPOSITORY / "scripts").glob("*.py")]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; "
            + "; ".join(f"importlib.import_module({name!r})" for name in modules),
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )


def test_daily_script_resume_input_changes_and_interrupted_write(tmp_path, monkeypatch):
    from analysis.comparison import persistence

    monkeypatch.setattr(config, "REPOSITORY", tmp_path)
    run = tmp_path / "run"
    for name, values in (("alpha", [25.0, 30.0]), ("beta", [26.0, 31.0])):
        _write_results(run, name, values)
    settings = dict(
        hsb_run_paths=[str(run)],
        start_date="2025-01-01",
        end_date="2025-01-01",
        implementations=["alpha", "beta"],
        output_root="out",
        workers=1,
        overwrite=False,
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"daily_pairwise": settings}))
    real_write = persistence.pq.write_table
    calls = 0

    def interrupt(table, destination, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted write")
        return real_write(table, destination, **kwargs)

    with patch.object(persistence.pq, "write_table", side_effect=interrupt):
        with pytest.raises(RuntimeError, match="interrupted write"):
            daily_pairwise.main()
    assert not persistence.completed_pairwise_dates(tmp_path / "out")
    daily_pairwise.main()
    result_path = tmp_path / "out/dates/2025-01-01/pairwise.parquet"
    original = pq.read_table(result_path)
    with patch(
        "analysis.comparison.workflows.compute_daily_pairwise",
        side_effect=AssertionError("must reuse"),
    ):
        daily_pairwise.main()
    assert pq.read_table(result_path).equals(original)
    _write_results(run, "alpha", [28.0, 30.0])
    with pytest.raises(ValueError, match="Incompatible pairwise inputs"):
        daily_pairwise.main()
    sentinel = tmp_path / "out/notes.txt"
    sentinel.write_text("keep")
    settings["overwrite"] = True
    path.write_text(yaml.safe_dump({"daily_pairwise": settings}))
    daily_pairwise.main()
    assert sentinel.read_text() == "keep"
    assert not pq.read_table(result_path).equals(original)
