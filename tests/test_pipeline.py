"""Sequential orchestration delegates reuse and overwrite to real workflows."""

import json
from pathlib import Path
import signal
import subprocess
import sys

import pyarrow.parquet as pq
import pytest
import yaml
from test_pairwise import _write_results

from analysis.config import REPOSITORY, load_settings
from scripts import run_pipeline


def test_pipeline_order_resume_overwrite_and_failure(tmp_path):
    run = tmp_path / "run"
    _write_results(run, "alpha", [25.0, 30.0])
    _write_results(run, "beta", [26.0, 31.0])
    dates = {"start_date": "2025-01-01", "end_date": "2025-01-01"}
    config = {
        "pipeline": {
            "output_root": str(tmp_path / "logs"),
            # Deliberately reversed; Python orders the workflows.
            "steps": {"yearly_pairwise": ["annual"], "daily_pairwise": ["daily"]},
        },
        "daily": {
            **dates, "hsb_run_paths": [str(run)], "implementations": ["alpha", "beta"],
            "output_root": str(tmp_path / "daily"), "workers": 1, "overwrite": False,
        },
        "annual": {
            **dates, "input_root": str(tmp_path / "daily"),
            "output_root": str(tmp_path / "annual"), "reference_implementation": "alpha",
        },
    }
    path = tmp_path / "config.yaml"

    def invoke(expected_code=0):
        path.write_text(yaml.safe_dump(config))
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_pipeline", "--config", str(path)],
            cwd=REPOSITORY, capture_output=True, text=True,
        )
        assert result.returncode == expected_code, result.stdout + result.stderr
        log = sorted((tmp_path / "logs").glob("*/events.jsonl"))[-1]
        return [json.loads(line) for line in log.read_text().splitlines()]

    events = invoke()
    assert [e["event"] for e in events] == [
        "started", "step_started", "step_finished", "step_started", "step_finished",
        "completed",
    ]
    assert [e["workflow"] for e in events if e["event"] == "step_started"] == [
        "daily_pairwise", "yearly_pairwise",
    ]
    daily = tmp_path / "daily/dates/2025-01-01/pairwise.parquet"
    annual = tmp_path / "annual/yearly_pairwise.parquet"
    assert pq.read_table(annual).num_rows == 3
    before = daily.stat().st_mtime_ns
    annual_before = annual.stat().st_mtime_ns
    invoke()
    assert daily.stat().st_mtime_ns == before  # workflow checkpoint reused
    assert annual.stat().st_mtime_ns != annual_before  # no runner-level skipping

    _write_results(run, "alpha", [28.0, 30.0])
    annual_before = annual.stat().st_mtime_ns
    events = invoke(expected_code=1)
    assert events[-1]["event"] == "failed"
    assert annual.stat().st_mtime_ns == annual_before  # stopped before next step
    assert "Incompatible pairwise inputs" in Path(events[-1]["log"]).read_text()
    config["daily"]["overwrite"] = True
    invoke()
    assert daily.stat().st_mtime_ns != before


def test_signal_exit_is_logged_and_stops_pipeline(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"pipeline": {
        "output_root": str(tmp_path / "logs"),
        "steps": {"daily_pairwise": ["first", "second"]},
    }}))
    real_popen = subprocess.Popen
    calls = []

    def terminated_step(command, **kwargs):
        calls.append(command)
        return real_popen([
            sys.executable, "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ], **kwargs)

    monkeypatch.setattr(run_pipeline.subprocess, "Popen", terminated_step)
    assert run_pipeline.run_pipeline(path) == 1
    assert len(calls) == 1
    events_path, = (tmp_path / "logs").glob("*/events.jsonl")
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert events[-1]["returncode"] == -signal.SIGTERM


def test_selection_and_example_sections():
    assert run_pipeline.selected_steps({"steps": {"population": []}}) == []
    with pytest.raises(ValueError, match="Unknown"):
        run_pipeline.selected_steps({"steps": {"typo": []}})
    with pytest.raises(ValueError, match="list"):
        run_pipeline.selected_steps({"steps": {"daily_pairwise": "daily"}})
    example = REPOSITORY / "config.example.yaml"
    for _, section in run_pipeline.selected_steps(load_settings("pipeline", example)):
        assert load_settings(section, example)
