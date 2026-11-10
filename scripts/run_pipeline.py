"""Run selected workflows sequentially; reuse and overwrite belong to each workflow."""

import argparse
from datetime import datetime, timezone
import importlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys

from analysis.config import REPOSITORY, load_settings

# Insertion order is execution order. Each workflow may use several YAML sections.
WORKFLOWS = {
    "population": "compute_era5_population_weights",
    "country_cells": "country_cells",
    "extract_country": "extract_country",
    "daily_pairwise": "daily_pairwise",
    "yearly_pairwise": "analyze_yearly_pairwise",
    "iso7243": "analyze_cellwise_iso7243_classification",
    "plot_iso7243": "plot_cellwise_iso7243_classification",
    "plot_non_consensus": "plot_cellwise_iso7243_non_consensus",
    "plot_non_consensus_stack": "plot_global_iso7243_non_consensus_stack",
    "country_classification": "analyze_era5_country_classification",
    "quantiles": "compute_joint_cellwise_temperature_quantiles",
    "quantile_spread": "analyze_cellwise_quantile_spread",
    "plot_quantile_spread": "plot_cellwise_quantile_spread",
}


def selected_steps(settings: dict) -> list[tuple[str, str]]:
    selected = settings["steps"]
    if not isinstance(selected, dict):
        raise ValueError("pipeline.steps must map workflows to lists of settings sections")
    if unknown := selected.keys() - WORKFLOWS.keys():
        raise ValueError(f"Unknown pipeline workflows: {sorted(unknown)}")
    steps = []
    for workflow in WORKFLOWS:
        sections = selected.get(workflow, [])
        if not isinstance(sections, list) or any(
            not isinstance(section, str) or not section for section in sections
        ):
            raise ValueError(f"pipeline.steps.{workflow} must be a list of section names")
        steps.extend((workflow, section) for section in sections)
    return steps


def run_pipeline(config_path: Path) -> int:
    settings = load_settings("pipeline", config_path)
    steps = selected_steps(settings)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    logs = settings["output_root"] / stamp
    logs.mkdir(parents=True)
    print(f"Pipeline logs: {logs}", flush=True)

    def record(event, **fields):
        entry = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with (logs / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    record("started", config=str(config_path))
    for index, (workflow, section) in enumerate(steps, 1):
        log_path = logs / f"{index:02d}-{workflow}.log"
        context = {"workflow": workflow, "settings": section, "log": str(log_path)}
        command = [
            sys.executable, "-u", "-m", "scripts.run_pipeline",
            "--config", str(config_path), "--workflow", workflow, "--settings", section,
        ]
        record("step_started", **context)
        process = None
        try:
            with log_path.open("w") as stream:
                process = subprocess.Popen(
                    command, cwd=REPOSITORY, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = process.wait()
        except BaseException as error:
            if process is not None:
                # Stop the step's workers too when the runner is interrupted.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record("failed", **context, error=f"{type(error).__name__}: {error}")
            raise
        record("step_finished", **context, returncode=code)
        if code:
            record("failed", **context, returncode=code)
            return 1
    record("completed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "config.yaml")
    parser.add_argument("--workflow", choices=WORKFLOWS, help=argparse.SUPPRESS)
    parser.add_argument("--settings", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workflow:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        module = importlib.import_module("scripts." + WORKFLOWS[args.workflow])
        module.main(load_settings(args.settings or args.workflow, args.config))
        return 0

    def interrupted(signum, frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, interrupted)
    return run_pipeline(args.config.expanduser().resolve())


if __name__ == "__main__":
    sys.exit(main())
