"""Read the local analysis settings; computation receives explicit arguments."""

from collections.abc import Sequence
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Literal

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
_PATH_KEYS = {
    "input_root",
    "output_root",
    "output_directory",
    "output",
    "cache_path",
    "country_cache",
    "input_raster",
    "population_weights_path",
    "boundaries",
    "boundary_cache",
}


SampleScope = Literal["pairwise", "all-implementations"]
SAMPLE_SCOPES: tuple[SampleScope, ...] = ("pairwise", "all-implementations")
DEFAULT_IMPLEMENTATIONS = (
    "argonne-liljegren-c",
    "heatstress",
    "meteor",
    "thermofeel",
    "kwodzicki-pywbgt",
    "kong-huber-pywbgt",
)
DEFAULT_PAIRWISE_IMPLEMENTATIONS = (
    "argonne-liljegren-c",
    "heatstress",
    "kong-huber-pywbgt",
    "kwodzicki-pywbgt",
    "meteor",
    "thermofeel",
)


def implementation_names(
    implementations: Sequence[str],
) -> tuple[str, ...]:
    """Keep the configured target order and reject ambiguous selections."""

    if isinstance(implementations, str):
        raise TypeError("implementations must be a sequence, not a string")
    selected = tuple(implementations)
    if len(selected) < 2:
        raise ValueError("implementations must contain at least two targets")
    if any(not implementation for implementation in selected):
        raise ValueError("implementation names must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("implementations must be unique")
    return selected


REFERENCE_IMPLEMENTATION = "argonne-liljegren-c"


DEFAULT_JOINT_OUTPUT_ROOT = Path("data/analysis/cellwise/quantiles")
DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT = Path("data/analysis/cellwise/spread")
DEFAULT_ISO7243_OUTPUT_ROOT = Path("data/analysis/cellwise/iso7243")

DEFAULT_DATE_WORKERS = 8
DEFAULT_PROBABILITIES = (0.50, 0.95)


def load_settings(workflow: str, config_path: str | Path | None = None) -> dict:
    """Load one YAML section with repository-relative paths and ISO dates.

    YAML anchors share settings explicitly in the file. Each call returns an
    independent dictionary so notebook overrides never mutate other workflows.
    """
    path = Path(config_path) if config_path is not None else REPOSITORY / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Copy config.example.yaml to config.yaml and edit it."
        )
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get(workflow), dict):
        raise ValueError(f"Missing or invalid configuration section: {workflow}")
    settings = deepcopy(config[workflow])
    for key, value in settings.items():
        if value is None:
            continue
        if key in _PATH_KEYS:
            settings[key] = _path(value)
        elif key in {"hsb_run_paths", "runs"}:
            if not isinstance(value, list):
                raise ValueError(f"{workflow}.{key} must be a list of paths")
            settings[key] = tuple(_path(item) for item in value)
        elif key in {"start_date", "end_date"}:
            settings[key] = date.fromisoformat(str(value))
        elif key in {
            "implementations",
            "probabilities",
            "export_implementations",
            "bounds",
        }:
            settings[key] = tuple(value)
    return settings


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return (REPOSITORY / path).resolve()
