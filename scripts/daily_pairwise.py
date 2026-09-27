"""Run daily pairwise using config.yaml."""

from analysis.comparison.workflows import run_daily_pairwise
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main() -> None:
    settings = load_settings("daily_pairwise")
    result = run_daily_pairwise(**settings)
    save_settings(settings, settings["output_root"])
    print(result)


if __name__ == "__main__":
    main()
