"""Run yearly pairwise using config.yaml."""

from analysis.comparison.yearly import analyze_yearly_pairwise
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main() -> None:
    settings = load_settings("yearly_pairwise")
    result = analyze_yearly_pairwise(**settings)
    save_settings(settings, settings["output_root"])
    print(result)


if __name__ == "__main__":
    main()
