"""Run extract country using config.yaml."""

from analysis.config import load_settings
from analysis.country.extraction import extract_countries_observations
from analysis.io.artifacts import save_settings


def main() -> None:
    settings = load_settings("extract_country")
    result = extract_countries_observations(**settings)
    save_settings(settings, settings["output_directory"])
    print(result)


if __name__ == "__main__":
    main()
