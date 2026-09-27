"""Run extract country using config.yaml."""

from analysis.config import load_settings
from analysis.country.extraction import extract_countries_observations
from analysis.io.artifacts import save_settings


def main(settings: dict | None = None) -> None:
    settings = dict(settings) if settings is not None else load_settings("extract_country")
    result = extract_countries_observations(**settings)
    save_settings(settings, settings["output_directory"])
    print(result)


if __name__ == "__main__":
    main()
