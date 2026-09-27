"""Run population using config.yaml."""

from analysis.cellwise.population import compute_era5_population_weights
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main() -> None:
    settings = load_settings("population")
    result = compute_era5_population_weights(**settings)
    save_settings(settings, settings["output_root"])
    print(result)


if __name__ == "__main__":
    main()
