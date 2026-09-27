"""Export country cells using config.yaml."""

from analysis.config import load_settings
from analysis.country.cells import export_country_cells
from analysis.io.artifacts import save_settings


def main() -> None:
    settings = load_settings("country_cells")
    print(export_country_cells(**settings))
    save_settings(settings, settings["output"].parent, filename="cells.settings.json")


if __name__ == "__main__":
    main()
