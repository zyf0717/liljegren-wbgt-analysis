"""Summarize ERA5 country classifications using config.yaml."""

from analysis.classification.era5 import analyze_era5_country_cache
from analysis.config import load_settings
from analysis.io.artifacts import save_settings, write_tables


def main() -> None:
    settings = load_settings("country_classification")
    arguments = dict(settings)
    output_root = arguments.pop("output_root")
    tables = analyze_era5_country_cache(**arguments)
    write_tables(tables, output_root)
    save_settings(settings, output_root)
    print(output_root)


if __name__ == "__main__":
    main()
