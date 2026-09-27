"""Run iso7243 using config.yaml."""

from analysis.cellwise.iso7243 import analyze_cellwise_iso7243_classification
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main(settings: dict | None = None) -> None:
    settings = dict(settings) if settings is not None else load_settings("iso7243")
    result = analyze_cellwise_iso7243_classification(**settings)
    save_settings(settings, settings["output_root"])
    print(result)


if __name__ == "__main__":
    main()
