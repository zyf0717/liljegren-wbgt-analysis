"""Run plot iso7243 using config.yaml."""

from analysis.cellwise.iso7243_plots import plot_cellwise_iso7243_classification
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main(settings: dict | None = None) -> None:
    settings = dict(settings) if settings is not None else load_settings("plot_iso7243")
    result = plot_cellwise_iso7243_classification(**settings)
    save_settings(
        settings,
        settings.get("output_directory") or settings["input_root"] / "figures",
        filename="plot_cellwise_iso7243_classification.settings.json",
    )
    print(result)


if __name__ == "__main__":
    main()
