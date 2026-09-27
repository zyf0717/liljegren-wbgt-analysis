"""Run plot non consensus using config.yaml."""

from analysis.cellwise.iso7243_plots import plot_cellwise_iso7243_non_consensus
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main() -> None:
    settings = load_settings("plot_non_consensus")
    result = plot_cellwise_iso7243_non_consensus(**settings)
    save_settings(
        settings,
        settings.get("output_directory") or settings["input_root"] / "figures",
        filename="plot_cellwise_iso7243_non_consensus.settings.json",
    )
    print(result)


if __name__ == "__main__":
    main()
