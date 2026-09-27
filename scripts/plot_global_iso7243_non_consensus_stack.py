"""Run plot non consensus stack using config.yaml."""

from analysis.cellwise.iso7243_plots import plot_global_iso7243_non_consensus_stack
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main() -> None:
    settings = load_settings("plot_non_consensus_stack")
    result = plot_global_iso7243_non_consensus_stack(**settings)
    save_settings(
        settings,
        settings.get("output_directory") or settings["input_root"] / "figures",
        filename="plot_global_iso7243_non_consensus_stack.settings.json",
    )
    print(result)


if __name__ == "__main__":
    main()
