"""Run plot quantile spread using config.yaml."""

from analysis.cellwise.spread import plot_stacked_cellwise_quantile_spread
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main() -> None:
    settings = load_settings("plot_quantile_spread")
    result = plot_stacked_cellwise_quantile_spread(**settings)
    save_settings(
        settings,
        settings["output_root"] / "figures",
        filename="plot_cellwise_quantile_spread.settings.json",
    )
    print(result)


if __name__ == "__main__":
    main()
