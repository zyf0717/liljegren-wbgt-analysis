"""Run plot quantile spread using config.yaml."""

from analysis.cellwise.spread_plots import (
    plot_quantile_spread,
    plot_stacked_cellwise_quantile_spread,
)
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main(settings: dict | None = None) -> None:
    settings = dict(settings) if settings is not None else load_settings("plot_quantile_spread")
    for quantity in ("wbgt", "tnwb", "tg"):
        plot_quantile_spread(settings["output_root"] / quantity)
    result = plot_stacked_cellwise_quantile_spread(**settings)
    save_settings(
        settings,
        settings["output_root"] / "figures",
        filename="plot_cellwise_quantile_spread.settings.json",
    )
    print(result)


if __name__ == "__main__":
    main()
