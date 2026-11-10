"""Analyze each configured temperature quantity using config.yaml."""

from analysis.cellwise.spread import analyze_cellwise_quantile_spread
from analysis.cellwise.spread_coverage import export_spread_coverage
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main(settings: dict | None = None) -> None:
    settings = dict(settings) if settings is not None else load_settings("quantile_spread")
    quantities = settings.pop("quantities")
    output_root = settings.pop("output_root")
    for quantity in quantities:
        arguments = {
            **settings,
            "quantity": quantity,
            "output_root": output_root / quantity,
        }
        outputs = analyze_cellwise_quantile_spread(**arguments)
        if settings.get("population_weights_path") is not None:
            export_spread_coverage(outputs, settings["population_weights_path"])
        save_settings(arguments, arguments["output_root"])
        print(outputs / "cellwise_quantile_spread.parquet")


if __name__ == "__main__":
    main()
