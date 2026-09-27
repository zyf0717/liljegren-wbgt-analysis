"""Analyze each configured temperature quantity using config.yaml."""

from analysis.cellwise.spread import analyze_cellwise_quantile_spread
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main() -> None:
    settings = load_settings("quantile_spread")
    quantities = settings.pop("quantities")
    output_root = settings.pop("output_root")
    for quantity in quantities:
        arguments = {
            **settings,
            "quantity": quantity,
            "output_root": output_root / quantity,
        }
        outputs = analyze_cellwise_quantile_spread(**arguments)
        save_settings(arguments, arguments["output_root"])
        print(outputs.cellwise_spread)


if __name__ == "__main__":
    main()
