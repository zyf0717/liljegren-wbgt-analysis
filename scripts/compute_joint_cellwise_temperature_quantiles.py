"""Run quantiles using config.yaml."""

from analysis.cellwise.joint_pipeline import (
    compute_joint_cellwise_temperature_quantiles,
)
from analysis.config import load_settings
from analysis.io.artifacts import save_settings


def main(settings: dict | None = None) -> None:
    settings = dict(settings) if settings is not None else load_settings("quantiles")
    result = compute_joint_cellwise_temperature_quantiles(**settings)
    save_settings(settings, settings["output_root"])
    print(result)


if __name__ == "__main__":
    main()
