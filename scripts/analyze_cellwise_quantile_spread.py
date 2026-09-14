#!/usr/bin/env python3
"""Analyze persisted cellwise temperature quantile spread without shard scans."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise.spread import (
    DEFAULT_QUANTILE_SPREAD_INPUT_ROOT,
    analyze_cellwise_quantile_spread,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_QUANTILE_SPREAD_INPUT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
    )
    parser.add_argument("--quantity", choices=("wbgt", "tg", "tnwb"), default="wbgt")
    parser.add_argument(
        "--weighting", choices=("none", "geometry", "population"), default="geometry"
    )
    parser.add_argument(
        "--population-weights-path",
        type=Path,
        help="Population counts required for regional coverage under every weighting",
    )
    arguments = parser.parse_args()
    outputs = analyze_cellwise_quantile_spread(
        arguments.input_root,
        output_root=arguments.output_root,
        quantity=arguments.quantity,
        weighting=arguments.weighting,
        **(
            {"population_weights_path": arguments.population_weights_path}
            if arguments.population_weights_path is not None
            else {}
        ),
    )
    print(outputs.cellwise_spread)
    print(outputs.extrema_frequency_csv)
    print(outputs.tropical_spread_coverage_csv)
    print(outputs.spread_coverage_csv)


if __name__ == "__main__":
    main()
