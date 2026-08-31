#!/usr/bin/env python3
"""Plot map and weighted distributions from cellwise ISO 7243 differences."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise.iso7243_plots import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_POPULATION_WEIGHTS_PATH,
    plot_cellwise_iso7243_classification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--population-weights-path", type=Path, default=DEFAULT_POPULATION_WEIGHTS_PATH
    )
    arguments = parser.parse_args()
    outputs = plot_cellwise_iso7243_classification(
        arguments.input_root,
        output_directory=arguments.output_directory,
        population_weights_path=arguments.population_weights_path,
    )
    print(outputs.pairwise_png)


if __name__ == "__main__":
    main()
