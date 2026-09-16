#!/usr/bin/env python3
"""Plot the standardized global ISO 7243 non-consensus map stack."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise import (
    DEFAULT_ISO7243_OUTPUT_ROOT,
    plot_global_iso7243_non_consensus_stack,
)
from liljegren_analysis.cellwise.population import DEFAULT_POPULATION_WEIGHTS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ISO7243_OUTPUT_ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--population-weights-path",
        type=Path,
        default=DEFAULT_POPULATION_WEIGHTS_PATH,
    )
    arguments = parser.parse_args()
    print(
        plot_global_iso7243_non_consensus_stack(
            arguments.input_root,
            output_directory=arguments.output_directory,
            population_weights_path=arguments.population_weights_path,
        )
    )


if __name__ == "__main__":
    main()
