#!/usr/bin/env python3
"""Plot annual cellwise ISO 7243 non-consensus percentages."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise.non_consensus import DEFAULT_OUTPUT_ROOT
from liljegren_analysis.cellwise.non_consensus_plots import (
    plot_cellwise_iso7243_non_consensus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()
    print(
        plot_cellwise_iso7243_non_consensus(
            arguments.input_root, output_directory=arguments.output_directory
        ).png
    )


if __name__ == "__main__":
    main()
