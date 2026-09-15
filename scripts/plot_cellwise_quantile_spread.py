#!/usr/bin/env python3
"""Plot stacked WBGT, Tnwb, and Tg cellwise quantile-spread maps."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise import (
    DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT,
    plot_stacked_cellwise_quantile_spread,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT,
    )
    arguments = parser.parse_args()
    for output in plot_stacked_cellwise_quantile_spread(arguments.output_root):
        print(output)


if __name__ == "__main__":
    main()
