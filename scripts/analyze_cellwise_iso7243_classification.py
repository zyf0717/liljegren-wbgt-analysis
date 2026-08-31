#!/usr/bin/env python3
"""Write annual ISO 7243 directional classification differences by cell pair."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise.iso7243 import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    analyze_cellwise_iso7243_classification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--thresholds-c", required=True, nargs="+", type=float, metavar="CELSIUS"
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    outputs = analyze_cellwise_iso7243_classification(
        arguments.input_root,
        thresholds_c=arguments.thresholds_c,
        output_root=arguments.output_root,
        force=arguments.force,
    )
    print(outputs.cell_pair_differences)


if __name__ == "__main__":
    main()
