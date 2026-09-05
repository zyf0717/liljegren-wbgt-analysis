#!/usr/bin/env python3
"""Write annual cellwise ISO 7243 non-consensus percentages from raw results."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise.non_consensus import (
    DEFAULT_OUTPUT_ROOT,
    analyze_cellwise_iso7243_non_consensus,
)


REPOSITORY = Path(__file__).parents[1]
DEFAULT_HSB_RUN = (
    REPOSITORY.parent
    / "HeatStressBench"
    / ".hsb"
    / "runs"
    / "wbgt-comparison-20260902-150355"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsb-run", action="append", type=Path)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--thresholds-c", required=True, nargs="+", type=float)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    outputs = analyze_cellwise_iso7243_non_consensus(
        arguments.hsb_run or [DEFAULT_HSB_RUN],
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        thresholds_c=arguments.thresholds_c,
        output_root=arguments.output_root,
        force=arguments.force,
    )
    print(outputs.cellwise_non_consensus)


if __name__ == "__main__":
    main()
