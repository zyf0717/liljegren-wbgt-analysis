#!/usr/bin/env python3
"""Compute one implementation's cellwise temporal WBGT quantiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise import compute_cellwise_wbgt_quantiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsb-run", action="append", required=True, type=Path)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/cellwise_wbgt_quantiles")
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--exact-validation-cells",
        type=int,
        default=100,
        help="deterministic exact-order-statistic validation sample; zero disables it",
    )
    arguments = parser.parse_args()
    output = compute_cellwise_wbgt_quantiles(
        arguments.hsb_run,
        implementation=arguments.implementation,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        output_root=arguments.output_root,
        threads=arguments.threads,
        exact_validation_cells=arguments.exact_validation_cells,
    )
    print(output)


if __name__ == "__main__":
    main()
