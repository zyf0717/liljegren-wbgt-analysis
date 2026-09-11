#!/usr/bin/env python3
"""Persist daily pairwise temperature statistics."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from liljegren_analysis.comparison import (
    DEFAULT_PAIRWISE_OUTPUT_ROOT,
    run_daily_pairwise,
)
from liljegren_analysis.comparison.config import SAMPLE_SCOPES

REPOSITORY = Path(__file__).parents[1]
DEFAULT_RUNS = (
    REPOSITORY.parent
    / "HeatStressBench"
    / ".hsb"
    / "runs"
    / "wbgt-comparison-20260902-150355",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date", type=date.fromisoformat, default=date(2025, 1, 1)
    )
    parser.add_argument(
        "--end-date", type=date.fromisoformat, default=date(2025, 12, 31)
    )
    parser.add_argument("--run", action="append", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=REPOSITORY / DEFAULT_PAIRWISE_OUTPUT_ROOT
    )
    parser.add_argument(
        "--sample-scope", choices=SAMPLE_SCOPES, default="all-implementations"
    )
    parser.add_argument(
        "--implementation",
        action="append",
        dest="implementations",
        help="target implementation; repeat to replace the default set",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="process workers (default: 1); Polars manages threads automatically",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(
        run_daily_pairwise(
            arguments.run or DEFAULT_RUNS,
            start_date=arguments.start_date,
            end_date=arguments.end_date,
            output_root=arguments.output_root,
            sample_scope=arguments.sample_scope,
            implementations=arguments.implementations,
            workers=arguments.workers,
            overwrite=arguments.force,
        )
    )


if __name__ == "__main__":
    main()
