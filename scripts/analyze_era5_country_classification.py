#!/usr/bin/env python3
"""Analyze WBGT classification changes from an extracted ERA5 country cache."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from liljegren_analysis.classification import (
    analyze_era5_country_cache,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path, help="Country observations.parquet")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--thresholds-c", required=True, nargs="+", type=float, metavar="CELSIUS"
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--temp-directory", type=Path)
    arguments = parser.parse_args()
    if arguments.start_date > arguments.end_date:
        parser.error("--start-date must not be after --end-date")
    return arguments


def main() -> int:
    arguments = parse_args()
    try:
        analysis = analyze_era5_country_cache(
            arguments.cache,
            arguments.start_date,
            arguments.end_date,
            arguments.thresholds_c,
            threads=arguments.threads,
            memory_limit=arguments.memory_limit,
            temp_directory=arguments.temp_directory,
        )
        write_outputs(analysis.outputs, arguments.output_root)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Saved {len(analysis.outputs)} classification outputs to {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
