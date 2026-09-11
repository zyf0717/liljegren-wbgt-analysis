#!/usr/bin/env python3
"""Analyze WBGT classification changes from an HSB NASA POWER run."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from liljegren_analysis.classification import analyze_nasa_power_run, write_outputs


REPOSITORY = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="HSB NASA POWER run root")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--thresholds-c", required=True, nargs="+", type=float, metavar="CELSIUS"
    )
    parser.add_argument("--shard-ordinal", type=int, default=1)
    parser.add_argument("--cycle-ordinal", type=int, default=1)
    parser.add_argument("--country", action="append", dest="country_codes")
    parser.add_argument("--location", action="append", dest="location_ids")
    parser.add_argument(
        "--implementation",
        action="append",
        dest="implementations",
        help="target implementation; repeat to replace the default six-target set",
    )
    parser.add_argument(
        "--boundary-cache", type=Path, default=REPOSITORY / "data" / "boundaries"
    )
    arguments = parser.parse_args()
    if arguments.start_date > arguments.end_date:
        parser.error("--start-date must not be after --end-date")
    return arguments


def main() -> int:
    arguments = parse_args()
    try:
        analysis = analyze_nasa_power_run(
            arguments.run,
            arguments.start_date,
            arguments.end_date,
            arguments.thresholds_c,
            shard_ordinal=arguments.shard_ordinal,
            cycle_ordinal=arguments.cycle_ordinal,
            country_codes=arguments.country_codes,
            location_ids=arguments.location_ids,
            boundary_cache=arguments.boundary_cache,
            implementations=arguments.implementations,
        )
        write_outputs(analysis.outputs, arguments.output_root)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Saved {len(analysis.outputs)} classification outputs to {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
