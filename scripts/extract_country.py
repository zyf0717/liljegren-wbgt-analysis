"""Extract valid observations for one country to a reusable Parquet cache."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import duckdb

from liljegren_analysis import extract_country_observations


REPOSITORY = Path(__file__).parents[1]
DEFAULT_RUNS = (
    REPOSITORY / ".hsb" / "campaigns" / "era5-2025-all-six-m2-w20-v8",
    REPOSITORY
    / ".hsb"
    / "runs"
    / "era5-2025-meteor-m4-w18"
    / "wbgt-comparison-20260810-231240",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache valid WBGT observations for one ADM0 country.",
    )
    parser.add_argument(
        "country",
        help="ISO alpha-2 or alpha-3 code, e.g. SG or JPN.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date(2025, 1, 1),
        help="Inclusive ISO date (default: 2025-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date(2025, 12, 31),
        help="Inclusive ISO date (default: 2025-12-31).",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        help="Campaign/run root; repeat for multiple roots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Parquet path; defaults to data/country/<iso3>/observations.parquet.",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        help="Local Natural Earth-compatible GeoJSON boundary file.",
    )
    parser.add_argument(
        "--threads",
        type=_non_negative_int,
        default=0,
        help="DuckDB threads; 0 keeps DuckDB's default (default: 0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing country cache.",
    )
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    return args


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def run() -> int:
    args = parse_args()
    runs = tuple(args.run) if args.run is not None else DEFAULT_RUNS
    output = args.output
    if output is None:
        from liljegren_analysis.country_cells import (
            cached_natural_earth_boundaries,
            load_countries,
        )

        boundaries = (
            args.boundaries.expanduser().resolve()
            if args.boundaries is not None
            else cached_natural_earth_boundaries(REPOSITORY / "data" / "boundaries")
        )
        country = load_countries(boundaries, [args.country])[0]
        output = (
            REPOSITORY
            / "data"
            / "country"
            / country.iso_a3.lower()
            / "observations.parquet"
        )

    extraction = extract_country_observations(
        args.country,
        runs,
        args.start_date,
        args.end_date,
        output,
        boundaries=args.boundaries,
        boundary_cache=REPOSITORY / "data" / "boundaries",
        threads=args.threads,
        overwrite=args.force,
    )
    print(
        f"Saved {extraction.observation_count:,} {extraction.country_iso_a3} "
        f"observations to {Path(output).expanduser().resolve()}"
    )
    return 0


def main() -> int:
    try:
        return run()
    except (OSError, ValueError, duckdb.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
