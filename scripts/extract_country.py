"""Extract valid observations for countries to reusable Parquet caches."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import duckdb

from liljegren_analysis import (
    CountryExtractionProgress,
    extract_countries_observations,
    extract_country_observations,
)


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
        description="Cache valid WBGT observations for one or more ADM0 countries.",
    )
    parser.add_argument(
        "countries",
        nargs="+",
        help="ISO alpha-2 or alpha-3 codes, e.g. SG JPN.",
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
    if args.output is not None and len(args.countries) != 1:
        parser.error("--output can only be used with one country")
    return args


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def run() -> int:
    args = parse_args()
    runs = tuple(args.run) if args.run is not None else DEFAULT_RUNS
    common = {
        "boundaries": args.boundaries,
        "boundary_cache": REPOSITORY / "data" / "boundaries",
        "threads": args.threads,
        "overwrite": args.force,
        "progress": _print_progress,
    }
    if args.output is not None:
        extractions = (
            extract_country_observations(
                args.countries[0],
                runs,
                args.start_date,
                args.end_date,
                args.output,
                **common,
            ),
        )
        outputs = (args.output.expanduser().resolve(),)
    else:
        output_directory = REPOSITORY / "data" / "country"
        extractions = extract_countries_observations(
            args.countries,
            runs,
            args.start_date,
            args.end_date,
            output_directory,
            **common,
        )
        outputs = tuple(
            output_directory
            / extraction.country_iso_a3.lower()
            / extraction.parquet_file
            for extraction in extractions
        )

    for extraction, output in zip(extractions, outputs, strict=True):
        print(
            f"Saved {extraction.observation_count:,} {extraction.country_iso_a3} "
            f"observations to {output}"
        )
    return 0


def _print_progress(progress: CountryExtractionProgress) -> None:
    percentage = (
        " unavailable"
        if progress.duckdb_percentage is None
        else f" {progress.duckdb_percentage:.1f}%"
    )
    print(
        f"[{progress.state}] phase={progress.phase} "
        f"elapsed={progress.elapsed_seconds:.1f}s "
        f"progress={percentage.strip()} "
        f"files={progress.selected_source_files}/{progress.total_source_files}",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, duckdb.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
