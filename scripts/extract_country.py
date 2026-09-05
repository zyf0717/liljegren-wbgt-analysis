"""Extract all matched country result rows to reusable Parquet caches."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from liljegren_analysis import (
    CountryExtractionProgress,
    extract_countries_observations,
    extract_country_observations,
)


REPOSITORY = Path(__file__).parents[1]
DEFAULT_RUNS = (
    REPOSITORY / ".hsb" / "runs" / "wbgt-comparison-20260823-181813",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache WBGT, Tg, and Tnwb result rows for one or more ADM0 countries."
        ),
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
        help="HSB run root; repeat for multiple roots.",
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


def run() -> int:
    args = parse_args()
    runs = tuple(args.run) if args.run is not None else DEFAULT_RUNS
    common = {
        "boundaries": args.boundaries,
        "boundary_cache": REPOSITORY / "data" / "boundaries",
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
            f"Saved {extraction.result_row_count:,} {extraction.country_iso_a3} "
            f"observations to {output}"
        )
    return 0


def _print_progress(progress: CountryExtractionProgress) -> None:
    print(
        f"[{progress.state}] phase={progress.phase} "
        f"elapsed={progress.elapsed_seconds:.1f}s "
        f"files={progress.files_completed}/{progress.files_total} "
        f"rows={progress.rows_selected:,}/{progress.rows_scanned:,}",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
