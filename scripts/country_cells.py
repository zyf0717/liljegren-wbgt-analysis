"""Emit ERA5 cells whose 0.25-degree footprints intersect countries."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TextIO

from liljegren_analysis.country.cells import (
    NATURAL_EARTH_VERSION,
    cached_natural_earth_boundaries,
    country_cells,
    load_countries,
)


REPOSITORY = Path(__file__).parents[1]
DEFAULT_CACHE_DIRECTORY = REPOSITORY / "data" / "boundaries"
FIELDNAMES = (
    "country_iso_a2",
    "country_iso_a3",
    "country_name",
    "cell_id",
    "row_id_prefix",
    "latitude",
    "longitude",
    "center_in_country",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select ERA5 0.25-degree cells with positive-area overlap with "
            "ADM0 country polygons. Accepts ISO alpha-2 or alpha-3 codes."
        ),
    )
    parser.add_argument("countries", nargs="+", help="Country codes, e.g. SG JPN.")
    parser.add_argument(
        "--boundaries",
        type=Path,
        help=(
            "Local Natural Earth-compatible GeoJSON. By default, download and "
            f"cache pinned Natural Earth {NATURAL_EARTH_VERSION} boundaries."
        ),
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=DEFAULT_CACHE_DIRECTORY,
        help=f"Boundary cache directory (default: {DEFAULT_CACHE_DIRECTORY}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path; omit to write CSV to stdout.",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    boundaries = (
        args.boundaries.expanduser().resolve()
        if args.boundaries is not None
        else cached_natural_earth_boundaries(
            args.cache_directory.expanduser().resolve()
        )
    )
    countries = load_countries(boundaries, args.countries)
    rows = [cell for country in countries for cell in country_cells(country)]

    if args.output is None:
        _write_csv(sys.stdout, rows)
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as destination:
            _write_csv(destination, rows)

    counts = ", ".join(
        f"{country.iso_a3}={sum(row.country_iso_a3 == country.iso_a3 for row in rows)}"
        for country in countries
    )
    print(f"Selected {len(rows)} country-cell rows ({counts})", file=sys.stderr)
    return 0


def _write_csv(destination: TextIO, rows: list) -> None:
    writer = csv.DictWriter(destination, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "country_iso_a2": row.country_iso_a2,
                "country_iso_a3": row.country_iso_a3,
                "country_name": row.country_name,
                "cell_id": row.cell_id,
                "row_id_prefix": row.row_id_prefix,
                "latitude": f"{row.latitude:.4f}",
                "longitude": f"{row.longitude:.4f}",
                "center_in_country": str(row.center_in_country).lower(),
            }
        )


def main() -> int:
    try:
        return run()
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
