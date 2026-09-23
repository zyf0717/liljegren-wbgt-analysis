"""Build full-boundary LandScan/ERA5 population weights for Korea or Taiwan."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
from shapely.geometry import box

from liljegren_analysis.cellwise.population import load_population_weights
from liljegren_analysis.io.artifacts import atomic_json, atomic_parquet
from heat_health_common import SPECS, BOUNDARY_METADATA, BOUNDARY_REVISION, boundaries, boundary_url, region_cells


def build(country: str, boundary_path: Path, population_path: Path, output: Path) -> None:
    spec = SPECS[country]
    geometries, digest = boundaries(spec, boundary_path)
    cells = region_cells(spec, geometries)
    ids = sorted({cell.cell_id for group in cells.values() for cell in group})
    populations, _ = load_population_weights(ids, population_path)
    population = dict(zip(ids, populations, strict=True))
    rows = []
    totals = defaultdict(float)
    fractions = defaultdict(float)
    for code, group in sorted(cells.items()):
        for cell in group:
            footprint = box(cell.longitude - .125, cell.latitude - .125,
                            cell.longitude + .125, cell.latitude + .125)
            fraction = geometries[code].intersection(footprint).area / footprint.area
            if fraction <= 0:
                continue
            weight = float(population[cell.cell_id]) * fraction
            rows.append({"source_cell_id": cell.cell_id, "region_code": code,
                         "cell_area_fraction": fraction, "population_weight": weight})
            totals[code] += weight
            fractions[cell.cell_id] += fraction
    if set(totals) != set(spec.names.values()) or any(value <= 0 for value in totals.values()):
        raise ValueError(f"{country} region has no population weight")
    if any(value > 1.001 for value in fractions.values()):
        raise ValueError(f"{country} ADM1 geometries overlap within an ERA5 cell")
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        ("source_cell_id", pa.string()), ("region_code", pa.string()),
        ("cell_area_fraction", pa.float64()), ("population_weight", pa.float64()),
    ]))
    atomic_parquet(table, output)
    atomic_json({"source_url": boundary_url(country), "source_sha256": digest,
                 "boundary_revision": BOUNDARY_REVISION, "boundary_metadata": BOUNDARY_METADATA[country],
                 "region_count": len(totals),
                 "cell_count": len(ids), "cell_region_pairs": len(rows),
                 "population_weight_by_region": dict(sorted(totals.items())),
                 "weight_method": "LandScan 2024 ERA5-cell population times ADM1 polygon overlap; all boundary cells retained, including cells absent from HSB"},
                output.with_suffix(".json"))
    print(f"Saved {len(rows)} cell-region weights to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True, choices=SPECS)
    parser.add_argument("--boundaries", type=Path)
    parser.add_argument("--population", type=Path, default=Path("data/cellwise_population_weight/population.parquet"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    boundary_path = args.boundaries or Path(f"data/boundaries/geoboundaries-{args.country.lower()}-adm1.geojson")
    output = args.output or SPECS[args.country].root / "region_cell_weights.parquet"
    build(args.country, boundary_path, args.population, output)


if __name__ == "__main__":
    main()
