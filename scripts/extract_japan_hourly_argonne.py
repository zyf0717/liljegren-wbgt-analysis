"""Extract country hourly Argonne WBGT components and acquired ERA5 air temperature.

Japan remains the default. Korea and Taiwan use pinned ADM1 boundaries so
their offshore and administrative cells are included. Install netCDF4 to run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from shapely.ops import unary_union

from liljegren_analysis.comparison.config import REFERENCE_IMPLEMENTATION
from liljegren_analysis.country import Country, CountryCell, country_cells, load_countries
from liljegren_analysis.results import HSBRunResults
from heat_health_common import SPECS, boundaries as load_region_boundaries

IMPLEMENTATION = REFERENCE_IMPLEMENTATION
ROW_ID = re.compile(r"era5-global-([ns])(\d{6})-([ew])(\d{7})-(\d{10})\Z")
SOURCE_COLUMNS = (
    "cycle_ordinal", "row_id", "row_ordinal", "wbgt_status", "wbgt_c",
    "tg_status", "tg_c", "tnwb_status", "tnwb_c",
)
OUTPUT_SCHEMA = pa.schema(
    [
        ("timestamp_utc", pa.timestamp("s", tz="UTC")),
        ("latitude", pa.float64()), ("longitude", pa.float64()),
        ("row_id", pa.string()), ("run_id", pa.string()),
        ("cycle_ordinal", pa.uint32()), ("row_ordinal", pa.uint32()),
        ("wbgt_status", pa.string()), ("wbgt_c", pa.float32()),
        ("tg_status", pa.string()), ("tg_c", pa.float32()),
        ("tnwb_status", pa.string()), ("tnwb_c", pa.float32()),
        ("ta_c", pa.float64()),
    ]
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", choices=("JPN", "KOR", "TWN"), default="JPN")
    parser.add_argument("--region-boundaries", type=Path,
                        help="Pinned ADM1 GeoJSON cache for KOR/TWN.")
    parser.add_argument("--run", type=Path, default=Path(".hsb/runs/wbgt-comparison-20260918-085051"))
    parser.add_argument("--acquired", type=Path, default=Path(".hsb/acquired/era5-global-2025"))
    parser.add_argument("--boundaries", type=Path, default=Path("data/boundaries/natural-earth-5.1.2-admin-0-countries-10m.geojson"))
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true", help="Replace an existing output after successful extraction.")
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")
    if args.output is None:
        args.output = Path(f"data/country/{args.country.lower()}/argonne-hourly-2025.parquet")
    return args


def selected_cells(source: netCDF4.Dataset, country_selected_cells: list[CountryCell]) -> dict[str, tuple[int, int, float, float]]:
    latitude = np.asarray(source.variables["latitude"][:], dtype=np.float64)
    longitude = np.asarray(source.variables["longitude"][:], dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("ERA5 latitude and longitude coordinates must be one-dimensional")
    normalized_longitude = np.where(longitude >= 180, longitude - 360, longitude)
    lat_index = {round(float(value), 4): index for index, value in enumerate(latitude)}
    lon_index = {round(float(value), 4): index for index, value in enumerate(normalized_longitude)}
    cells = {}
    for cell in country_selected_cells:
        try:
            cells[cell.cell_id] = (
                lat_index[round(cell.latitude, 4)], lon_index[round(cell.longitude, 4)],
                cell.latitude, cell.longitude,
            )
        except KeyError as error:
            raise ValueError(f"Country cell is absent from acquired ERA5 grid: {cell.cell_id}") from error
    return cells


def expected_shards(run: Path, start: date, end: date) -> dict[str, set[str]]:
    manifest = json.loads((run / "run-manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"]["provider"] != "era5" or IMPLEMENTATION not in manifest["targets"]:
        raise ValueError("Run is not an ERA5 run with the Argonne target")
    expected = {}
    for shard in manifest["execution"]["shard_order"]:
        day = shard["date"]
        if start.isoformat() <= day <= end.isoformat():
            expected.setdefault(day, set()).add(f"{shard['ordinal']:04d}")
    return expected


def air_temperature_for_day(
    source: netCDF4.Dataset, day: date, cells: dict[str, tuple[int, int, float, float]]
) -> dict[str, float]:
    variable = source.variables["t2m"]
    time_variable = source.variables.get("valid_time", source.variables.get("time"))
    if variable.dimensions not in {("valid_time", "latitude", "longitude"), ("time", "latitude", "longitude")} or variable.units != "K" or time_variable is None:
        raise ValueError("Unsupported acquired ERA5 t2m schema")
    start = (day.day - 1) * 24
    times = netCDF4.num2date(time_variable[start:start + 24], time_variable.units)
    actual_hours = [(value.year, value.month, value.day, value.hour) for value in times]
    expected_hours = [(day.year, day.month, day.day, hour) for hour in range(24)]
    if actual_hours != expected_hours:
        raise ValueError(f"Acquired ERA5 hours do not match {day}")
    lat_indices = [cell[0] for cell in cells.values()]
    lon_indices = [cell[1] for cell in cells.values()]
    lat_start, lat_stop = min(lat_indices), max(lat_indices) + 1
    lon_start, lon_stop = min(lon_indices), max(lon_indices) + 1
    values = np.ma.asarray(variable[start:start + 24, lat_start:lat_stop, lon_start:lon_stop])
    result = {}
    for prefix, (lat_index, lon_index, _, _) in cells.items():
        hourly = values[:, lat_index - lat_start, lon_index - lon_start]
        for hour, kelvin in enumerate(hourly):
            if not np.ma.is_masked(kelvin) and math.isfinite(float(kelvin)):
                result[f"{prefix}-{day:%Y%m%d}{hour:02d}"] = float(kelvin) - 273.15
    return result


def candidate_groups(reader: pq.ParquetFile, prefixes: set[str]) -> list[int]:
    column_index = reader.schema_arrow.names.index("row_id")
    first, last = min(prefixes), max(prefixes)
    selected = []
    for index in range(reader.metadata.num_row_groups):
        statistics = reader.metadata.row_group(index).column(column_index).statistics
        if statistics is None or not statistics.has_min_max:
            selected.append(index)
            continue
        minimum = statistics.min.decode() if isinstance(statistics.min, bytes) else statistics.min
        maximum = statistics.max.decode() if isinstance(statistics.max, bytes) else statistics.max
        if maximum[:28] >= first and minimum[:28] <= last:
            selected.append(index)
    return selected


def extract(args: argparse.Namespace) -> int:
    run, acquired, boundaries = (path.expanduser().resolve() for path in (args.run, args.acquired, args.boundaries))
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Output exists: {output}; pass --force to replace it")
    if args.country == "JPN":
        selected_country_cells = country_cells(load_countries(boundaries, ("JPN",))[0])
    else:
        spec = SPECS[args.country]
        region_path = args.region_boundaries or Path(f"data/boundaries/geoboundaries-{args.country.lower()}-adm1.geojson")
        geometries, _ = load_region_boundaries(spec, region_path)
        selected_country_cells = country_cells(Country("", spec.iso3, spec.iso3, unary_union(list(geometries.values()))))
    if not selected_country_cells:
        raise ValueError(f"No ERA5 grid cells overlap {args.country}")
    results = HSBRunResults.from_hsb_run(run)
    expected = expected_shards(run, args.start_date, args.end_date)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=output.parent, prefix=f".{output.stem}.", suffix=".parquet.tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    writer = None
    total = 0
    pending: list[pa.Table] = []
    pending_rows = 0
    try:
        month = None
        source = None
        cells = None
        day = args.start_date
        try:
            writer = pq.ParquetWriter(temporary, OUTPUT_SCHEMA, compression="zstd")
            while day <= args.end_date:
                period = (day.year, day.month)
                if period != month:
                    if source is not None:
                        source.close()
                    path = acquired / "raw" / str(day.year) / f"{day.month:02d}" / "instantaneous.nc"
                    source = netCDF4.Dataset(path)
                    current_cells = selected_cells(source, selected_country_cells)
                    if cells is not None and set(cells) != set(current_cells):
                        raise ValueError(f"ERA5 grid changed in {path}")
                    cells = current_cells
                    month = period
                assert source is not None and cells is not None
                ta = air_temperature_for_day(source, day, cells)
                actual = {item.run_id: item.path for item in results.files(result_date=day, implementation=IMPLEMENTATION)}
                if not expected.get(day.isoformat()) or set(actual) != expected[day.isoformat()]:
                    raise ValueError(f"Missing or unexpected Argonne result shards for {day}")
                seen = set()
                day_count = 0
                prefixes = set(cells)
                value_set = pa.array(sorted(prefixes))
                for run_id, path in sorted(actual.items()):
                    reader = pq.ParquetFile(path)
                    if not set(SOURCE_COLUMNS).issubset(reader.schema_arrow.names):
                        raise ValueError(f"Unsupported result schema: {path}")
                    for batch in reader.iter_batches(columns=list(SOURCE_COLUMNS), row_groups=candidate_groups(reader, prefixes), batch_size=65_536):
                        table = pa.Table.from_batches([batch])
                        selected = table.filter(pc.is_in(pc.utf8_slice_codeunits(table["row_id"], 0, 28), value_set=value_set))
                        if not selected.num_rows:
                            continue
                        ids = selected["row_id"].to_pylist()
                        cycles = selected["cycle_ordinal"].to_pylist()
                        for cycle, row_id in zip(cycles, ids, strict=True):
                            if ROW_ID.fullmatch(row_id) is None or row_id[-10:-2] != day.strftime("%Y%m%d"):
                                raise ValueError(f"Invalid ERA5 row identity: {row_id}")
                            if (cycle, row_id) in seen:
                                raise ValueError(f"Duplicate result row: {cycle}, {row_id}")
                            seen.add((cycle, row_id))
                        missing = [row_id for row_id in ids if row_id not in ta]
                        if missing:
                            raise ValueError(f"No finite acquired Ta for result row: {missing[0]}")
                        coordinate = [cells[row_id[:28]] for row_id in ids]
                        result = pa.table({
                            "timestamp_utc": pa.array([datetime.strptime(row_id[-10:], "%Y%m%d%H").replace(tzinfo=UTC) for row_id in ids], type=pa.timestamp("s", tz="UTC")),
                            "latitude": pa.array([item[2] for item in coordinate], type=pa.float64()),
                            "longitude": pa.array([item[3] for item in coordinate], type=pa.float64()),
                            "row_id": selected["row_id"],
                            "run_id": pa.array([run_id] * len(ids)),
                            **{name: selected[name] for name in SOURCE_COLUMNS if name != "row_id"},
                            "ta_c": pa.array([ta[row_id] for row_id in ids], type=pa.float64()),
                        }).cast(OUTPUT_SCHEMA).select(OUTPUT_SCHEMA.names)
                        pending.append(result)
                        pending_rows += result.num_rows
                        if pending_rows >= 250_000:
                            writer.write_table(pa.concat_tables(pending), row_group_size=250_000)
                            pending.clear()
                            pending_rows = 0
                        day_count += result.num_rows
                if day_count == 0:
                    raise ValueError(f"No {args.country} Argonne rows found for {day}")
                total += day_count
                print(f"{day}: {day_count:,} rows", file=sys.stderr, flush=True)
                day += timedelta(days=1)
        finally:
            if source is not None:
                source.close()
            if writer is not None:
                if pending:
                    writer.write_table(pa.concat_tables(pending), row_group_size=250_000)
                writer.close()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {total:,} rows to {output}")
    return total


def main() -> int:
    try:
        extract(arguments())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
