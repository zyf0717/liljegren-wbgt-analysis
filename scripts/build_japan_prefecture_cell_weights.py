"""Map Japan ERA5 result cells to 2025 MLIT prefectures with population weights.

Run: uv run --with ijson python scripts/build_japan_prefecture_cell_weights.py
The source is the 2025-01-01 N03 administrative-area archive (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import box, shape
from shapely.strtree import STRtree

from liljegren_analysis.cellwise.population import load_population_weights
from liljegren_analysis.io.artifacts import atomic_json, atomic_parquet

SOURCE_PAGE = "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2025.html"
SOURCE_URL = "https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2025/N03-20250101_GML.zip"
GEOJSON_MEMBER = "N03-20250101_prefecture.geojson"


def download_archive(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as target, urlopen(SOURCE_URL, timeout=60) as source:
            while chunk := source.read(4 * 1024 * 1024):
                target.write(chunk)
        with ZipFile(name) as archive:
            if GEOJSON_MEMBER not in archive.namelist():
                raise ValueError(f"Missing {GEOJSON_MEMBER} in MLIT archive")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def result_cells(path: Path) -> dict[str, tuple[float, float]]:
    reader = pq.ParquetFile(path)
    cells = {}
    for batch in reader.iter_batches(columns=["row_id", "latitude", "longitude"], batch_size=65_536):
        ids, latitudes, longitudes = (batch.column(i).to_pylist() for i in range(3))
        for row_id, latitude, longitude in zip(ids, latitudes, longitudes, strict=True):
            cell_id = row_id[:28]
            coordinate = (latitude, longitude)
            if cell_id in cells and cells[cell_id] != coordinate:
                raise ValueError(f"Inconsistent coordinate for {cell_id}")
            cells[cell_id] = coordinate
    if not cells:
        raise ValueError(f"No ERA5 cells in {path}")
    return cells


def fdma_names(path: Path) -> dict[str, str]:
    table = pq.read_table(path, columns=["prefecture_code", "prefecture_name_jp"])
    names = {}
    for code, name in zip(table["prefecture_code"].to_pylist(), table["prefecture_name_jp"].to_pylist(), strict=True):
        if code in names and names[code] != name:
            raise ValueError(f"Inconsistent FDMA prefecture name for {code}")
        names[code] = name
    if set(names) != {f"{value:02d}" for value in range(1, 48)}:
        raise ValueError("FDMA table does not contain all 47 prefecture codes")
    return names


def build_weights(
    archive_path: Path, hourly_path: Path, population_path: Path, fdma_path: Path,
    output_path: Path,
) -> None:
    cells = result_cells(hourly_path)
    names = fdma_names(fdma_path)
    ids = sorted(cells)
    population, _ = load_population_weights(ids, population_path)
    footprints = [box(cells[id][1] - 0.125, cells[id][0] - 0.125,
                      cells[id][1] + 0.125, cells[id][0] + 0.125) for id in ids]
    index = STRtree(footprints)
    area_by_pair: dict[tuple[int, str], float] = defaultdict(float)
    matched_features = 0
    with ZipFile(archive_path) as archive, archive.open(GEOJSON_MEMBER) as source:
        for feature in ijson.items(source, "features.item", use_float=True):
            properties = feature["properties"]
            code = str(properties["N03_007"])[:2]
            if code not in names or properties["N03_001"] != names[code]:
                raise ValueError(f"MLIT/FDMA prefecture mismatch: {code}")
            polygon = shape(feature["geometry"])
            for cell_index in index.query(polygon):
                overlap = polygon.intersection(footprints[cell_index]).area
                if overlap > 0:
                    area_by_pair[(int(cell_index), code)] += overlap
                    matched_features += 1
    records = []
    prefecture_weight = defaultdict(float)
    cell_fraction = defaultdict(float)
    for (cell_index, code), area in sorted(area_by_pair.items()):
        fraction = area / footprints[cell_index].area
        weight = float(population[cell_index]) * fraction
        records.append({
            "source_cell_id": ids[cell_index], "prefecture_code": code,
            "prefecture_name_jp": names[code], "cell_area_fraction": fraction,
            "population_weight": weight,
        })
        prefecture_weight[code] += weight
        cell_fraction[cell_index] += fraction
    missing_cells = [ids[index] for index in range(len(ids)) if index not in cell_fraction]
    if missing_cells:
        raise ValueError(f"Result cells have no MLIT prefecture overlap: {missing_cells[:5]}")
    if any(fraction > 1.001 for fraction in cell_fraction.values()):
        raise ValueError("MLIT prefecture geometries overlap within an ERA5 cell")
    missing_prefectures = [code for code in names if prefecture_weight[code] <= 0]
    if missing_prefectures and missing_prefectures != ["47"]:
        raise ValueError(f"Unexpected prefectures without ERA5 result coverage: {missing_prefectures}")
    table = pa.Table.from_pylist(records, schema=pa.schema([
        ("source_cell_id", pa.string()), ("prefecture_code", pa.string()),
        ("prefecture_name_jp", pa.string()), ("cell_area_fraction", pa.float64()),
        ("population_weight", pa.float64()),
    ]))
    atomic_parquet(table, output_path)
    with archive_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    atomic_json({
        "source_page": SOURCE_PAGE, "source_url": SOURCE_URL,
        "source_archive_sha256": digest, "source_member": GEOJSON_MEMBER,
        "source_date": "2025-01-01", "source_license": "CC BY 4.0",
        "weight_method": "LandScan 2024 ERA5-cell population multiplied by MLIT prefecture polygon overlap fraction; assumes uniform population within each ERA5 cell",
        "result_cells": len(ids), "cell_prefecture_pairs": len(records),
        "matched_feature_cell_intersections": matched_features,
        "missing_prefecture_codes": missing_prefectures,
        "prefecture_population_weights": dict(sorted(prefecture_weight.items())),
    }, output_path.with_suffix(".json"))
    print(f"Saved {len(records)} cell-prefecture weights for {len(names) - len(missing_prefectures)} prefectures to {output_path}; missing={missing_prefectures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/boundaries/N03-20250101_GML.zip"))
    parser.add_argument("--hourly", type=Path, default=Path("data/country/jpn/argonne-hourly-2025.parquet"))
    parser.add_argument("--population", type=Path, default=Path("data/cellwise_population_weight/population.parquet"))
    parser.add_argument("--fdma", type=Path, default=Path("data/fdma/2025/prefecture_daily.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/fdma/2025/prefecture_cell_weights.parquet"))
    args = parser.parse_args()
    download_archive(args.archive)
    build_weights(args.archive, args.hourly, args.population, args.fdma, args.output)


if __name__ == "__main__":
    main()
