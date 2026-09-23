"""Normalize Taiwan's 2025 heat-injury visit CSV to county-day Parquet."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow as pa

from liljegren_analysis.io.artifacts import atomic_json, atomic_parquet
from heat_health_common import SPECS, cache_input, download

DATASET_PAGE = "https://data.gov.tw/dataset/157637"
METADATA_URL = "https://data.gov.tw/api/v2/rest/dataset/157637"
HEADERS = ("year", "admit_date", "age1", "SEX", "county", "SUM_of_count")


def parse_csv(payload: bytes) -> list[dict]:
    try:
        source = csv.DictReader(io.StringIO(payload.decode("big5")))
        if tuple(source.fieldnames or ()) != HEADERS:
            raise ValueError("Taiwan CSV columns changed")
        counts = defaultdict(int)
        spec = SPECS["TWN"]
        for row in source:
            if row["year"] != "2025":
                continue
            day = date.fromisoformat(row["admit_date"])
            if not spec.first_day <= day <= spec.last_day or row["county"] not in spec.names:
                raise ValueError(f"Invalid Taiwan date/county: {row}")
            if row["SEX"] not in {"M", "F"} or not row["age1"]:
                raise ValueError(f"Invalid Taiwan sex/age: {row}")
            count = int(row["SUM_of_count"])
            if count < 0:
                raise ValueError("Negative Taiwan visit count")
            counts[(day, spec.names[row["county"]])] += count
    except (UnicodeError, csv.Error, KeyError, TypeError) as error:
        raise ValueError("Unsupported Taiwan CSV schema or encoding") from error
    if not counts:
        raise ValueError("Taiwan CSV has no 2025 visits")
    return [{"local_date": spec.first_day + timedelta(days=i), "region_code": code,
             "outcome_count": counts[(spec.first_day + timedelta(days=i), code)]}
            for i in range(365) for code in sorted(spec.names.values())]


def run(output: Path, input_csv: Path | None, ca_bundle: Path | None, refresh: bool) -> None:
    spec = SPECS["TWN"]
    output.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(METADATA_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
        metadata = json.load(response)["result"]
    distributions = [item for item in metadata["distribution"] if item["resourceFormat"] == "CSV"]
    if len(distributions) != 1 or distributions[0]["resourceCharacterEncoding"].upper() != "BIG5":
        raise ValueError("Taiwan dataset distribution changed")
    url = distributions[0]["resourceDownloadUrl"]
    raw = output / "raw" / "heat-injury.csv"
    digest = (download(url, raw, refresh=refresh, ca_bundle=ca_bundle) if input_csv is None
              else cache_input(input_csv, raw))
    rows = parse_csv(raw.read_bytes())
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        ("local_date", pa.date32()), ("region_code", pa.string()), ("outcome_count", pa.int32()),
    ]))
    atomic_parquet(table, output / "region_daily.parquet")
    atomic_json({"dataset_page": DATASET_PAGE, "metadata_url": METADATA_URL,
                 "metadata_modified": metadata["modifiedDate"], "source_url": url,
                 "license_code": metadata["license"],
                 "raw_path": str(raw.resolve()), "raw_sha256": digest,
                 "date_meaning": "admission date, Asia/Taipei", "period": ["2025-01-01", "2025-12-31"],
                 "regions": 22, "days": 365, "rows": len(rows),
                 "outcome_count": sum(row["outcome_count"] for row in rows),
                 "surveillance_limitation": "Trend-monitoring sample; does not cover all medical institutions"},
                output / "manifest.json")
    print(f"Saved {len(rows)} county-days to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SPECS["TWN"].root)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.input_csv, args.ca_bundle, args.refresh)


if __name__ == "__main__":
    main()
