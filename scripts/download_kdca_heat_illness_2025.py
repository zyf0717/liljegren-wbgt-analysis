"""Normalize KDCA's final 2025 daily heat-illness PDF to regional Parquet."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa

from liljegren_analysis.io.artifacts import atomic_json, atomic_parquet
from heat_health_common import SPECS, cache_input, download

SOURCE_PAGE = "https://www.kdca.go.kr/eng/4442/subview.do"
REPORT_PAGE = "https://www.kdca.go.kr/kdca/4921/subview.do"
PDF_URL = "https://www.kdca.go.kr/bbs/kdca/258/265700/download.do"
ROW = re.compile(r"^\s*(2025-\d{2}-\d{2})\s+((?:\d+\s*)+)$")


def parse_pdf(payload: bytes) -> list[dict]:
    if not payload.startswith(b"%PDF-"):
        raise ValueError("KDCA response is not a PDF")
    try:
        result = subprocess.run(["pdftotext", "-layout", "-", "-"], input=payload,
                                check=True, capture_output=True)
    except FileNotFoundError as error:
        raise RuntimeError("Install Poppler's pdftotext to parse KDCA PDF") from error
    text = result.stdout.decode("utf-8")
    spec = SPECS["KOR"]
    codes = list(spec.names.values())
    rows = []
    days = set()
    for line in text.splitlines():
        match = ROW.fullmatch(line)
        if not match:
            continue
        day = date.fromisoformat(match.group(1))
        values = [int(value) for value in match.group(2).split()]
        if len(values) != 36 or day in days or not spec.first_day <= day <= spec.last_day:
            raise ValueError(f"Unexpected KDCA daily row: {day}")
        days.add(day)
        if values[0] != sum(values[2::2]) or values[1] != sum(values[3::2]):
            raise ValueError(f"KDCA regional total differs on {day}")
        rows.extend({"local_date": day, "region_code": code, "outcome_count": values[2 + 2 * i],
                     "deaths": values[3 + 2 * i]} for i, code in enumerate(codes))
    expected = {spec.first_day + timedelta(days=i) for i in range((spec.last_day - spec.first_day).days + 1)}
    if days != expected or sum(row["outcome_count"] for row in rows) != 4460 or sum(row["deaths"] for row in rows) != 29:
        raise ValueError("KDCA dates or final 2025 totals differ")
    return rows


def run(output: Path, input_pdf: Path | None, refresh: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw" / "kdca-final-2025.pdf"
    digest = (download(PDF_URL, raw, refresh=refresh) if input_pdf is None
              else cache_input(input_pdf, raw))
    rows = parse_pdf(raw.read_bytes())
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        ("local_date", pa.date32()), ("region_code", pa.string()),
        ("outcome_count", pa.int32()), ("deaths", pa.int32()),
    ]))
    atomic_parquet(table, output / "region_daily.parquet")
    atomic_json({"source_page": SOURCE_PAGE, "report_page": REPORT_PAGE, "pdf_url": PDF_URL,
                 "raw_path": str(raw.resolve()), "raw_sha256": digest, "date_meaning": "symptom onset, Asia/Seoul",
                 "period": ["2025-05-15", "2025-09-25"], "regions": 17, "days": 134,
                 "rows": len(rows), "outcome_count": 4460, "deaths_included_in_outcome": 29}, output / "manifest.json")
    print(f"Saved {len(rows)} region-days to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SPECS["KOR"].root)
    parser.add_argument("--input-pdf", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.input_pdf, args.refresh)


if __name__ == "__main__":
    main()
