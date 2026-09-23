"""Download FDMA's finalized 2025 heatstroke transports and consolidate daily prefecture counts.

Run: uv run python scripts/download_fdma_heatstroke_2025.py
Output: data/fdma/2025/prefecture_daily.csv and prefecture_daily.parquet.
FDMA reports May-September only; dates are Japanese local calendar dates.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import os
import tempfile
import unicodedata
import zipfile
from datetime import date, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq
from openpyxl import load_workbook

SOURCE_PAGE = "https://www.fdma.go.jp/disaster/heatstroke/post4.html"
DATA_URL = "https://www.fdma.go.jp/disaster/heatstroke/items/heatstroke003_data_r7.xlsx"
CODE_URL = "https://www.fdma.go.jp/disaster/heatstroke/items/heatstroke003_code.xlsx"
ANNUAL_REPORT = "https://www.fdma.go.jp/pressrelease/houdou/items/kyuuki_20251027_2.pdf"
EXPECTED_MONTHS = (5, 6, 7, 8, 9)
EXPECTED_TOTAL = 100_510
EXPECTED_MONTH_TOTALS = {5: 2_614, 6: 17_229, 7: 39_375, 8: 31_526, 9: 9_766}

# Ordered exactly as in the FDMA workbook. The source has a trailing space on one header.
SOURCE_HEADERS = (
    "日付", "都道府県コード", "搬送人員（計）", "年齢区分：新生児", "年齢区分：乳幼児",
    "年齢区分：少年", "年齢区分：成人", "年齢区分：高齢者", "年齢区分：不明",
    "傷病程度：死亡", "傷病程度：重症", "傷病程度：中等症", "傷病程度：軽症",
    "傷病程度：その他", "発生場所：住居", "発生場所：仕事場①", "発生場所：仕事場②",
    "発生場所：教育機関", "発生場所：公衆(屋内)", "発生場所：公衆(屋外)",
    "発生場所：道路", "発生場所：その他",
)
COUNT_COLUMNS = (
    "transports", "age_newborn", "age_infant", "age_child", "age_adult",
    "age_elderly", "age_unknown", "severity_death", "severity_severe",
    "severity_moderate", "severity_mild", "severity_other", "place_residence",
    "place_work_1", "place_work_2", "place_school", "place_public_indoor",
    "place_public_outdoor", "place_road", "place_other",
)
FIELDS = ("date_jst", "prefecture_code", "prefecture_name_jp", *COUNT_COLUMNS)
SCHEMA = pa.schema(
    [("date_jst", pa.date32()), ("prefecture_code", pa.string()),
     ("prefecture_name_jp", pa.string()),
     *((column, pa.int32()) for column in COUNT_COLUMNS)]
)


def _download(url: str, path: Path, refresh: bool) -> str:
    if not path.exists() or refresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as target, urlopen(
                Request(url, headers={"User-Agent": "liljegren-wbgt-analysis/1.0"}),
                timeout=60,
            ) as response:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 30_000_000:
                        raise ValueError(f"FDMA download exceeds 30 MB: {url}")
                    target.write(chunk)
            if not zipfile.is_zipfile(name):
                raise ValueError(f"FDMA response is not an Excel workbook: {url}")
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Cached file is not an Excel workbook: {path}; use --refresh")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    if not normalized.isdecimal() or not 1 <= int(normalized) <= 47:
        raise ValueError(f"Invalid FDMA prefecture code: {value!r}")
    return f"{int(normalized):02d}"


def _prefectures(path: Path) -> dict[str, str]:
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        if book.sheetnames != ["都道府県コード"]:
            raise ValueError("FDMA prefecture code sheet changed")
        rows = book.active.iter_rows(values_only=True)
        next(rows)
        names = {}
        for row in rows:
            code = _code(row[0])
            name = row[1]
            if code in names or not isinstance(name, str) or not name.strip():
                raise ValueError(f"Invalid or duplicate prefecture name for code {code}")
            names[code] = name.strip()
        if set(names) != {f"{value:02d}" for value in range(1, 48)}:
            raise ValueError("FDMA code table does not contain all 47 prefectures")
        return names
    finally:
        book.close()


def _count(value: object, sheet: str, row: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid count in {sheet} row {row}: {value!r}")
    return value


def _consolidate(path: Path, names: dict[str, str]) -> tuple[list[dict[str, object]], dict[int, int]]:
    book = load_workbook(path, read_only=True, data_only=True)
    records: list[dict[str, object]] = []
    month_totals = {}
    try:
        expected_sheets = [f"2025_{month:02d}" for month in EXPECTED_MONTHS]
        if book.sheetnames != expected_sheets:
            raise ValueError(f"FDMA data sheets changed: {book.sheetnames}")
        for month, sheet in zip(EXPECTED_MONTHS, book, strict=True):
            values = sheet.iter_rows(values_only=True)
            headers = tuple(str(item).strip() for item in next(values))
            if headers != SOURCE_HEADERS:
                raise ValueError(f"FDMA columns changed in {sheet.title}")
            observed = set()
            total = 0
            for row_number, source in enumerate(values, start=2):
                stamp, raw_code, *raw_counts = source
                if not isinstance(stamp, (date, datetime)):
                    raise TypeError(f"Invalid date in {sheet.title} row {row_number}")
                day = stamp.date() if isinstance(stamp, datetime) else stamp
                if day.year != 2025 or day.month != month:
                    raise ValueError(f"Date outside {sheet.title} in row {row_number}")
                code = _code(raw_code)
                identity = (day, code)
                if identity in observed:
                    raise ValueError(f"Duplicate FDMA row: {identity}")
                observed.add(identity)
                counts = [_count(item, sheet.title, row_number) for item in raw_counts]
                if len(counts) != len(COUNT_COLUMNS):
                    raise ValueError(f"FDMA count columns changed in {sheet.title}")
                if any(sum(counts[start:stop]) != counts[0] for start, stop in ((1, 7), (7, 12), (12, 20))):
                    raise ValueError(f"FDMA category totals differ in {sheet.title} row {row_number}")
                records.append({"date_jst": day, "prefecture_code": code,
                                "prefecture_name_jp": names[code],
                                **dict(zip(COUNT_COLUMNS, counts, strict=True))})
                total += counts[0]
            expected = {
                (date(2025, month, day), code)
                for day in range(1, calendar.monthrange(2025, month)[1] + 1)
                for code in names
            }
            if observed != expected:
                raise ValueError(f"Missing or unexpected prefecture-days in {sheet.title}")
            if total != EXPECTED_MONTH_TOTALS[month]:
                raise ValueError(f"FDMA monthly total differs in {sheet.title}: {total}")
            month_totals[month] = total
    finally:
        book.close()
    if sum(month_totals.values()) != EXPECTED_TOTAL:
        raise ValueError("FDMA 2025 national total differs")
    records.sort(key=lambda item: (item["date_jst"], item["prefecture_code"]))
    return records, month_totals


def _publish(path: Path, writer) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as target:
            writer(target)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def run(output: Path, refresh: bool) -> None:
    output = output.expanduser().resolve()
    raw = output / "raw"
    data_path, code_path = raw / "heatstroke003_data_r7.xlsx", raw / "heatstroke003_code.xlsx"
    data_sha = _download(DATA_URL, data_path, refresh)
    code_sha = _download(CODE_URL, code_path, refresh)
    names = _prefectures(code_path)
    records, month_totals = _consolidate(data_path, names)
    output.mkdir(parents=True, exist_ok=True)

    def write_csv(stream) -> None:
        import io

        text = io.TextIOWrapper(stream, encoding="utf-8", newline="", write_through=True)
        writer = csv.DictWriter(text, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
        text.flush()
        text.detach()

    _publish(output / "prefecture_daily.csv", write_csv)
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    _publish(output / "prefecture_daily.parquet", lambda stream: pq.write_table(table, stream, compression="zstd"))
    manifest = {
        "source_page": SOURCE_PAGE,
        "annual_report": ANNUAL_REPORT,
        "data_url": DATA_URL,
        "data_sha256": data_sha,
        "code_url": CODE_URL,
        "code_sha256": code_sha,
        "date_basis": "Japan local calendar date",
        "period": ["2025-05-01", "2025-09-30"],
        "rows": len(records),
        "prefectures": len(names),
        "monthly_transports": {f"2025-{month:02d}": total for month, total in month_totals.items()},
        "total_transports": EXPECTED_TOTAL,
    }
    _publish(output / "manifest.json", lambda stream: stream.write((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")))
    print(f"Saved {len(records):,} prefecture-days and {EXPECTED_TOTAL:,} transports to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fdma/2025"))
    parser.add_argument("--refresh", action="store_true", help="Download both official workbooks again.")
    args = parser.parse_args()
    try:
        run(args.output_dir, args.refresh)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
