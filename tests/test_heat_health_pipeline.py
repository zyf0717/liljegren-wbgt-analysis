"""Source, geography, and hourly quality contracts for the regional pipelines."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_hpa_heat_injury_2025 import parse_csv
from download_kdca_heat_illness_2025 import parse_pdf
from heat_health_common import KOREA_NAMES, TAIWAN_NAMES
from join_region_daily import join_daily, region_hourly_components


def test_kdca_layout_totals_and_drift(monkeypatch):
    lines = []
    for offset in range(134):
        values = [0] * 36
        if offset == 0:
            values[0] = values[2] = 4460
            values[1] = values[3] = 29
        lines.append(f"{date(2025, 5, 15) + timedelta(days=offset)} " + " ".join(map(str, values)))
    text = "\n".join(lines).encode()
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, text, b""))
    rows = parse_pdf(b"%PDF-test")
    assert len(rows) == 2278
    assert rows[0] == {"local_date": date(2025, 5, 15), "region_code": "KR-11",
                       "outcome_count": 4460, "deaths": 29}
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, text.replace(b"4460 29 4460 29", b"4460 29 4459 29"), b""))
    with pytest.raises(ValueError, match="regional total"):
        parse_pdf(b"%PDF-test")


def test_taiwan_big5_aggregation_zero_fill_and_schema():
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("year", "admit_date", "age1", "SEX", "county", "SUM_of_count"))
    writer.writerows(((2025, "2025-07-01", "19-64", "M", "連江縣", 2),
                      (2025, "2025-07-01", "65+", "F", "連江縣", 3),
                      (2024, "2024-07-01", "19-64", "M", "連江縣", 9)))
    rows = parse_csv(stream.getvalue().encode("big5"))
    assert len(rows) == 8030
    assert sum(row["outcome_count"] for row in rows) == 5
    assert next(row for row in rows if row["local_date"] == date(2025, 7, 1) and row["region_code"] == "TW-LIE")["outcome_count"] == 5
    assert next(row for row in rows if row["local_date"] == date(2025, 7, 2) and row["region_code"] == "TW-LIE")["outcome_count"] == 0
    with pytest.raises(ValueError, match="columns changed"):
        parse_csv(b"year,admit_date\n2025,2025-07-01\n")


def test_region_crosswalks_are_complete():
    assert len(set(KOREA_NAMES.values())) == 17
    assert len(set(TAIWAN_NAMES.values())) == 22
    assert TAIWAN_NAMES["連江縣"] == "TW-LIE"
    assert KOREA_NAMES["세종"] == "KR-50"


def test_full_boundary_population_is_coverage_denominator(tmp_path):
    cell = "era5-global-n250000-e1210000"
    stamp = datetime(2025, 1, 1, 16, tzinfo=timezone.utc)
    hourly = tmp_path / "hourly.parquet"
    weights = tmp_path / "weights.parquet"
    pl.DataFrame({
        "timestamp_utc": [stamp], "row_id": [cell + "-2025010116"],
        "wbgt_status": ["ok"], "tg_status": ["ok"], "tnwb_status": ["ok"],
        "tnwb_c": [20.], "tg_c": [30.], "ta_c": [25.],
    }).write_parquet(hourly)
    pl.DataFrame({"source_cell_id": [cell, "era5-global-n252500-e1210000"],
                  "region_code": ["TW-TPE", "TW-TPE"], "population_weight": [50., 50.]}).write_parquet(weights)
    result = region_hourly_components("TWN", hourly, weights).collect()
    assert result["local_date"].item() == date(2025, 1, 2)
    assert result["population_coverage"].item() == .5


def test_daily_join_requires_24_local_hours_and_retains_missing_regions(tmp_path):
    cell = "era5-global-n250000-e1210000"
    stamps = ([datetime(2025, 1, 1, 16, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(24)]
              + [datetime(2025, 1, 2, 16, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(23)]
              + [datetime(2025, 12, 30, 16, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(24)])
    hourly, weights, health, output = (tmp_path / name for name in ("hourly.parquet", "weights.parquet", "health.parquet", "joined.parquet"))
    pl.DataFrame({"timestamp_utc": stamps,
                  "row_id": [f"{cell}-{stamp:%Y%m%d%H}" for stamp in stamps],
                  "wbgt_status": ["ok"] * len(stamps), "tg_status": ["ok"] * len(stamps),
                  "tnwb_status": ["ok"] * len(stamps), "tnwb_c": [20.] * len(stamps),
                  "tg_c": [30.] * len(stamps), "ta_c": [25.] * len(stamps)}).write_parquet(hourly)
    pl.DataFrame({"source_cell_id": [cell], "region_code": ["TW-TPE"],
                  "population_weight": [100.]}).write_parquet(weights)
    first = date(2025, 1, 1)
    pl.DataFrame({"local_date": [first + timedelta(days=i) for i in range(365) for _ in TAIWAN_NAMES],
                  "region_code": [code for _ in range(365) for code in sorted(TAIWAN_NAMES.values())],
                  "outcome_count": [0] * 8030}).write_parquet(health)
    join_daily("TWN", hourly, weights, health, output, (.7, .2, .1), 28., .95)
    panel = pl.read_parquet(output)
    assert panel.height == 8030
    one = panel.filter((pl.col("local_date") == date(2025, 1, 2)) & (pl.col("region_code") == "TW-TPE"))
    assert one["exposure_complete"].item()
    assert one["hw_daily_max_c"].item() == 22.5
    two = panel.filter((pl.col("local_date") == date(2025, 1, 3)) & (pl.col("region_code") == "TW-TPE"))
    assert not two["exposure_complete"].item() and two["hw_daily_max_c"].is_null().all()
    last = panel.filter((pl.col("local_date") == date(2025, 12, 31)) & (pl.col("region_code") == "TW-TPE"))
    assert last["exposure_complete"].item()
    assert not panel.filter(pl.col("local_date") == date(2025, 1, 1))["exposure_complete"].any()
    assert not panel.filter(pl.col("region_code") == "TW-LIE")["exposure_complete"].any()


def test_month_holdout_predictions_exclude_held_out_outcomes():
    pytest.importorskip("statsmodels")
    import numpy as np
    from search_fdma_wbgt_weights import cross_validate

    months = np.repeat(np.arange(1, 13), 10)
    exposure = 20 + np.sin(np.arange(120) / 9) * 4 + np.arange(120) / 120
    outcomes = np.array([2 + (index % 4) for index in range(120)], dtype=float)
    design = np.ones((120, 1))
    _, folds, predictions = cross_validate(outcomes, design, months, exposure)
    revised = outcomes.copy()
    revised[months == 1] += 100
    _, _, revised_predictions = cross_validate(revised, design, months, exposure)
    assert len(folds) == 12
    np.testing.assert_allclose(predictions[months == 1], revised_predictions[months == 1])


def test_japan_extractor_keeps_existing_defaults(monkeypatch):
    pytest.importorskip("netCDF4")
    from extract_japan_hourly_argonne import arguments

    monkeypatch.setattr(sys, "argv", ["extract_japan_hourly_argonne.py"])
    args = arguments()
    assert args.country == "JPN"
    assert args.output == Path("data/country/jpn/argonne-hourly-2025.parquet")
