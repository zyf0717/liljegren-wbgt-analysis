"""Aggregate hourly Japan Argonne components to prefecture days and join FDMA counts.

Run: uv run python scripts/join_fdma_prefecture_daily.py
The primary exposure is the daily maximum of population-weighted hourly H_w.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from liljegren_analysis.io.artifacts import atomic_json, atomic_parquet


def prefecture_hourly_components(hourly_path: Path, weights_path: Path) -> pl.LazyFrame:
    """Population-weighted component means on a common valid-cell mask."""
    weights = pl.scan_parquet(weights_path).select("source_cell_id", "prefecture_code", "population_weight")
    totals = weights.group_by("prefecture_code").agg(
        pl.col("population_weight").sum().alias("prefecture_weight_total")
    )
    source = (
        pl.scan_parquet(hourly_path)
        .select("timestamp_utc", "row_id", "wbgt_status", "tg_status", "tnwb_status",
                "tnwb_c", "tg_c", "ta_c")
        .filter(
            (pl.col("wbgt_status") == "ok") & (pl.col("tg_status") == "ok")
            & (pl.col("tnwb_status") == "ok")
            & pl.all_horizontal(*(pl.col(column).is_finite() for column in ("tnwb_c", "tg_c", "ta_c")))
        )
        .with_columns(
            pl.col("row_id").str.slice(0, 28).alias("source_cell_id"),
            pl.col("timestamp_utc").dt.convert_time_zone("Asia/Tokyo").dt.date().alias("date_jst"),
        )
        .filter(pl.col("date_jst").is_between(pl.date(2025, 5, 1), pl.date(2025, 9, 30)))
        .select("timestamp_utc", "date_jst", "source_cell_id", "tnwb_c", "tg_c", "ta_c")
    )
    return (
        source.join(weights, on="source_cell_id", how="inner")
        .group_by("prefecture_code", "date_jst", "timestamp_utc")
        .agg(
            *(((pl.col(column).cast(pl.Float64) * pl.col("population_weight")).sum()
               / pl.col("population_weight").sum()).alias(column)
              for column in ("tnwb_c", "tg_c", "ta_c")),
            pl.col("population_weight").sum().alias("covered_population_weight"),
        )
        .join(totals, on="prefecture_code")
        .with_columns(
            (pl.col("covered_population_weight") / pl.col("prefecture_weight_total"))
            .alias("population_coverage")
        )
    )


def join_daily(
    hourly_path: Path, weights_path: Path, fdma_path: Path, output_path: Path,
    component_weights: tuple[float, float, float], threshold_c: float,
    min_coverage: float,
) -> None:
    wn, wg, wa = component_weights
    if any(not math.isfinite(value) or value < 0 for value in component_weights) or not math.isclose(sum(component_weights), 1.0, abs_tol=1e-9):
        raise ValueError("Component weights must be finite, nonnegative, and sum to 1")
    if not math.isfinite(threshold_c):
        raise ValueError("Threshold must be finite")
    if not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("Minimum hourly population coverage must be in (0, 1]")

    hourly = prefecture_hourly_components(hourly_path, weights_path).with_columns(
        (wn * pl.col("tnwb_c") + wg * pl.col("tg_c") + wa * pl.col("ta_c")).alias("hw_hour_c")
    )
    daily = hourly.group_by("prefecture_code", "date_jst").agg(
        pl.col("hw_hour_c").max().alias("hw_daily_max_c"),
        pl.col("hw_hour_c").mean().alias("hw_daily_mean_c"),
        (pl.col("hw_hour_c") >= threshold_c).sum().alias("hw_hours_ge_threshold"),
        pl.len().alias("valid_hours"),
        pl.col("population_coverage").min().alias("min_hourly_population_coverage"),
        pl.col("prefecture_weight_total").first().alias("prefecture_weight_total"),
    )
    raw_panel = (
        pl.scan_parquet(fdma_path)
        .join(daily, on=["prefecture_code", "date_jst"], how="left")
        .with_columns(pl.col("valid_hours").fill_null(0))
        .sort("date_jst", "prefecture_code")
        .collect(engine="streaming")
    )
    if raw_panel.height != 7_191 or raw_panel["transports"].sum() != 100_510:
        raise ValueError("Joined FDMA row count or transport total differs")
    observed_missing = set(raw_panel.filter(pl.col("hw_daily_max_c").is_null())["prefecture_code"])
    if observed_missing != {"47"}:
        raise ValueError(f"Unexpected prefectures without hourly exposure: {sorted(observed_missing)}")
    if not raw_panel.filter(pl.col("min_hourly_population_coverage") > 1.001).is_empty():
        raise ValueError("Population coverage exceeds 100%; check boundary weights")
    complete = (
        (pl.col("valid_hours") == 24)
        & (pl.col("min_hourly_population_coverage") >= min_coverage)
    ).fill_null(False)
    panel = raw_panel.with_columns(
        complete.alias("exposure_complete"),
        *(pl.when(complete).then(pl.col(column)).otherwise(None).alias(column)
          for column in ("hw_daily_max_c", "hw_daily_mean_c", "hw_hours_ge_threshold")),
    )
    excluded = int(panel.filter((pl.col("prefecture_code") != "47") & ~pl.col("exposure_complete")).height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(panel.to_arrow(), output_path)
    atomic_json({
        "source_hourly": str(hourly_path.resolve()),
        "source_weights": str(weights_path.resolve()),
        "source_fdma": str(fdma_path.resolve()),
        "date_basis": "Japan local calendar date (Asia/Tokyo)",
        "h_w_weights": {"tnwb": wn, "tg": wg, "ta": wa},
        "threshold_c": threshold_c,
        "hourly_spatial_method": "sum(H_w * MLIT-overlap-adjusted LandScan population) / sum(MLIT-overlap-adjusted LandScan population)",
        "daily_method": "maximum and arithmetic mean of 24 hourly prefecture exposures; threshold count uses those hourly exposures",
        "minimum_hourly_population_coverage": min_coverage,
        "quality_excluded_prefecture_days": excluded,
        "rows": panel.height, "fdma_transports": int(panel["transports"].sum()),
        "exposure_prefectures": 46, "missing_exposure_prefecture_codes": ["47"],
    }, output_path.with_suffix(".json"))
    print(f"Saved {panel.height:,} prefecture-days to {output_path}; exposure unavailable for Okinawa (47) and {excluded} low-coverage days")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", type=Path, default=Path("data/country/jpn/argonne-hourly-2025.parquet"))
    parser.add_argument("--weights", type=Path, default=Path("data/fdma/2025/prefecture_cell_weights.parquet"))
    parser.add_argument("--fdma", type=Path, default=Path("data/fdma/2025/prefecture_daily.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/fdma/2025/prefecture_daily_exposure.parquet"))
    parser.add_argument("--wn", type=float, default=0.7)
    parser.add_argument("--wg", type=float, default=0.2)
    parser.add_argument("--wa", type=float, default=0.1)
    parser.add_argument("--threshold-c", type=float, default=28.0)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    args = parser.parse_args()
    join_daily(args.hourly, args.weights, args.fdma, args.output,
               (args.wn, args.wg, args.wa), args.threshold_c, args.min_coverage)


if __name__ == "__main__":
    main()
