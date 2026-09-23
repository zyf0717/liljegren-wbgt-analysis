"""Join 2025 Korean or Taiwanese regional outcomes to local-day Argonne exposure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl

from liljegren_analysis.io.artifacts import atomic_json, atomic_parquet
from heat_health_common import SPECS


def region_hourly_components(country: str, hourly_path: Path, weights_path: Path) -> pl.LazyFrame:
    spec = SPECS[country]
    weights = pl.scan_parquet(weights_path).select("source_cell_id", "region_code", "population_weight")
    totals = weights.group_by("region_code").agg(pl.col("population_weight").sum().alias("region_weight_total"))
    source = (
        pl.scan_parquet(hourly_path)
        .select("timestamp_utc", "row_id", "wbgt_status", "tg_status", "tnwb_status",
                "tnwb_c", "tg_c", "ta_c")
        .filter((pl.col("wbgt_status") == "ok") & (pl.col("tg_status") == "ok")
                & (pl.col("tnwb_status") == "ok")
                & pl.all_horizontal(*(pl.col(name).is_finite() for name in ("tnwb_c", "tg_c", "ta_c"))))
        .with_columns(
            pl.col("row_id").str.slice(0, 28).alias("source_cell_id"),
            pl.col("timestamp_utc").dt.convert_time_zone(spec.timezone).dt.date().alias("local_date"),
        )
        .filter(pl.col("local_date").is_between(spec.first_day, spec.last_day))
        .select("timestamp_utc", "local_date", "source_cell_id", "tnwb_c", "tg_c", "ta_c")
    )
    return (
        source.join(weights, on="source_cell_id", how="inner")
        .group_by("region_code", "local_date", "timestamp_utc")
        .agg(
            *(((pl.col(name).cast(pl.Float64) * pl.col("population_weight")).sum()
               / pl.col("population_weight").sum()).alias(name)
              for name in ("tnwb_c", "tg_c", "ta_c")),
            pl.col("population_weight").sum().alias("covered_population_weight"),
        )
        .join(totals, on="region_code")
        .with_columns((pl.col("covered_population_weight") / pl.col("region_weight_total"))
                      .alias("population_coverage"))
    )


def join_daily(country: str, hourly_path: Path, weights_path: Path, health_path: Path,
               output: Path, component_weights: tuple[float, float, float],
               threshold_c: float, min_coverage: float) -> None:
    spec = SPECS[country]
    wn, wg, wa = component_weights
    if any(not math.isfinite(x) or x < 0 for x in component_weights) or not math.isclose(sum(component_weights), 1, abs_tol=1e-9):
        raise ValueError("Component weights must be finite, nonnegative, and sum to 1")
    if not math.isfinite(threshold_c) or not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("Invalid threshold or minimum coverage")
    hourly = region_hourly_components(country, hourly_path, weights_path).with_columns(
        (wn * pl.col("tnwb_c") + wg * pl.col("tg_c") + wa * pl.col("ta_c")).alias("hw_hour_c")
    )
    daily = hourly.group_by("region_code", "local_date").agg(
        pl.col("hw_hour_c").max().alias("hw_daily_max_c"),
        pl.col("hw_hour_c").mean().alias("hw_daily_mean_c"),
        (pl.col("hw_hour_c") >= threshold_c).sum().alias("hw_hours_ge_threshold"),
        pl.len().alias("valid_hours"),
        pl.col("population_coverage").min().alias("min_hourly_population_coverage"),
        pl.col("region_weight_total").first().alias("region_weight_total"),
    )
    panel = (pl.scan_parquet(health_path).join(daily, on=["region_code", "local_date"], how="left")
             .with_columns(pl.col("valid_hours").fill_null(0))
             .sort("local_date", "region_code").collect(engine="streaming"))
    expected = len(spec.names) * ((spec.last_day - spec.first_day).days + 1)
    if panel.height != expected or panel.select("local_date", "region_code").unique().height != expected:
        raise ValueError(f"{country} outcome panel must contain all {expected} region-days")
    days = (spec.last_day - spec.first_day).days + 1
    if (set(panel["region_code"].to_list()) != set(spec.names.values())
            or panel["local_date"].min() != spec.first_day
            or panel["local_date"].max() != spec.last_day
            or panel.group_by("region_code").len()["len"].to_list().count(days) != len(spec.names)):
        raise ValueError(f"{country} outcome panel has missing or unexpected region dates")
    if panel.filter(pl.col("outcome_count") < 0).height:
        raise ValueError("Negative outcome count")
    if panel.filter(pl.col("min_hourly_population_coverage") > 1.001).height:
        raise ValueError("Population coverage exceeds 100%")
    complete = ((pl.col("valid_hours") == 24) &
                (pl.col("min_hourly_population_coverage") >= min_coverage)).fill_null(False)
    panel = panel.with_columns(
        complete.alias("exposure_complete"),
        *(pl.when(complete).then(pl.col(name)).otherwise(None).alias(name)
          for name in ("hw_daily_max_c", "hw_daily_mean_c", "hw_hours_ge_threshold")),
    )
    atomic_parquet(panel.to_arrow(), output)
    health_manifest = health_path.with_name("manifest.json")
    source = json.loads(health_manifest.read_text()) if health_manifest.exists() else {}
    atomic_json({"country": country, "source_hourly": str(hourly_path.resolve()),
                 "source_weights": str(weights_path.resolve()), "source_health": str(health_path.resolve()),
                 "health_raw_sha256": source.get("raw_sha256"), "timezone": spec.timezone,
                 "h_w_weights": {"tnwb": wn, "tg": wg, "ta": wa}, "threshold_c": threshold_c,
                 "min_hourly_population_coverage": min_coverage,
                 "coverage_denominator": "all ADM1 boundary weighted LandScan population, including cells absent from HSB",
                 "rows": panel.height, "outcome_count": int(panel["outcome_count"].sum()),
                 "eligible_rows": int(panel["exposure_complete"].sum()),
                 "eligible_regions": sorted(panel.filter(pl.col("exposure_complete"))["region_code"].unique().to_list()),
                 "regions_without_exposure": sorted(panel.filter(pl.col("valid_hours") == 0)["region_code"].unique().to_list())},
                output.with_suffix(".json"))
    print(f"Saved {panel.height} region-days ({int(panel['exposure_complete'].sum())} eligible) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True, choices=SPECS)
    parser.add_argument("--hourly", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--health", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wn", type=float, default=.7)
    parser.add_argument("--wg", type=float, default=.2)
    parser.add_argument("--wa", type=float, default=.1)
    parser.add_argument("--threshold-c", type=float, default=28.)
    parser.add_argument("--min-coverage", type=float, default=.95)
    args = parser.parse_args()
    spec = SPECS[args.country]
    join_daily(args.country,
               args.hourly or Path(f"data/country/{args.country.lower()}/argonne-hourly-2025.parquet"),
               args.weights or spec.root / "region_cell_weights.parquet",
               args.health or spec.root / "region_daily.parquet",
               args.output or spec.root / "region_daily_exposure.parquet",
               (args.wn, args.wg, args.wa), args.threshold_c, args.min_coverage)


if __name__ == "__main__":
    main()
