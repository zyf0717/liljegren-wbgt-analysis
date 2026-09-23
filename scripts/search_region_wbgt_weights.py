"""Search separate 2025 Korean or Taiwanese WBGT weights against daily outcomes.

Run with OPENBLAS_NUM_THREADS=1 uv run --with statsmodels python scripts/search_region_wbgt_weights.py --country KOR|TWN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from liljegren_analysis.io.artifacts import atomic_csv, atomic_json, atomic_parquet
from heat_health_common import SPECS
from join_region_daily import join_daily, region_hourly_components
from search_fdma_wbgt_weights import AIR_ONLY, STANDARD, COMPONENTS, candidates, cross_validate, daily_exposure


def load_panel(country: str, hourly_path: Path, weights_path: Path,
               health_path: Path, min_coverage: float):
    hourly = region_hourly_components(country, hourly_path, weights_path).collect(engine="streaming")
    quality = hourly.group_by("region_code", "local_date").agg(
        pl.len().alias("valid_hours"), pl.col("population_coverage").min().alias("min_coverage")
    )
    panel = (pl.read_parquet(health_path)
             .join(quality, on=["region_code", "local_date"], how="left")
             .filter((pl.col("valid_hours") == 24) & (pl.col("min_coverage") >= min_coverage))
             .sort("local_date", "region_code").with_row_index("row_idx"))
    if panel.is_empty():
        raise ValueError("No complete region-days available")
    selected = hourly.join(panel.select("region_code", "local_date", "row_idx"),
                           on=["region_code", "local_date"], how="inner").sort("row_idx", "timestamp_utc")
    day_idx = selected["row_idx"].to_numpy().astype(np.intp)
    if len(day_idx) != 24 * panel.height or not np.array_equal(
        np.bincount(day_idx, minlength=panel.height), np.full(panel.height, 24)
    ):
        raise ValueError("Eligible region-days must each have exactly 24 hours")
    components = selected.select(COMPONENTS).to_numpy().astype(np.float64)
    if not np.isfinite(components).all():
        raise ValueError("Nonfinite hourly components")
    return panel, day_idx, components


def base_design(country: str, panel: pl.DataFrame) -> np.ndarray:
    dates = panel["local_date"].to_list()
    regions = panel["region_code"].to_list()
    levels = sorted(set(regions))
    columns = [np.ones(len(dates))]
    columns.extend(np.array([code == level for code in regions], dtype=float) for level in levels[1:])
    columns.extend(np.array([day.weekday() == weekday for day in dates], dtype=float) for weekday in range(1, 7))
    if country == "KOR":
        midpoint = (SPECS[country].first_day.toordinal() + SPECS[country].last_day.toordinal()) / 2
        halfspan = (SPECS[country].last_day.toordinal() - SPECS[country].first_day.toordinal()) / 2
        t = np.array([(day.toordinal() - midpoint) / halfspan for day in dates])
        columns.extend((t, t * t))
    else:
        angle = 2 * np.pi * np.array([day.timetuple().tm_yday - 1 for day in dates]) / 365
        columns.extend((np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)))
    return np.column_stack(columns)


def run_search(country: str, panel: pl.DataFrame, day_idx: np.ndarray,
               components: np.ndarray, measure: str):
    y = panel["outcome_count"].to_numpy().astype(float)
    months = np.array([day.month for day in panel["local_date"].to_list()])
    base = base_design(country, panel)
    scores = []
    cache = {}

    def evaluate(weight: tuple[int, int, int], phase: str) -> None:
        if weight in cache:
            return
        exposure = daily_exposure(day_idx, components, weight, panel.height, measure)
        score, folds, predictions = cross_validate(y, base, months, exposure)
        scores.append({"measure": measure, "phase": phase,
                       "wn": weight[0] / 100, "wg": weight[1] / 100, "wa": weight[2] / 100,
                       "mean_poisson_deviance": score,
                       **{f"fold_{month}_deviance": value for month, value in folds.items()}})
        cache[weight] = predictions

    coarse = candidates(5)
    for number, weight in enumerate(coarse, 1):
        evaluate(weight, "coarse")
        if number % 25 == 0 or number == len(coarse):
            print(f"{country} {measure}: {number}/{len(coarse)} coarse weights", flush=True)
    best_coarse = min(scores, key=lambda row: row["mean_poisson_deviance"])
    center = tuple(round(100 * best_coarse[key]) for key in ("wn", "wg", "wa"))
    refine = candidates(1, center)
    for number, weight in enumerate(refine, 1):
        evaluate(weight, "refine")
        if number % 25 == 0 or number == len(refine):
            print(f"{country} {measure}: {number}/{len(refine)} refined weights", flush=True)
    scores.sort(key=lambda row: row["mean_poisson_deviance"])
    best = scores[0]
    selected = tuple(round(100 * best[key]) for key in ("wn", "wg", "wa"))
    weights = {"standard": STANDARD, "air_only": AIR_ONLY, "selected": selected}
    exposures = {label: daily_exposure(day_idx, components, weight, panel.height, measure)
                 for label, weight in weights.items()}
    predictions = {label: cache[weight] for label, weight in weights.items()}
    return scores, best, predictions, exposures


def search(country: str, hourly: Path, weights: Path, health: Path,
           output_dir: Path, min_coverage: float) -> None:
    if not 0 < min_coverage <= 1:
        raise ValueError("Minimum coverage must be in (0, 1]")
    panel, day_idx, components = load_panel(country, hourly, weights, health, min_coverage)
    source_manifest = health.with_name("manifest.json")
    source = json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
    summary = {"country": country, "source_hourly": str(hourly.resolve()),
               "source_weights": str(weights.resolve()), "source_health": str(health.resolve()),
               "health_raw_sha256": source.get("raw_sha256"), "min_coverage": min_coverage,
               "eligible_region_days": panel.height, "eligible_regions": sorted(panel["region_code"].unique().to_list()),
               "eligible_outcomes": int(panel["outcome_count"].sum()),
               "validation": "leave-one-calendar-month-out",
               "model": "Poisson GLM; region intercepts, weekday, exposure spline, " +
                        ("quadratic season trend" if country == "KOR" else "two annual Fourier harmonics"),
               "score": "mean held-out Poisson deviance per region-day; smaller is better",
               "search": "5% simplex grid, then 1% grid within 5 points of the best coarse weight",
               "measures": {}}
    all_scores = []
    predictions = []
    for measure in ("max", "mean"):
        scores, best, fitted_predictions, exposures = run_search(
            country, panel, day_idx, components, measure
        )
        all_scores.extend(scores)
        baseline = next(row for row in scores if tuple(round(100 * row[key]) for key in ("wn", "wg", "wa")) == STANDARD)
        air_only = next(row for row in scores if tuple(round(100 * row[key]) for key in ("wn", "wg", "wa")) == AIR_ONLY)
        summary["measures"][measure] = {
            "best_weights": {key: best[key] for key in ("wn", "wg", "wa")},
            "best_mean_poisson_deviance": best["mean_poisson_deviance"],
            "standard_mean_poisson_deviance": baseline["mean_poisson_deviance"],
            "air_only_mean_poisson_deviance": air_only["mean_poisson_deviance"],
            "improvement_mean_deviance": baseline["mean_poisson_deviance"] - best["mean_poisson_deviance"],
            "candidates_evaluated": len(scores),
        }
        for label, fitted in fitted_predictions.items():
            predictions.append(panel.select("local_date", "region_code", "outcome_count").with_columns(
                pl.lit(measure).alias("measure"), pl.lit(label).alias("weights_type"),
                pl.Series("exposure_c", exposures[label]), pl.Series("predicted_outcome", fitted)))
        print(f"{country} {measure}: best {best['wn']:.2f}/{best['wg']:.2f}/{best['wa']:.2f}; "
              f"deviance {best['mean_poisson_deviance']:.4f} vs standard {baseline['mean_poisson_deviance']:.4f} "
              f"vs air-only {air_only['mean_poisson_deviance']:.4f}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pl.DataFrame(all_scores).to_arrow(), output_dir / "candidate_scores.csv")
    atomic_parquet(pl.concat(predictions).to_arrow(), output_dir / "heldout_predictions.parquet")
    atomic_json(summary, output_dir / "summary.json")
    best = summary["measures"]["max"]["best_weights"]
    join_daily(country, hourly, weights, health, output_dir / "region_daily_exposure_optimized.parquet",
               (best["wn"], best["wg"], best["wa"]), 28., min_coverage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True, choices=SPECS)
    parser.add_argument("--hourly", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--health", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-coverage", type=float, default=.95)
    args = parser.parse_args()
    spec = SPECS[args.country]
    search(args.country,
           args.hourly or Path(f"data/country/{args.country.lower()}/argonne-hourly-2025.parquet"),
           args.weights or spec.root / "region_cell_weights.parquet",
           args.health or spec.root / "region_daily.parquet",
           args.output_dir or spec.root / "weight_search", args.min_coverage)


if __name__ == "__main__":
    main()
