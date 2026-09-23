"""Search WBGT component weights against 2025 FDMA prefecture-day counts.

Run: OPENBLAS_NUM_THREADS=1 uv run --with statsmodels python scripts/search_fdma_wbgt_weights.py
Outputs are written to data/fdma/2025/weight_search by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import patsy
import polars as pl
import statsmodels.api as sm
from scipy.special import xlogy

from liljegren_analysis.io.artifacts import atomic_csv, atomic_json, atomic_parquet
from join_fdma_prefecture_daily import join_daily, prefecture_hourly_components


STANDARD = (70, 20, 10)
AIR_ONLY = (0, 0, 100)
COMPONENTS = ("tnwb_c", "tg_c", "ta_c")


def candidates(step: int, around: tuple[int, int, int] | None = None) -> list[tuple[int, int, int]]:
    """Integer-percent simplex; refinement is within five points in each coordinate."""
    return [
        (wn, wg, 100 - wn - wg)
        for wn in range(0, 101, step)
        for wg in range(0, 101 - wn, step)
        if around is None or max(abs(a - b) for a, b in zip((wn, wg, 100 - wn - wg), around)) <= 5
    ]


def load_panel(hourly_path: Path, weights_path: Path, fdma_path: Path, min_coverage: float):
    hourly = prefecture_hourly_components(hourly_path, weights_path).collect(engine="streaming")
    daily_quality = hourly.group_by("prefecture_code", "date_jst").agg(
        pl.len().alias("valid_hours"),
        pl.col("population_coverage").min().alias("min_coverage"),
    )
    panel = (
        pl.read_parquet(fdma_path)
        .join(daily_quality, on=["prefecture_code", "date_jst"], how="left")
        .filter((pl.col("valid_hours") == 24) & (pl.col("min_coverage") >= min_coverage))
        .sort("date_jst", "prefecture_code")
        .with_row_index("row_idx")
    )
    if panel.height == 0:
        raise ValueError("No complete prefecture-days available")
    selected = hourly.join(
        panel.select("prefecture_code", "date_jst", "row_idx"),
        on=["prefecture_code", "date_jst"], how="inner",
    ).sort("row_idx", "timestamp_utc")
    day_idx = selected["row_idx"].to_numpy().astype(np.intp)
    if len(day_idx) != 24 * panel.height or not np.array_equal(np.bincount(day_idx, minlength=panel.height), np.full(panel.height, 24)):
        raise ValueError("Eligible prefecture-days must each have exactly 24 hours")
    values = selected.select(COMPONENTS).to_numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite hourly component means")
    return panel, day_idx, values


def base_design(panel: pl.DataFrame) -> np.ndarray:
    dates = panel["date_jst"].to_list()
    prefectures = panel["prefecture_code"].to_list()
    prefecture_levels = sorted(set(prefectures))
    t = np.array([(day.toordinal() - dates[0].toordinal()) / 76.0 - 1.0 for day in dates])
    columns = [np.ones(len(dates))]
    columns.extend(np.array([code == level for code in prefectures], dtype=float) for level in prefecture_levels[1:])
    columns.extend(np.array([day.weekday() == weekday for day in dates], dtype=float) for weekday in range(1, 7))
    columns.extend((t, t * t))
    return np.column_stack(columns)


def daily_exposure(day_idx: np.ndarray, components: np.ndarray, weights: tuple[int, int, int],
                   n_days: int, measure: str) -> np.ndarray:
    hourly = components @ (np.asarray(weights, dtype=float) / 100.0)
    if measure == "max":
        result = np.full(n_days, -np.inf)
        np.maximum.at(result, day_idx, hourly)
        return result
    return np.bincount(day_idx, weights=hourly, minlength=n_days) / 24.0


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    if not np.isfinite(mu).all() or np.any(mu <= 0):
        raise ValueError("Nonpositive or nonfinite model prediction")
    return 2.0 * (xlogy(y, y / mu) - y + mu)


def cross_validate(y: np.ndarray, base: np.ndarray, months: np.ndarray,
                   exposure: np.ndarray) -> tuple[float, dict[int, float], np.ndarray]:
    predictions = np.full(len(y), np.nan)
    fold_scores = {}
    for month in sorted(set(months)):
        train = months != month
        test = ~train
        x_train = exposure[train]
        knots = np.quantile(x_train, [0.25, 0.5, 0.75]).tolist()
        spline_train = patsy.dmatrix("cr(x, knots=knots, constraints='center') - 1",
                                     {"x": x_train, "knots": knots})
        spline_test = patsy.build_design_matrices([spline_train.design_info], {"x": exposure[test]})[0]
        design_train = np.column_stack((base[train], np.asarray(spline_train)))
        design_test = np.column_stack((base[test], np.asarray(spline_test)))
        fit = sm.GLM(y[train], design_train, family=sm.families.Poisson()).fit(maxiter=60, tol=1e-8, disp=0)
        if not fit.converged:
            raise RuntimeError(f"Poisson fit did not converge for held-out month {month}")
        predictions[test] = fit.predict(design_test)
        fold_scores[int(month)] = float(poisson_deviance(y[test], predictions[test]).mean())
    return float(poisson_deviance(y, predictions).mean()), fold_scores, predictions


def run_search(panel: pl.DataFrame, day_idx: np.ndarray, components: np.ndarray,
               measure: str) -> tuple[list[dict], dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = panel["transports"].to_numpy().astype(float)
    months = np.array([day.month for day in panel["date_jst"].to_list()])
    base = base_design(panel)
    scores = []
    cache = {}

    def evaluate(weight: tuple[int, int, int], phase: str) -> None:
        if weight in cache:
            return
        exposure = daily_exposure(day_idx, components, weight, panel.height, measure)
        score, folds, predictions = cross_validate(y, base, months, exposure)
        record = {
            "measure": measure, "phase": phase,
            "wn": weight[0] / 100, "wg": weight[1] / 100, "wa": weight[2] / 100,
            "mean_poisson_deviance": score,
            **{f"fold_{month}_deviance": folds[month] for month in sorted(folds)},
        }
        scores.append(record)
        cache[weight] = predictions

    coarse = candidates(5)
    for number, weight in enumerate(coarse, 1):
        evaluate(weight, "coarse")
        if number % 25 == 0 or number == len(coarse):
            print(f"{measure}: evaluated {number}/{len(coarse)} coarse weights", flush=True)
    best_coarse = min(scores, key=lambda row: row["mean_poisson_deviance"])
    center = tuple(round(100 * best_coarse[key]) for key in ("wn", "wg", "wa"))
    refine = candidates(1, center)
    for number, weight in enumerate(refine, 1):
        evaluate(weight, "refine")
        if number % 25 == 0 or number == len(refine):
            print(f"{measure}: evaluated {number}/{len(refine)} refinement weights", flush=True)
    scores.sort(key=lambda row: row["mean_poisson_deviance"])
    best = scores[0]
    selected = tuple(round(100 * best[key]) for key in ("wn", "wg", "wa"))
    weights = {"standard": STANDARD, "air_only": AIR_ONLY, "selected": selected}
    exposures = {label: daily_exposure(day_idx, components, weight, panel.height, measure)
                 for label, weight in weights.items()}
    predictions = {label: cache[weight] for label, weight in weights.items()}
    return scores, best, predictions, exposures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", type=Path, default=Path("data/country/jpn/argonne-hourly-2025.parquet"))
    parser.add_argument("--weights", type=Path, default=Path("data/fdma/2025/prefecture_cell_weights.parquet"))
    parser.add_argument("--fdma", type=Path, default=Path("data/fdma/2025/prefecture_daily.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fdma/2025/weight_search"))
    parser.add_argument("--min-coverage", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be in (0, 1]")
    panel, day_idx, components = load_panel(args.hourly, args.weights, args.fdma, args.min_coverage)
    if args.min_coverage == 0.95 and (panel.height != 6_892 or int(panel["transports"].sum()) != 98_755):
        raise ValueError("Eligible 2025 sample differs from the established join; inspect source data and coverage")
    all_scores = []
    all_predictions = []
    summary = {
        "source_hourly": str(args.hourly.resolve()), "source_weights": str(args.weights.resolve()),
        "source_fdma": str(args.fdma.resolve()), "min_coverage": args.min_coverage,
        "eligible_prefecture_days": panel.height, "eligible_transports": int(panel["transports"].sum()),
        "validation": "leave-one-calendar-month-out, same Japan date across all prefectures",
        "model": "Poisson GLM; prefecture intercepts, weekday, quadratic season trend, natural cubic exposure spline",
        "score": "mean held-out Poisson deviance per prefecture-day; smaller is better",
        "search": "0.05 global simplex grid, then 0.01 grid within 0.05 of best coarse weight in each coordinate",
        "measures": {},
    }
    for measure in ("max", "mean"):
        scores, best, predictions, exposures = run_search(
            panel, day_idx, components, measure
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
        for label, predicted in predictions.items():
            all_predictions.append(panel.select("date_jst", "prefecture_code", "transports").with_columns(
                pl.lit(measure).alias("measure"), pl.lit(label).alias("weights_type"),
                pl.Series("exposure_c", exposures[label]), pl.Series("predicted_transports", predicted),
            ))
        print(f"{measure}: best {best['wn']:.2f}/{best['wg']:.2f}/{best['wa']:.2f}; "
              f"deviance {best['mean_poisson_deviance']:.4f} vs standard {baseline['mean_poisson_deviance']:.4f} "
              f"vs air-only {air_only['mean_poisson_deviance']:.4f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pl.DataFrame(all_scores).to_arrow(), args.output_dir / "candidate_scores.csv")
    atomic_parquet(pl.concat(all_predictions).to_arrow(), args.output_dir / "heldout_predictions.parquet")
    atomic_json(summary, args.output_dir / "summary.json")
    winner = summary["measures"]["max"]["best_weights"]
    join_daily(args.hourly, args.weights, args.fdma, args.output_dir / "prefecture_daily_exposure_optimized.parquet",
               (winner["wn"], winner["wg"], winner["wa"]), 28.0, args.min_coverage)


if __name__ == "__main__":
    main()
