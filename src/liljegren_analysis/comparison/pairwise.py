"""Polars-native daily pairwise summaries over HSB result Parquet files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from time import monotonic

import polars as pl
import pyarrow as pa

from ..results import HSBRunResults, RESULT_QUANTITIES
from .area_weighted import _ERA5_LATITUDE_WEIGHTS
from .config import EXPECTED_IMPLEMENTATIONS, SAMPLE_SCOPES, SampleScope
from .persistence import persist_date_partitions


PAIRWISE_COLUMNS = (
    "result_date", "quantity", "implementation_a", "implementation_b", "n",
    "sum_difference_c", "sum_squared_difference_c2", "sum_absolute_difference_c",
    "min_difference_c", "max_difference_c", "max_absolute_difference_c",
    "area_weight_sum", "area_weighted_difference_sum_c",
    "area_weighted_squared_difference_sum_c2", "area_weighted_absolute_difference_sum_c",
    "n_absolute_difference_gt_0_1_c", "n_absolute_difference_gt_0_2_c",
    "n_absolute_difference_gt_0_5_c", "area_weight_absolute_difference_gt_0_1_c",
    "area_weight_absolute_difference_gt_0_2_c", "area_weight_absolute_difference_gt_0_5_c",
)
ABS_HISTOGRAM_COLUMNS = (
    "result_date", "quantity", "implementation_a", "implementation_b", "abs_bin_id",
    "abs_bin_lower_c", "abs_bin_upper_c", "n", "area_weight_sum",
)
QUANTITY_STATUS_COLUMNS = (
    "result_date", "quantity", "implementation", "status", "value_present", "n", "area_weight_sum",
)


@dataclass(frozen=True)
class DailyPairwiseSummaries:
    pairwise: pa.Table
    abs_histogram: pa.Table
    quantity_status: pa.Table


def compute_daily_pairwise(
    result_date: date, runs: tuple[Path, ...], sample_scope: SampleScope = "pairwise"
) -> tuple[date, DailyPairwiseSummaries, int, float]:
    """Compute one date's pairwise and status summaries with Polars LazyFrames."""

    if not runs:
        raise ValueError("runs is empty")
    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    started = monotonic()
    base = _base_results(result_date, runs)
    invalid = base.filter(pl.col("area_weight").is_null()).select("row_id").limit(1).collect()
    if invalid.height:
        raise ValueError(f"Invalid ERA5 row_id: {invalid.item(0, 0)}")
    values = _pairwise_values(base, sample_scope)
    pairwise = _pairwise_summary(values).collect().sort(
        ["quantity", "implementation_a", "implementation_b"]
    )
    if sample_scope == "all-implementations":
        expected = len(RESULT_QUANTITIES) * len(EXPECTED_IMPLEMENTATIONS) * (len(EXPECTED_IMPLEMENTATIONS) - 1) // 2
        if pairwise.height != expected:
            raise ValueError(
                f"Expected {expected} implementation pairs for {result_date}, found "
                f"{pairwise.height}; no complete seven-implementation population exists"
            )
    histogram = _histogram_summary(values).collect().sort(
        ["quantity", "implementation_a", "implementation_b", "abs_bin_id"]
    )
    status = _status_summary(base).collect().sort(
        ["quantity", "implementation", "status", "value_present"]
    )
    return result_date, DailyPairwiseSummaries(
        pairwise.select(PAIRWISE_COLUMNS).to_arrow(),
        histogram.select(ABS_HISTOGRAM_COLUMNS).to_arrow(),
        status.select(QUANTITY_STATUS_COLUMNS).to_arrow(),
    ), os.getpid(), monotonic() - started


def persist_daily_pairwise(
    output_root: str | Path,
    result_date: date,
    summaries: DailyPairwiseSummaries,
    sample_scope: SampleScope = "pairwise",
) -> Path:
    """Atomically publish one date's three Parquet summary tables."""

    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    return persist_date_partitions(
        Path(output_root), result_date,
        {"pairwise": summaries.pairwise, "abs_histogram": summaries.abs_histogram, "quantity_status": summaries.quantity_status},
        sample_scope,
    )


def _base_results(result_date: date, runs: tuple[Path, ...]) -> pl.LazyFrame:
    _validate_nonoverlapping_scopes(result_date, runs)
    files = tuple(
        item for run in runs
        for item in HSBRunResults.from_hsb_run(run).files(result_date=result_date)
    )
    if not files:
        raise FileNotFoundError(f"No result shards found for {result_date.isoformat()}")
    scans = [
        pl.scan_parquet(item.path).select(
            "cycle_ordinal", "row_id", "row_ordinal",
            "wbgt_status", "wbgt_c", "tg_status", "tg_c", "tnwb_status", "tnwb_c",
        ).with_columns(
            pl.lit(item.result_date).cast(pl.Date).alias("result_date"),
            pl.lit(item.run_id).alias("run_id"),
            pl.lit(item.implementation).alias("implementation"),
        )
        for item in files
    ]
    weights = pl.from_arrow(_ERA5_LATITUDE_WEIGHTS).lazy()
    return pl.concat(scans).with_columns(
        pl.col("row_id").str.slice(12, 7).alias("latitude_code")
    ).join(weights, on="latitude_code", how="left").drop("latitude_code")


def _joint_valid(base: pl.LazyFrame) -> pl.LazyFrame:
    return base.filter(pl.all_horizontal(
        *[
            (pl.col(f"{quantity}_status") == "ok") & pl.col(f"{quantity}_c").is_not_null()
            for quantity in RESULT_QUANTITIES
        ]
    ))


def _pairwise_values(base: pl.LazyFrame, scope: SampleScope) -> pl.LazyFrame:
    joint = _joint_valid(base)
    identity = ("result_date", "run_id", "cycle_ordinal", "row_id")
    if scope == "pairwise":
        right = joint.rename({
            "implementation": "implementation_b", "area_weight": "area_weight_b",
            **{f"{quantity}_c": f"{quantity}_c_b" for quantity in RESULT_QUANTITIES},
        })
        wide = joint.join(right, on=identity, how="inner").filter(
            pl.col("implementation") < pl.col("implementation_b")
        )
        frames = [
            wide.select(
                "result_date", pl.lit(quantity).alias("quantity"),
                pl.col("implementation").alias("implementation_a"), "implementation_b", "area_weight",
                (pl.col(f"{quantity}_c") - pl.col(f"{quantity}_c_b")).alias("difference_c"),
            )
            for quantity in RESULT_QUANTITIES
        ]
        return pl.concat(frames)

    by_implementation: dict[str, pl.LazyFrame] = {}
    for implementation in EXPECTED_IMPLEMENTATIONS:
        grouped = joint.filter(pl.col("implementation") == implementation).group_by(*identity).agg(
            pl.len().alias("_n"), pl.col("area_weight").first().alias("area_weight"),
            *[pl.col(f"{quantity}_c").first().alias(f"{implementation}_{quantity}") for quantity in RESULT_QUANTITIES],
        ).filter(pl.col("_n") == 1).drop("_n")
        by_implementation[implementation] = grouped
    first, *rest = EXPECTED_IMPLEMENTATIONS
    wide = by_implementation[first]
    for implementation in rest:
        wide = wide.join(
            by_implementation[implementation].drop("area_weight"), on=identity, how="inner"
        )
    return pl.concat([
        wide.select(
            "result_date", pl.lit(quantity).alias("quantity"),
            pl.lit(left).alias("implementation_a"), pl.lit(right).alias("implementation_b"), "area_weight",
            (pl.col(f"{left}_{quantity}") - pl.col(f"{right}_{quantity}")).alias("difference_c"),
        )
        for quantity in RESULT_QUANTITIES
        for left, right in combinations(EXPECTED_IMPLEMENTATIONS, 2)
    ])


def _pairwise_summary(values: pl.LazyFrame) -> pl.LazyFrame:
    difference = pl.col("difference_c")
    absolute = difference.abs()
    weight = pl.col("area_weight")
    return values.group_by("result_date", "quantity", "implementation_a", "implementation_b").agg(
        pl.len().alias("n"), difference.sum().alias("sum_difference_c"),
        (difference * difference).sum().alias("sum_squared_difference_c2"), absolute.sum().alias("sum_absolute_difference_c"),
        difference.min().alias("min_difference_c"), difference.max().alias("max_difference_c"), absolute.max().alias("max_absolute_difference_c"),
        weight.sum().alias("area_weight_sum"), (weight * difference).sum().alias("area_weighted_difference_sum_c"),
        (weight * difference * difference).sum().alias("area_weighted_squared_difference_sum_c2"),
        (weight * absolute).sum().alias("area_weighted_absolute_difference_sum_c"),
        *[
            expression
            for threshold in (0.1, 0.2, 0.5)
            for expression in _threshold_aggregates(absolute, weight, threshold)
        ],
    )


def _threshold_aggregates(absolute: pl.Expr, weight: pl.Expr, threshold: float) -> list[pl.Expr]:
    label = str(threshold).replace(".", "_")
    selected = absolute > threshold
    return [
        selected.sum().alias(f"n_absolute_difference_gt_{label}_c"),
        pl.when(selected).then(weight).otherwise(0.0).sum().alias(
            f"area_weight_absolute_difference_gt_{label}_c"
        ),
    ]


def _histogram_summary(values: pl.LazyFrame) -> pl.LazyFrame:
    absolute = pl.col("difference_c").abs()
    bin_id = pl.when(absolute < 2).then((absolute / 0.01).floor()).when(
        absolute < 5
    ).then(200 + ((absolute - 2) / 0.05).floor()).when(absolute < 20).then(
        260 + ((absolute - 5) / 0.25).floor()
    ).when(absolute < 100).then(320 + (absolute - 20).floor()).otherwise(400).cast(pl.Int32)
    binned = values.with_columns(bin_id.alias("abs_bin_id"))
    lower = pl.when(pl.col("abs_bin_id") < 200).then(pl.col("abs_bin_id") * 0.01).when(
        pl.col("abs_bin_id") < 260
    ).then(2 + (pl.col("abs_bin_id") - 200) * 0.05).when(pl.col("abs_bin_id") < 320).then(
        5 + (pl.col("abs_bin_id") - 260) * 0.25
    ).when(pl.col("abs_bin_id") < 400).then(20 + (pl.col("abs_bin_id") - 320)).otherwise(100.0)
    upper = pl.when(pl.col("abs_bin_id") < 200).then((pl.col("abs_bin_id") + 1) * 0.01).when(
        pl.col("abs_bin_id") < 260
    ).then(2 + (pl.col("abs_bin_id") - 199) * 0.05).when(pl.col("abs_bin_id") < 320).then(
        5 + (pl.col("abs_bin_id") - 259) * 0.25
    ).when(pl.col("abs_bin_id") < 400).then(21 + (pl.col("abs_bin_id") - 320)).otherwise(None)
    return binned.group_by("result_date", "quantity", "implementation_a", "implementation_b", "abs_bin_id").agg(
        pl.len().alias("n"), pl.col("area_weight").sum().alias("area_weight_sum"),
    ).with_columns(lower.alias("abs_bin_lower_c"), upper.alias("abs_bin_upper_c"))


def _status_summary(base: pl.LazyFrame) -> pl.LazyFrame:
    return pl.concat([
        base.select(
            "result_date", pl.lit(quantity).alias("quantity"), "implementation",
            pl.col(f"{quantity}_status").alias("status"),
            pl.col(f"{quantity}_c").is_not_null().alias("value_present"), "area_weight",
        )
        for quantity in RESULT_QUANTITIES
    ]).group_by("result_date", "quantity", "implementation", "status", "value_present").agg(
        pl.len().alias("n"), pl.col("area_weight").sum().alias("area_weight_sum")
    )


def _validate_nonoverlapping_scopes(result_date: date, runs: tuple[Path, ...]) -> None:
    scopes: dict[tuple[str, str], Path] = {}
    for run in runs:
        for item in HSBRunResults.from_hsb_run(run).files(result_date=result_date):
            scope = (item.run_id, item.implementation)
            previous = scopes.setdefault(scope, item.path)
            if previous != item.path:
                raise ValueError(
                    f"Overlapping result scope for {result_date} run_id={item.run_id} "
                    f"implementation={item.implementation}: {previous} and {item.path}"
                )
