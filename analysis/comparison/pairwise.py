"""Polars-native daily pairwise summaries over HSB result Parquet files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path

import polars as pl
import pyarrow as pa

from ..config import (
    DEFAULT_PAIRWISE_IMPLEMENTATIONS,
    SAMPLE_SCOPES,
    SampleScope,
    implementation_names,
)
from ..grid import LATITUDE_WEIGHT_TABLE
from ..results import RESULT_QUANTITIES, ResultFile
from .distributions import absolute_difference_bins
from .persistence import (
    ABS_HISTOGRAM_COLUMNS,
    PAIRWISE_COLUMNS,
    QUANTITY_STATUS_COLUMNS,
    persist_date_partitions,
)

PAIRWISE_GROUP_COLUMNS = (
    "result_date",
    "quantity",
    "implementation_a",
    "implementation_b",
)
PAIRWISE_MIN_COLUMNS = ("min_difference_c",)
PAIRWISE_MAX_COLUMNS = (
    "max_difference_c",
    "max_absolute_difference_c",
)


@dataclass(frozen=True)
class DailyPairwiseSummaries:
    pairwise: pa.Table
    abs_histogram: pa.Table
    quantity_status: pa.Table


def compute_daily_pairwise(
    result_date: date,
    files: Sequence[ResultFile],
    sample_scope: SampleScope = "pairwise",
    *,
    implementations: Sequence[str] | None = None,
) -> DailyPairwiseSummaries:
    """Compute one date's pairwise and status summaries with Polars LazyFrames."""

    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    selected = (
        implementation_names(
            DEFAULT_PAIRWISE_IMPLEMENTATIONS
            if implementations is None
            else implementations
        )
        if sample_scope == "all-implementations" or implementations is not None
        else None
    )
    base, implementations = _base_results(
        result_date,
        files,
        require_common_shards=sample_scope == "all-implementations",
        required_implementations=selected,
    )
    enriched_histogram = _enriched_histogram_summary(
        _pairwise_values(base, sample_scope, implementations)
    ).collect(engine="streaming")
    pairwise = _pairwise_summary(enriched_histogram).sort(
        ["quantity", "implementation_a", "implementation_b"]
    )
    if sample_scope == "all-implementations":
        expected = (
            len(RESULT_QUANTITIES)
            * len(implementations)
            * (len(implementations) - 1)
            // 2
        )
        if pairwise.height != expected:
            raise ValueError(
                f"Expected {expected} implementation pairs for {result_date}, found "
                f"{pairwise.height}; no complete implementation population exists"
            )
    histogram = enriched_histogram.select(ABS_HISTOGRAM_COLUMNS).sort(
        ["quantity", "implementation_a", "implementation_b", "abs_bin_id"]
    )
    status = (
        _status_summary(base)
        .collect(engine="streaming")
        .sort(["quantity", "implementation", "status", "value_present"])
    )
    return DailyPairwiseSummaries(
        pairwise.select(PAIRWISE_COLUMNS).to_arrow(),
        histogram.select(ABS_HISTOGRAM_COLUMNS).to_arrow(),
        status.select(QUANTITY_STATUS_COLUMNS).to_arrow(),
    )


def persist_daily_pairwise(
    output_root: str | Path,
    result_date: date,
    summaries: DailyPairwiseSummaries,
) -> Path:
    """Atomically publish one date's three Parquet summary tables."""

    return persist_date_partitions(
        Path(output_root),
        result_date,
        {
            "pairwise": summaries.pairwise,
            "abs_histogram": summaries.abs_histogram,
            "quantity_status": summaries.quantity_status,
        },
    )


def _base_results(
    result_date: date,
    files: Sequence[ResultFile],
    *,
    require_common_shards: bool = False,
    required_implementations: tuple[str, ...] | None = None,
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    selected = frozenset(required_implementations) if required_implementations else None
    files = tuple(
        item for item in files if selected is None or item.implementation in selected
    )
    if not files:
        raise FileNotFoundError(f"No result shards found for {result_date.isoformat()}")
    discovered = {item.implementation for item in files}
    if required_implementations is not None:
        missing = set(required_implementations) - discovered
        if missing:
            raise ValueError(
                f"Configured pairwise implementations are missing for {result_date}: "
                f"{sorted(missing)}"
            )
        implementations = tuple(sorted(required_implementations))
    else:
        implementations = tuple(sorted(discovered))
    if len(implementations) < 2:
        raise ValueError(f"Fewer than two implementations exist for {result_date}")
    if require_common_shards:
        scopes = {
            implementation: tuple(
                sorted(
                    item.run_id
                    for item in files
                    if item.implementation == implementation
                )
            )
            for implementation in implementations
        }
        reference_scopes = scopes[implementations[0]]
        if not reference_scopes or any(
            implementation_scopes != reference_scopes
            for implementation_scopes in scopes.values()
        ):
            raise ValueError(
                f"Implementation shard scopes differ for {result_date}: {scopes}"
            )
    scans = [
        pl.scan_parquet(item.path)
        .select(
            "cycle_ordinal",
            "row_id",
            "row_ordinal",
            "wbgt_status",
            "wbgt_c",
            "tg_status",
            "tg_c",
            "tnwb_status",
            "tnwb_c",
        )
        .with_columns(
            pl.col(f"{quantity}_c").cast(pl.Float64) for quantity in RESULT_QUANTITIES
        )
        .with_columns(
            pl.lit(item.result_date).cast(pl.Date).alias("result_date"),
            pl.lit(item.run_id).alias("run_id"),
            pl.lit(item.implementation).alias("implementation"),
        )
        for item in files
    ]
    weights = pl.from_arrow(LATITUDE_WEIGHT_TABLE).lazy()
    base = (
        pl.concat(scans)
        .with_columns(pl.col("row_id").str.slice(12, 7).alias("latitude_code"))
        .join(weights, on="latitude_code", how="left")
        .drop("latitude_code")
    )
    return base, implementations


def _joint_valid(base: pl.LazyFrame) -> pl.LazyFrame:
    return base.filter(
        pl.all_horizontal(
            *[
                (pl.col(f"{quantity}_status") == "ok")
                & pl.col(f"{quantity}_c").is_not_null()
                for quantity in RESULT_QUANTITIES
            ]
        )
    )


def _pairwise_values(
    base: pl.LazyFrame,
    scope: SampleScope,
    implementations: tuple[str, ...],
) -> pl.LazyFrame:
    joint = _joint_valid(base)
    identity = ("result_date", "run_id", "cycle_ordinal", "row_id")
    if scope == "pairwise":
        right = joint.rename(
            {
                "implementation": "implementation_b",
                "area_weight": "area_weight_b",
                **{
                    f"{quantity}_c": f"{quantity}_c_b" for quantity in RESULT_QUANTITIES
                },
            }
        )
        wide = joint.join(right, on=identity, how="inner").filter(
            pl.col("implementation") < pl.col("implementation_b")
        )
        frames = [
            wide.select(
                "result_date",
                pl.lit(quantity).alias("quantity"),
                pl.col("implementation").alias("implementation_a"),
                "implementation_b",
                "area_weight",
                (pl.col(f"{quantity}_c") - pl.col(f"{quantity}_c_b")).alias(
                    "difference_c"
                ),
            )
            for quantity in RESULT_QUANTITIES
        ]
        return pl.concat(frames)

    by_implementation: dict[str, pl.LazyFrame] = {}
    for implementation in implementations:
        grouped = (
            joint.filter(pl.col("implementation") == implementation)
            .group_by(*identity)
            .agg(
                pl.len().alias("_n"),
                pl.col("area_weight").first().alias("area_weight"),
                *[
                    pl.col(f"{quantity}_c")
                    .first()
                    .alias(f"{implementation}_{quantity}")
                    for quantity in RESULT_QUANTITIES
                ],
            )
            .filter(pl.col("_n") == 1)
            .drop("_n")
        )
        by_implementation[implementation] = grouped
    first, *rest = implementations
    wide = by_implementation[first]
    for implementation in rest:
        wide = wide.join(
            by_implementation[implementation].drop("area_weight"),
            on=identity,
            how="inner",
        )
    return pl.concat(
        [
            wide.select(
                "result_date",
                pl.lit(quantity).alias("quantity"),
                pl.lit(left).alias("implementation_a"),
                pl.lit(right).alias("implementation_b"),
                "area_weight",
                (pl.col(f"{left}_{quantity}") - pl.col(f"{right}_{quantity}")).alias(
                    "difference_c"
                ),
            )
            for quantity in RESULT_QUANTITIES
            for left, right in combinations(implementations, 2)
        ]
    )


def _pairwise_aggregations() -> list[pl.Expr]:
    difference = pl.col("difference_c")
    absolute = difference.abs()
    weight = pl.col("area_weight")
    return [
        pl.len().alias("n"),
        difference.sum().alias("sum_difference_c"),
        (difference * difference).sum().alias("sum_squared_difference_c2"),
        absolute.sum().alias("sum_absolute_difference_c"),
        difference.min().alias("min_difference_c"),
        difference.max().alias("max_difference_c"),
        absolute.max().alias("max_absolute_difference_c"),
        weight.sum().alias("area_weight_sum"),
        (weight * difference).sum().alias("area_weighted_difference_sum_c"),
        (weight * difference * difference)
        .sum()
        .alias("area_weighted_squared_difference_sum_c2"),
        (weight * absolute).sum().alias("area_weighted_absolute_difference_sum_c"),
        *[
            expression
            for threshold in (0.1, 0.2, 0.5)
            for expression in _threshold_aggregates(absolute, weight, threshold)
        ],
    ]


def _pairwise_summary(enriched_histogram: pl.DataFrame) -> pl.DataFrame:
    additive_columns = (
        column
        for column in PAIRWISE_COLUMNS[len(PAIRWISE_GROUP_COLUMNS) :]
        if column not in PAIRWISE_MIN_COLUMNS + PAIRWISE_MAX_COLUMNS
    )
    return (
        enriched_histogram.group_by(*PAIRWISE_GROUP_COLUMNS)
        .agg(
            *(pl.col(column).sum() for column in additive_columns),
            *(pl.col(column).min() for column in PAIRWISE_MIN_COLUMNS),
            *(pl.col(column).max() for column in PAIRWISE_MAX_COLUMNS),
        )
        .select(PAIRWISE_COLUMNS)
    )


def _threshold_aggregates(
    absolute: pl.Expr, weight: pl.Expr, threshold: float
) -> list[pl.Expr]:
    label = str(threshold).replace(".", "_")
    selected = absolute > threshold
    return [
        selected.sum().alias(f"n_absolute_difference_gt_{label}_c"),
        pl.when(selected)
        .then(weight)
        .otherwise(0.0)
        .sum()
        .alias(f"area_weight_absolute_difference_gt_{label}_c"),
    ]


def _enriched_histogram_summary(values: pl.LazyFrame) -> pl.LazyFrame:
    bin_id, lower, upper = absolute_difference_bins(pl.col("difference_c"))
    return (
        values.with_columns(bin_id.alias("abs_bin_id"))
        .group_by(*PAIRWISE_GROUP_COLUMNS, "abs_bin_id")
        .agg(*_pairwise_aggregations())
        .with_columns(lower.alias("abs_bin_lower_c"), upper.alias("abs_bin_upper_c"))
    )


def _status_summary(base: pl.LazyFrame) -> pl.LazyFrame:
    return (
        pl.concat(
            [
                base.select(
                    "result_date",
                    pl.lit(quantity).alias("quantity"),
                    "implementation",
                    pl.col(f"{quantity}_status").alias("status"),
                    pl.col(f"{quantity}_c").is_not_null().alias("value_present"),
                    "area_weight",
                )
                for quantity in RESULT_QUANTITIES
            ]
        )
        .group_by(
            "result_date", "quantity", "implementation", "status", "value_present"
        )
        .agg(pl.len().alias("n"), pl.col("area_weight").sum().alias("area_weight_sum"))
    )
