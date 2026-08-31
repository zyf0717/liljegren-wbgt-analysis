"""Compute and persist single-pass daily temperature summaries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import monotonic

import duckdb
import pyarrow as pa
import pyarrow.compute as pc

from .persistence import (
    PairwiseDatabaseSchema,
    TableSchema,
    persist_date_partitions,
)
from .area_weighted import _ERA5_LATITUDE_WEIGHTS
from .distributions import ABS_HISTOGRAM_VERSION
from .sql import (
    EXPECTED_IMPLEMENTATIONS,
    IMPLEMENTATION_SET_VERSION,
    QUANTITY_SCOPE,
    QUANTITY_SET_VERSION,
    SAMPLE_SCOPES,
    SampleScope,
    build_daily_pairwise_sql,
)
from ..results import HSBRunResults, RESULT_QUANTITIES, RESULT_SCHEMA_ID


PAIRWISE_COLUMNS = (
    "result_date",
    "quantity",
    "implementation_a",
    "implementation_b",
    "n",
    "sum_difference_c",
    "sum_squared_difference_c2",
    "sum_absolute_difference_c",
    "min_difference_c",
    "max_difference_c",
    "max_absolute_difference_c",
    "area_weight_sum",
    "area_weighted_difference_sum_c",
    "area_weighted_squared_difference_sum_c2",
    "area_weighted_absolute_difference_sum_c",
    "n_absolute_difference_gt_0_1_c",
    "n_absolute_difference_gt_0_2_c",
    "n_absolute_difference_gt_0_5_c",
    "area_weight_absolute_difference_gt_0_1_c",
    "area_weight_absolute_difference_gt_0_2_c",
    "area_weight_absolute_difference_gt_0_5_c",
)
ABS_HISTOGRAM_COLUMNS = (
    "result_date",
    "quantity",
    "implementation_a",
    "implementation_b",
    "abs_bin_id",
    "abs_bin_lower_c",
    "abs_bin_upper_c",
    "n",
    "area_weight_sum",
)
QUANTITY_STATUS_COLUMNS = (
    "result_date",
    "quantity",
    "implementation",
    "status",
    "value_present",
    "n",
    "area_weight_sum",
)


@dataclass(frozen=True)
class DailyPairwiseSummaries:
    """Compact results from one date-scoped full-data pass."""

    pairwise: pa.Table
    abs_histogram: pa.Table
    quantity_status: pa.Table


def compute_daily_pairwise(
    result_date: date,
    runs: tuple[Path, ...],
    threads: int,
    sample_scope: SampleScope = "pairwise",
) -> tuple[date, DailyPairwiseSummaries, int, float]:
    """Compute totals and distribution summaries in one query for one date."""

    if not runs:
        raise ValueError("runs is empty")
    if threads < 1:
        raise ValueError("threads must be positive")
    _validate_sample_scope(sample_scope)

    started = monotonic()
    with duckdb.connect() as connection:
        connection.execute(f"SET threads = {threads}")
        connection.register("_era5_latitude_weights", _ERA5_LATITUDE_WEIGHTS)
        all_results = _combined_results(result_date, runs, connection)
        combined = all_results.query(
            "results",
            build_daily_pairwise_sql(sample_scope),
        ).arrow().read_all()

    summaries = _split_summaries(combined)
    if sample_scope == "all-implementations":
        expected_pairs = len(RESULT_QUANTITIES) * len(EXPECTED_IMPLEMENTATIONS) * (
            len(EXPECTED_IMPLEMENTATIONS) - 1
        ) // 2
        if summaries.pairwise.num_rows != expected_pairs:
            raise ValueError(
                f"Expected {expected_pairs} implementation pairs for "
                f"{result_date}, found {summaries.pairwise.num_rows}; no "
                "complete seven-implementation population exists"
            )
    return result_date, summaries, os.getpid(), monotonic() - started


def persist_daily_pairwise(
    connection: duckdb.DuckDBPyConnection,
    result_date: date,
    summaries: DailyPairwiseSummaries,
    sample_scope: SampleScope = "pairwise",
) -> None:
    """Atomically replace one date across all compact summary tables."""

    _validate_sample_scope(sample_scope)
    persist_date_partitions(
        connection,
        result_date,
        {
            "daily_pairwise": summaries.pairwise,
            "daily_pairwise_abs_histogram": summaries.abs_histogram,
            "daily_quantity_status": summaries.quantity_status,
        },
        _database_schema(sample_scope),
    )


def _split_summaries(combined: pa.Table) -> DailyPairwiseSummaries:
    def rows(kind: str, columns: tuple[str, ...]) -> pa.Table:
        selected = combined.filter(pc.equal(combined["summary_kind"], kind))
        return selected.select(columns)

    return DailyPairwiseSummaries(
        pairwise=rows("pairwise", PAIRWISE_COLUMNS),
        abs_histogram=rows("abs_histogram", ABS_HISTOGRAM_COLUMNS),
        quantity_status=rows("quantity_status", QUANTITY_STATUS_COLUMNS),
    )


def _database_schema(sample_scope: SampleScope) -> PairwiseDatabaseSchema:
    pair_identity = (
        ("result_date", "DATE", False),
        ("quantity", "VARCHAR", False),
        ("implementation_a", "VARCHAR", False),
        ("implementation_b", "VARCHAR", False),
    )
    pairwise_columns = pair_identity + (
        ("n", "BIGINT", False),
        ("sum_difference_c", "DOUBLE", False),
        ("sum_squared_difference_c2", "DOUBLE", False),
        ("sum_absolute_difference_c", "DOUBLE", False),
        ("min_difference_c", "DOUBLE", False),
        ("max_difference_c", "DOUBLE", False),
        ("max_absolute_difference_c", "DOUBLE", False),
        ("area_weight_sum", "DOUBLE", False),
        ("area_weighted_difference_sum_c", "DOUBLE", False),
        ("area_weighted_squared_difference_sum_c2", "DOUBLE", False),
        ("area_weighted_absolute_difference_sum_c", "DOUBLE", False),
        ("n_absolute_difference_gt_0_1_c", "BIGINT", False),
        ("n_absolute_difference_gt_0_2_c", "BIGINT", False),
        ("n_absolute_difference_gt_0_5_c", "BIGINT", False),
        ("area_weight_absolute_difference_gt_0_1_c", "DOUBLE", False),
        ("area_weight_absolute_difference_gt_0_2_c", "DOUBLE", False),
        ("area_weight_absolute_difference_gt_0_5_c", "DOUBLE", False),
    )
    histogram_columns = pair_identity + (
        ("abs_bin_id", "INTEGER", False),
        ("abs_bin_lower_c", "DOUBLE", False),
        ("abs_bin_upper_c", "DOUBLE", True),
        ("n", "BIGINT", False),
        ("area_weight_sum", "DOUBLE", False),
    )
    quantity_status_columns = (
        ("result_date", "DATE", False),
        ("quantity", "VARCHAR", False),
        ("implementation", "VARCHAR", False),
        ("status", "VARCHAR", False),
        ("value_present", "BOOLEAN", False),
        ("n", "BIGINT", False),
        ("area_weight_sum", "DOUBLE", False),
    )
    return PairwiseDatabaseSchema(
        version=5,
        sample_scope=sample_scope,
        result_schema_id=RESULT_SCHEMA_ID,
        quantity_scope=QUANTITY_SCOPE,
        quantity_set_version=QUANTITY_SET_VERSION,
        implementation_set_version=IMPLEMENTATION_SET_VERSION,
        abs_histogram_version=ABS_HISTOGRAM_VERSION,
        tables=(
            TableSchema(
                "daily_pairwise",
                pairwise_columns,
                (
                    "result_date",
                    "quantity",
                    "implementation_a",
                    "implementation_b",
                ),
            ),
            TableSchema(
                "daily_pairwise_abs_histogram",
                histogram_columns,
                (
                    "result_date",
                    "quantity",
                    "implementation_a",
                    "implementation_b",
                    "abs_bin_id",
                ),
            ),
            TableSchema(
                "daily_quantity_status",
                quantity_status_columns,
                (
                    "result_date",
                    "quantity",
                    "implementation",
                    "status",
                    "value_present",
                ),
            ),
        ),
    )


def _validate_sample_scope(sample_scope: SampleScope) -> None:
    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")


def _combined_results(
    result_date: date,
    runs: tuple[Path, ...],
    connection: duckdb.DuckDBPyConnection,
) -> duckdb.DuckDBPyRelation:
    _validate_nonoverlapping_scopes(result_date, runs)
    relations = [
        HSBRunResults.from_hsb_run(run, connection=connection).scan(
            result_date=result_date,
        )
        for run in runs
    ]
    combined = relations[0]
    for relation in relations[1:]:
        combined = combined.union(relation)
    return combined


def _validate_nonoverlapping_scopes(
    result_date: date,
    runs: tuple[Path, ...],
) -> None:
    scopes: dict[tuple[str, str], Path] = {}
    for run in runs:
        date_root = run.expanduser().resolve() / "results" / result_date.isoformat()
        for path in sorted(date_root.glob("*/*/results.parquet")):
            run_id = path.parents[1].name
            implementation = path.parent.name
            scope = (run_id, implementation)
            previous = scopes.get(scope)
            if previous is not None:
                raise ValueError(
                    "Overlapping result scope for "
                    f"{result_date} run_id={run_id} "
                    f"implementation={implementation}: {previous} and {path}"
                )
            scopes[scope] = path.resolve()
