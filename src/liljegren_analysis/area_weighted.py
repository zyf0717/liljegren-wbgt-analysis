"""ERA5 cell-area-weighted pairwise WBGT statistics."""

from __future__ import annotations

import math
import os
from datetime import date
from pathlib import Path
from time import monotonic

import duckdb
import pyarrow as pa

from ._persistence import PairwiseTableSchema, persist_date_partition
from .pairwise import _combined_results


ERA5_GRID_SPACING_DEGREES = 0.25
_ERA5_LATITUDE_SCALE = 10_000
_ERA5_LATITUDE_STEP = round(
    ERA5_GRID_SPACING_DEGREES * _ERA5_LATITUDE_SCALE
)
_ERA5_MAX_LATITUDE = 90 * _ERA5_LATITUDE_SCALE


def _normalized_cell_area_weight(latitude_degrees: float) -> float:
    half_spacing = ERA5_GRID_SPACING_DEGREES / 2.0
    lower = math.radians(max(-90.0, latitude_degrees - half_spacing))
    upper = math.radians(min(90.0, latitude_degrees + half_spacing))
    normalization = 2.0 * math.sin(math.radians(half_spacing))
    return (math.sin(upper) - math.sin(lower)) / normalization


def _precomputed_latitude_weights() -> pa.Table:
    latitude_units = range(
        -_ERA5_MAX_LATITUDE,
        _ERA5_MAX_LATITUDE + 1,
        _ERA5_LATITUDE_STEP,
    )
    codes: list[str] = []
    weights: list[float] = []
    for units in latitude_units:
        hemisphere = "s" if units < 0 else "n"
        codes.append(f"{hemisphere}{abs(units):06d}")
        weights.append(
            _normalized_cell_area_weight(units / _ERA5_LATITUDE_SCALE)
        )
    return pa.table({"latitude_code": codes, "area_weight": weights})


_ERA5_LATITUDE_WEIGHTS = _precomputed_latitude_weights()


DAILY_AREA_WEIGHTED_PAIRWISE_SCHEMA = PairwiseTableSchema(
    table_name="daily_pairwise_area_weighted",
    version=1,
    columns=(
        ("result_date", "DATE", False),
        ("implementation_a", "VARCHAR", False),
        ("implementation_b", "VARCHAR", False),
        ("n", "BIGINT", False),
        ("area_weight_sum", "DOUBLE", False),
        ("area_weighted_mean_difference_c", "DOUBLE", False),
        ("area_weighted_variance_difference", "DOUBLE", False),
        ("area_weighted_stddev_difference_c", "DOUBLE", False),
        ("area_weighted_mean_absolute_difference_c", "DOUBLE", False),
        ("area_weighted_rmse_c", "DOUBLE", False),
    ),
)


DAILY_AREA_WEIGHTED_PAIRWISE_SQL = """
WITH valid_results AS (
    SELECT
        result_date,
        run_id,
        implementation,
        cycle_ordinal,
        row_id,
        value_c
    FROM results
    WHERE status = 'ok'
      AND value_c IS NOT NULL
),
paired_values AS (
    SELECT
        left_result.result_date,
        left_result.implementation AS implementation_a,
        right_result.implementation AS implementation_b,
        left_result.row_id,
        left_result.value_c - right_result.value_c AS difference_c
    FROM valid_results AS left_result
    INNER JOIN valid_results AS right_result
        ON left_result.result_date = right_result.result_date
       AND left_result.run_id = right_result.run_id
       AND left_result.cycle_ordinal = right_result.cycle_ordinal
       AND left_result.row_id = right_result.row_id
       AND left_result.implementation < right_result.implementation
),
weighted_values AS (
    SELECT
        paired.result_date,
        paired.implementation_a,
        paired.implementation_b,
        paired.difference_c,
        CASE
            WHEN regexp_full_match(
                paired.row_id,
                '^era5-global-[ns][0-9]{6}-.+$'
            ) AND latitude.area_weight IS NOT NULL
            THEN latitude.area_weight
            ELSE error('Invalid ERA5 row_id: ' || paired.row_id)
        END AS area_weight
    FROM paired_values AS paired
    LEFT JOIN _era5_latitude_weights AS latitude
        ON substring(paired.row_id, 13, 7) = latitude.latitude_code
),
daily_sums AS (
    SELECT
        result_date,
        implementation_a,
        implementation_b,
        count(*) AS n,
        sum(area_weight) AS area_weight_sum,
        sum(area_weight * difference_c)
            AS area_weighted_difference_sum_c,
        sum(area_weight * difference_c * difference_c)
            AS area_weighted_squared_difference_sum_c2,
        sum(area_weight * abs(difference_c))
            AS area_weighted_absolute_difference_sum_c
    FROM weighted_values
    GROUP BY
        result_date,
        implementation_a,
        implementation_b
),
daily_statistics AS (
    SELECT
        result_date,
        implementation_a,
        implementation_b,
        n,
        area_weight_sum,
        area_weighted_difference_sum_c / area_weight_sum
            AS area_weighted_mean_difference_c,
        greatest(
            area_weighted_squared_difference_sum_c2 / area_weight_sum
            - power(
                area_weighted_difference_sum_c / area_weight_sum,
                2
            ),
            0.0
        ) AS area_weighted_variance_difference,
        area_weighted_absolute_difference_sum_c / area_weight_sum
            AS area_weighted_mean_absolute_difference_c,
        sqrt(
            area_weighted_squared_difference_sum_c2 / area_weight_sum
        ) AS area_weighted_rmse_c
    FROM daily_sums
)
SELECT
    result_date,
    implementation_a,
    implementation_b,
    n,
    area_weight_sum,
    area_weighted_mean_difference_c,
    area_weighted_variance_difference,
    sqrt(area_weighted_variance_difference)
        AS area_weighted_stddev_difference_c,
    area_weighted_mean_absolute_difference_c,
    area_weighted_rmse_c
FROM daily_statistics
ORDER BY
    result_date,
    implementation_a,
    implementation_b
"""


def compute_daily_area_weighted_pairwise(
    result_date: date,
    runs: tuple[Path, ...],
    threads: int,
) -> tuple[date, pa.Table, int, float]:
    """Compute one date of ERA5 cell-area-weighted pairwise statistics."""

    if not runs:
        raise ValueError("runs is empty")
    if threads < 1:
        raise ValueError("threads must be positive")

    started = monotonic()
    with duckdb.connect() as connection:
        connection.execute(f"SET threads = {threads}")
        connection.register(
            "_era5_latitude_weights",
            _ERA5_LATITUDE_WEIGHTS,
        )
        all_results = _combined_results(result_date, runs, connection)
        result = all_results.query(
            "results",
            DAILY_AREA_WEIGHTED_PAIRWISE_SQL,
        )
        return (
            result_date,
            result.arrow().read_all(),
            os.getpid(),
            monotonic() - started,
        )


def persist_daily_area_weighted_pairwise(
    connection: duckdb.DuckDBPyConnection,
    result_date: date,
    result: pa.Table,
) -> None:
    """Replace one date in ``daily_pairwise_area_weighted``."""

    persist_date_partition(
        connection,
        result_date,
        result,
        DAILY_AREA_WEIGHTED_PAIRWISE_SCHEMA,
    )
