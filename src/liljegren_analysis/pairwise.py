"""Compute and persist daily pairwise WBGT statistics."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from time import monotonic

import duckdb
import pyarrow as pa

from ._persistence import PairwiseTableSchema, persist_date_partition
from .results import CampaignResults


DAILY_PAIRWISE_SCHEMA = PairwiseTableSchema(
    table_name="daily_pairwise",
    version=2,
    columns=(
        ("result_date", "DATE", False),
        ("implementation_a", "VARCHAR", False),
        ("implementation_b", "VARCHAR", False),
        ("n", "BIGINT", False),
        ("mean_difference_c", "DOUBLE", False),
        ("variance_difference", "DOUBLE", True),
        ("stddev_difference_c", "DOUBLE", True),
        ("mean_absolute_difference_c", "DOUBLE", False),
        ("rmse_c", "DOUBLE", False),
        ("min_difference_c", "DOUBLE", False),
        ("max_difference_c", "DOUBLE", False),
        ("max_absolute_difference_c", "DOUBLE", False),
        ("p05_difference_c", "DOUBLE", False),
        ("p25_difference_c", "DOUBLE", False),
        ("median_difference_c", "DOUBLE", False),
        ("p75_difference_c", "DOUBLE", False),
        ("p95_difference_c", "DOUBLE", False),
        ("median_absolute_difference_c", "DOUBLE", False),
        ("p95_absolute_difference_c", "DOUBLE", False),
    ),
)


DAILY_PAIRWISE_SQL = """
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
        left_result.value_c - right_result.value_c AS difference_c
    FROM valid_results AS left_result
    INNER JOIN valid_results AS right_result
        ON left_result.result_date = right_result.result_date
       AND left_result.run_id = right_result.run_id
       AND left_result.cycle_ordinal = right_result.cycle_ordinal
       AND left_result.row_id = right_result.row_id
       AND left_result.implementation < right_result.implementation
),
daily_statistics AS (
    SELECT
        result_date,
        implementation_a,
        implementation_b,

        count(*) AS n,
        avg(difference_c) AS mean_difference_c,
        var_samp(difference_c) AS variance_difference,

        avg(abs(difference_c)) AS mean_absolute_difference_c,
        sqrt(avg(difference_c * difference_c)) AS rmse_c,

        min(difference_c) AS min_difference_c,
        max(difference_c) AS max_difference_c,
        max(abs(difference_c)) AS max_absolute_difference_c,

        approx_quantile(
            difference_c,
            [0.05, 0.25, 0.50, 0.75, 0.95]
        ) AS difference_quantiles_c,

        approx_quantile(
            abs(difference_c),
            [0.50, 0.95]
        ) AS absolute_difference_quantiles_c
    FROM paired_values
    GROUP BY
        result_date,
        implementation_a,
        implementation_b
)
SELECT
    result_date,
    implementation_a,
    implementation_b,

    n,
    mean_difference_c,
    variance_difference,
    sqrt(variance_difference) AS stddev_difference_c,

    mean_absolute_difference_c,
    rmse_c,

    min_difference_c,
    max_difference_c,
    max_absolute_difference_c,

    difference_quantiles_c[1] AS p05_difference_c,
    difference_quantiles_c[2] AS p25_difference_c,
    difference_quantiles_c[3] AS median_difference_c,
    difference_quantiles_c[4] AS p75_difference_c,
    difference_quantiles_c[5] AS p95_difference_c,

    absolute_difference_quantiles_c[1]
        AS median_absolute_difference_c,
    absolute_difference_quantiles_c[2]
        AS p95_absolute_difference_c
FROM daily_statistics
ORDER BY
    result_date,
    implementation_a,
    implementation_b
"""


def compute_daily_pairwise(
    result_date: date,
    runs: tuple[Path, ...],
    threads: int,
) -> tuple[date, pa.Table, int, float]:
    """Compute one date in an isolated-process-compatible work unit."""

    if not runs:
        raise ValueError("runs is empty")
    if threads < 1:
        raise ValueError("threads must be positive")

    started = monotonic()
    with duckdb.connect() as connection:
        connection.execute(f"SET threads = {threads}")
        all_results = _combined_results(result_date, runs, connection)
        result = all_results.query("results", DAILY_PAIRWISE_SQL)
        return (
            result_date,
            result.arrow().read_all(),
            os.getpid(),
            monotonic() - started,
        )


def persist_daily_pairwise(
    connection: duckdb.DuckDBPyConnection,
    result_date: date,
    result: pa.Table,
) -> None:
    """Transactionally replace one date in ``daily_pairwise``."""

    persist_date_partition(connection, result_date, result, DAILY_PAIRWISE_SCHEMA)


def _combined_results(
    result_date: date,
    runs: tuple[Path, ...],
    connection: duckdb.DuckDBPyConnection,
) -> duckdb.DuckDBPyRelation:
    _validate_nonoverlapping_scopes(result_date, runs)
    relations = [
        CampaignResults.from_campaign(run, connection=connection).scan(
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
