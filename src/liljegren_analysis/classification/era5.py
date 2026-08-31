"""Scalable ERA5 country-cache WBGT classification analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from .core import ClassificationAnalysis, class_definitions, validate_thresholds


@dataclass(frozen=True)
class Era5ClassificationAnalysis:
    """Aggregate ERA5 classifications without materializing source observations."""

    classification: ClassificationAnalysis
    observation_summary: pd.DataFrame
    quantity_coverage: pd.DataFrame

    @property
    def outputs(self) -> dict[str, pd.DataFrame]:
        return {
            "observation_summary.csv": self.observation_summary,
            "quantity_coverage.csv": self.quantity_coverage,
            **self.classification.outputs,
        }


def analyze_era5_country_cache(
    cache_path: str | Path,
    start_date: date,
    end_date: date,
    thresholds_c: Sequence[float],
    *,
    threads: int = 0,
    memory_limit: str = "8GB",
    temp_directory: str | Path | None = None,
) -> Era5ClassificationAnalysis:
    """Aggregate classifications from a preselected country cache with DuckDB.

    The cache is never loaded into pandas and no raw implementation-pair rows
    are materialized. DuckDB may spill joins and aggregations to
    ``temp_directory`` under the configured memory limit.
    """

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if threads < 0:
        raise ValueError("threads must be non-negative")
    if not memory_limit.strip():
        raise ValueError("memory_limit must not be empty")
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"country cache does not exist: {path}")
    thresholds = validate_thresholds(thresholds_c)
    classes = class_definitions(thresholds)

    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = ?", [memory_limit])
        connection.execute("SET preserve_insertion_order = false")
        if threads:
            connection.execute("SET threads = ?", [threads])
        if temp_directory is not None:
            temporary = Path(temp_directory).expanduser().resolve()
            temporary.mkdir(parents=True, exist_ok=True)
            connection.execute(f"SET temp_directory = {_sql_literal(temporary)}")
        _create_observations_view(connection, path, start_date, end_date)
        _validate_observations(connection)
        summary = _query_observation_summary(connection)
        coverage = _query_quantity_coverage(connection)
        _create_classified_view(connection, thresholds)
        _create_paired_view(connection)
        if not connection.sql("SELECT EXISTS (SELECT 1 FROM paired_classifications)").fetchone()[0]:
            raise ValueError("fewer than two implementations have matched valid WBGT rows")
        outputs = _query_outputs(connection, classes)
    finally:
        connection.close()

    return Era5ClassificationAnalysis(
        ClassificationAnalysis(
            classes=classes,
            classified=pd.DataFrame(),
            paired=pd.DataFrame(),
            outputs=outputs,
        ),
        summary,
        coverage,
    )


def _create_observations_view(
    connection: duckdb.DuckDBPyConnection,
    cache_path: Path,
    start_date: date,
    end_date: date,
) -> None:
    connection.execute(
        f"""
        CREATE TEMP VIEW observations AS
        SELECT
            result_date, run_id, implementation, cycle_ordinal, row_id, cell_id,
            wbgt_status, wbgt_c, tg_status, tg_c, tnwb_status, tnwb_c
        FROM read_parquet({_sql_literal(cache_path)})
        WHERE result_date BETWEEN DATE {_sql_literal(start_date)} AND DATE {_sql_literal(end_date)}
        """
    )


def _validate_observations(connection: duckdb.DuckDBPyConnection) -> None:
    result_rows = connection.sql("SELECT count(*)::UBIGINT FROM observations").fetchone()[0]
    duplicate_rows = connection.sql(
        """
        SELECT coalesce(sum(n), 0)::UBIGINT
        FROM (
            SELECT count(*) AS n
            FROM observations
            GROUP BY result_date, run_id, implementation, cycle_ordinal, row_id
            HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    if not result_rows:
        raise ValueError("country cache has no rows in the requested date range")
    if duplicate_rows:
        raise ValueError(f"country cache contains {duplicate_rows} duplicate result rows")


def _query_observation_summary(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.sql(
        """
        SELECT
            count(*)::UBIGINT AS result_rows,
            count(*) FILTER (
                WHERE wbgt_status = 'ok' AND wbgt_c IS NOT NULL AND NOT isnan(wbgt_c)
            )::UBIGINT AS valid_wbgt_observations,
            count(DISTINCT implementation)::UBIGINT AS implementations,
            count(DISTINCT cell_id)::UBIGINT AS cells,
            min(result_date) AS first_date,
            max(result_date) AS last_date
        FROM observations
        """
    ).df()


def _query_quantity_coverage(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.sql(
        """
        SELECT * FROM (
            SELECT 'wbgt' AS quantity, implementation, wbgt_status AS quantity_status,
                   wbgt_c IS NOT NULL AND NOT isnan(wbgt_c) AS quantity_value_present,
                   count(*)::UBIGINT AS n
            FROM observations
            GROUP BY implementation, wbgt_status, wbgt_c IS NOT NULL AND NOT isnan(wbgt_c)
            UNION ALL
            SELECT 'tg', implementation, tg_status,
                   tg_c IS NOT NULL AND NOT isnan(tg_c), count(*)::UBIGINT
            FROM observations
            GROUP BY implementation, tg_status, tg_c IS NOT NULL AND NOT isnan(tg_c)
            UNION ALL
            SELECT 'tnwb', implementation, tnwb_status,
                   tnwb_c IS NOT NULL AND NOT isnan(tnwb_c), count(*)::UBIGINT
            FROM observations
            GROUP BY implementation, tnwb_status, tnwb_c IS NOT NULL AND NOT isnan(tnwb_c)
        )
        ORDER BY quantity, implementation, quantity_status, quantity_value_present
        """
    ).df()


def _create_classified_view(
    connection: duckdb.DuckDBPyConnection, thresholds: tuple[float, ...]
) -> None:
    classification_case = "CASE " + " ".join(
        f"WHEN wbgt_c < {threshold:.17g} THEN {class_id}"
        for class_id, threshold in enumerate(thresholds)
    ) + f" ELSE {len(thresholds)} END::TINYINT"
    connection.execute(
        f"""
        CREATE TEMP VIEW classified AS
        SELECT
            result_date, run_id, implementation, cycle_ordinal, row_id, cell_id, wbgt_c,
            {classification_case} AS class_id
        FROM observations
        WHERE wbgt_status = 'ok' AND wbgt_c IS NOT NULL AND NOT isnan(wbgt_c)
        """
    )


def _create_paired_view(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP VIEW paired_classifications AS
        SELECT
            a.result_date, a.run_id, a.cycle_ordinal, a.row_id, a.cell_id,
            a.implementation AS implementation_a,
            b.implementation AS implementation_b,
            a.wbgt_c AS wbgt_c_a, b.wbgt_c AS wbgt_c_b,
            a.class_id AS class_id_a, b.class_id AS class_id_b,
            a.wbgt_c - b.wbgt_c AS wbgt_difference_c,
            a.class_id - b.class_id AS class_difference,
            a.class_id != b.class_id AS classification_changed
        FROM classified AS a
        INNER JOIN classified AS b
            USING (result_date, run_id, cycle_ordinal, row_id, cell_id)
        WHERE a.implementation < b.implementation
        """
    )


def _query_outputs(
    connection: duckdb.DuckDBPyConnection, classes: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    pair_columns = "implementation_a, implementation_b"
    monthly_columns = "strftime(result_date, '%Y-%m') AS result_month, " + pair_columns
    return {
        "classes.csv": classes,
        "pair_change_summary.csv": _query_change_summary(connection, pair_columns),
        "pair_transition_counts.csv": _add_transition_labels(
            _query_transition_counts(connection, pair_columns), classes
        ),
        "monthly_pair_change_summary.csv": _query_change_summary(
            connection, monthly_columns
        ),
        "monthly_pair_transition_counts.csv": _add_transition_labels(
            _query_transition_counts(connection, monthly_columns), classes
        ),
    }


def _query_change_summary(
    connection: duckdb.DuckDBPyConnection, group_columns: str
) -> pd.DataFrame:
    return connection.sql(
        f"""
        SELECT
            {group_columns},
            count(*)::UBIGINT AS n_paired,
            count(*) FILTER (WHERE classification_changed)::UBIGINT AS n_changed,
            round(100.0 * n_changed / n_paired, 4) AS percent_changed,
            count(*) FILTER (WHERE class_difference > 0)::UBIGINT AS a_higher_class,
            count(*) FILTER (WHERE class_difference < 0)::UBIGINT AS b_higher_class
        FROM paired_classifications
        GROUP BY ALL
        ORDER BY ALL
        """
    ).df()


def _query_transition_counts(
    connection: duckdb.DuckDBPyConnection, group_columns: str
) -> pd.DataFrame:
    return connection.sql(
        f"""
        SELECT
            {group_columns}, class_id_a, class_id_b,
            count(*)::UBIGINT AS n,
            class_id_a != class_id_b AS classification_changed
        FROM paired_classifications
        GROUP BY ALL
        ORDER BY ALL
        """
    ).df()


def _add_transition_labels(counts: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    labels = classes.set_index("class_id")["class_label"]
    counts.insert(
        counts.columns.get_loc("class_id_a") + 1,
        "class_a_label",
        counts["class_id_a"].map(labels),
    )
    counts.insert(
        counts.columns.get_loc("class_id_b") + 1,
        "class_b_label",
        counts["class_id_b"].map(labels),
    )
    return counts


def _sql_literal(value: str | Path | date) -> str:
    return "'" + str(value).replace("'", "''") + "'"
