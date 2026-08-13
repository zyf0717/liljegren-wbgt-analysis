"""Versioned atomic persistence for daily pairwise summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb
import pyarrow as pa


@dataclass(frozen=True)
class TableSchema:
    """Describe one persisted daily summary table."""

    table_name: str
    columns: tuple[tuple[str, str, bool], ...]
    primary_key: tuple[str, ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.columns)


@dataclass(frozen=True)
class PairwiseDatabaseSchema:
    """Describe the complete pairwise-cache contract."""

    version: int
    sample_scope: str
    implementation_set_version: int
    abs_histogram_version: int
    tables: tuple[TableSchema, ...]


def persist_date_partitions(
    connection: duckdb.DuckDBPyConnection,
    result_date: date,
    results: dict[str, pa.Table],
    schema: PairwiseDatabaseSchema,
) -> None:
    """Atomically replace one date in every v4 summary table."""

    expected_tables = {table.table_name for table in schema.tables}
    if set(results) != expected_tables:
        raise ValueError(
            "Daily summary tables differ: "
            f"missing={sorted(expected_tables - set(results))}, "
            f"extra={sorted(set(results) - expected_tables)}"
        )

    registrations: list[str] = []
    try:
        for table_schema in schema.tables:
            result = results[table_schema.table_name]
            _validate_result(result_date, result, table_schema)
            registration = f"_day_{table_schema.table_name}"
            connection.register(registration, result)
            registrations.append(registration)

        _validate_cross_table_totals(connection)
        connection.execute("BEGIN")
        try:
            _ensure_schema(connection, schema)
            for table_schema in schema.tables:
                connection.execute(
                    f"DELETE FROM {table_schema.table_name} "
                    "WHERE result_date = ?",
                    [result_date],
                )
                columns = ", ".join(table_schema.column_names)
                connection.execute(
                    f"""
                    INSERT INTO {table_schema.table_name} ({columns})
                    SELECT {columns}
                    FROM _day_{table_schema.table_name}
                    """
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    finally:
        for registration in registrations:
            connection.unregister(registration)


def _validate_result(
    result_date: date,
    result: pa.Table,
    schema: TableSchema,
) -> None:
    expected = set(schema.column_names)
    actual = set(result.column_names)
    if actual != expected or len(result.column_names) != len(expected):
        raise ValueError(
            f"{schema.table_name} result schema differs: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    if any(value != result_date for value in result["result_date"].to_pylist()):
        raise ValueError(
            f"{schema.table_name} contains a date other than {result_date}"
        )


def _validate_cross_table_totals(connection: duckdb.DuckDBPyConnection) -> None:
    invalid = connection.execute(
        """
        WITH histogram AS (
            SELECT
                implementation_a,
                implementation_b,
                sum(n)::BIGINT AS n,
                sum(area_weight_sum) AS area_weight_sum
            FROM _day_daily_pairwise_abs_histogram
            GROUP BY implementation_a, implementation_b
        )
        SELECT count(*)
        FROM _day_daily_pairwise AS totals
        FULL OUTER JOIN histogram USING (implementation_a, implementation_b)
        WHERE totals.implementation_a IS NULL
           OR histogram.implementation_a IS NULL
           OR totals.n <> histogram.n
           OR abs(totals.area_weight_sum - histogram.area_weight_sum)
                > greatest(1e-6, abs(totals.area_weight_sum) * 1e-12)
        """
    ).fetchone()[0]
    if invalid:
        raise ValueError("Daily histogram totals differ from pairwise totals")


def _ensure_schema(
    connection: duckdb.DuckDBPyConnection,
    schema: PairwiseDatabaseSchema,
) -> None:
    metadata_exists = _table_exists(connection, "pairwise_schema")
    table_existence = {
        table.table_name: _table_exists(connection, table.table_name)
        for table in schema.tables
    }
    if not metadata_exists and not any(table_existence.values()):
        _create_schema(connection, schema)
        return
    if not metadata_exists or not all(table_existence.values()):
        raise ValueError(_incompatible_message(schema))

    metadata_columns = _columns(connection, "pairwise_schema")
    expected_metadata_columns = [
        ("schema_version", "INTEGER", False),
        ("sample_scope", "VARCHAR", False),
        ("implementation_set_version", "INTEGER", False),
        ("abs_histogram_version", "INTEGER", False),
    ]
    if metadata_columns != expected_metadata_columns:
        raise ValueError(_incompatible_message(schema))

    metadata = connection.execute("SELECT * FROM pairwise_schema").fetchall()
    expected_metadata = [
        (
            schema.version,
            schema.sample_scope,
            schema.implementation_set_version,
            schema.abs_histogram_version,
        )
    ]
    if metadata != expected_metadata:
        raise ValueError(_incompatible_message(schema))

    for table in schema.tables:
        if _columns(connection, table.table_name) != list(table.columns):
            raise ValueError(_incompatible_message(schema))


def _create_schema(
    connection: duckdb.DuckDBPyConnection,
    schema: PairwiseDatabaseSchema,
) -> None:
    connection.execute(
        """
        CREATE TABLE pairwise_schema (
            schema_version INTEGER NOT NULL,
            sample_scope VARCHAR NOT NULL,
            implementation_set_version INTEGER NOT NULL,
            abs_histogram_version INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO pairwise_schema VALUES (?, ?, ?, ?)",
        [
            schema.version,
            schema.sample_scope,
            schema.implementation_set_version,
            schema.abs_histogram_version,
        ],
    )
    for table in schema.tables:
        definitions = [
            f"{name} {data_type}{'' if nullable else ' NOT NULL'}"
            for name, data_type, nullable in table.columns
        ]
        definitions.append(f"PRIMARY KEY ({', '.join(table.primary_key)})")
        connection.execute(
            f"CREATE TABLE {table.table_name} ({', '.join(definitions)})"
        )


def _columns(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> list[tuple[str, str, bool]]:
    return connection.execute(
        """
        SELECT column_name, data_type, is_nullable = 'YES'
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table_name],
    ).fetchall()


def _table_exists(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> bool:
    return bool(
        connection.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def _incompatible_message(schema: PairwiseDatabaseSchema) -> str:
    return (
        "Incompatible pairwise database schema; expected version "
        f"{schema.version}, sample_scope={schema.sample_scope}, "
        f"implementation_set_version={schema.implementation_set_version}, "
        f"and abs_histogram_version={schema.abs_histogram_version}. "
        "Use a new versioned database path and recompute."
    )
