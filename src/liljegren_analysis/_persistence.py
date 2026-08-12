"""Versioned persistence for date-partitioned pairwise statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb
import pyarrow as pa


@dataclass(frozen=True)
class PairwiseTableSchema:
    """Describe one persisted pairwise-statistics table."""

    table_name: str
    version: int
    columns: tuple[tuple[str, str, bool], ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.columns)


def persist_date_partition(
    connection: duckdb.DuckDBPyConnection,
    result_date: date,
    result: pa.Table,
    schema: PairwiseTableSchema,
) -> None:
    """Transactionally replace one date after validating the schema contract."""

    _validate_result(result_date, result, schema)
    connection.register("_day_result", result)
    connection.execute("BEGIN")
    try:
        _ensure_schema(connection, schema)
        connection.execute(
            f"DELETE FROM {schema.table_name} WHERE result_date = ?",
            [result_date],
        )
        columns = ", ".join(schema.column_names)
        connection.execute(
            f"""
            INSERT INTO {schema.table_name} ({columns})
            SELECT {columns}
            FROM _day_result
            """
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.unregister("_day_result")


def _validate_result(
    result_date: date,
    result: pa.Table,
    schema: PairwiseTableSchema,
) -> None:
    expected = set(schema.column_names)
    actual = set(result.column_names)
    if actual != expected or len(result.column_names) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{schema.table_name} result schema differs: "
            f"missing={missing}, extra={extra}"
        )

    dates = result.column("result_date").to_pylist()
    if any(value != result_date for value in dates):
        raise ValueError(
            f"{schema.table_name} result contains a date other than {result_date}"
        )


def _ensure_schema(
    connection: duckdb.DuckDBPyConnection,
    schema: PairwiseTableSchema,
) -> None:
    table_exists = _table_exists(connection, schema.table_name)
    metadata_exists = _table_exists(connection, "pairwise_schema")

    if not table_exists and not metadata_exists:
        _create_schema(connection, schema)
        return
    if not table_exists or not metadata_exists:
        raise ValueError(_incompatible_message(schema))

    metadata = connection.execute(
        "SELECT table_name, schema_version FROM pairwise_schema"
    ).fetchall()
    if metadata != [(schema.table_name, schema.version)]:
        raise ValueError(_incompatible_message(schema))

    actual_columns = connection.execute(
        """
        SELECT column_name, data_type, is_nullable = 'YES'
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = ?
        ORDER BY ordinal_position
        """,
        [schema.table_name],
    ).fetchall()
    if actual_columns != list(schema.columns):
        raise ValueError(_incompatible_message(schema))


def _create_schema(
    connection: duckdb.DuckDBPyConnection,
    schema: PairwiseTableSchema,
) -> None:
    definitions = [
        f"{name} {data_type}{'' if nullable else ' NOT NULL'}"
        for name, data_type, nullable in schema.columns
    ]
    definitions.append(
        "PRIMARY KEY (result_date, implementation_a, implementation_b)"
    )
    connection.execute(
        """
        CREATE TABLE pairwise_schema (
            table_name VARCHAR PRIMARY KEY,
            schema_version INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO pairwise_schema VALUES (?, ?)",
        [schema.table_name, schema.version],
    )
    connection.execute(
        f"CREATE TABLE {schema.table_name} ({', '.join(definitions)})"
    )


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


def _incompatible_message(schema: PairwiseTableSchema) -> str:
    return (
        f"Incompatible {schema.table_name} database schema; expected version "
        f"{schema.version}. Use a new versioned database path and recompute."
    )
