"""Expose HeatStressBench campaign results as lazy DuckDB relations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from typing import Self

import duckdb


class CampaignResults:
    """Scan ``results/<date>/<run>/<implementation>/results.parquet``."""

    def __init__(
        self,
        root: str | Path,
        *,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"results directory does not exist: {self.root}")
        self.connection = connection or duckdb.connect()

    @classmethod
    def from_campaign(
        cls,
        campaign: str | Path,
        *,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> Self:
        """Create a scanner from a directory containing ``results``."""

        return cls(Path(campaign) / "results", connection=connection)

    def scan(
        self,
        *,
        result_date: str | date | None = None,
        dates: Sequence[str | date] | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        run_id: str | int | None = None,
        run_ids: Sequence[str | int] | None = None,
        implementation: str | None = None,
        implementations: Sequence[str] | None = None,
        include_filename: bool = False,
    ) -> duckdb.DuckDBPyRelation:
        """Return a lazy DuckDB relation over a filesystem-scoped result set.

        Parameters
        ----------
        result_date:
            One exact ``YYYY-MM-DD`` directory. Mutually exclusive with the
            explicit-date and range arguments.
        dates:
            Non-contiguous exact date directories.
        start_date, end_date:
            Inclusive ``YYYY-MM-DD`` range. Both must be provided. Exact daily
            paths are generated so files outside the range are not scanned.
        run_id:
            One run/shard directory. Integers are padded to four digits, so
            ``1`` selects ``0001``. Strings are used unchanged.
        run_ids:
            Multiple exact run/shard directories. Mutually exclusive with
            ``run_id``.
        implementation:
            One implementation directory, such as ``"heatstress"`` or
            ``"meteor"``.
        implementations:
            Multiple exact implementation directories. Mutually exclusive with
            ``implementation``.
        include_filename:
            Include DuckDB's source ``filename`` column for provenance.

        Returns
        -------
        duckdb.DuckDBPyRelation
            A lazy relation with ``result_date``, ``run_id``,
            ``implementation``, ``cycle_ordinal``, ``row_id``,
            ``row_ordinal``, ``status``, and ``value_c``.

        Notes
        -----
        Any omitted scope selects ``*`` at that directory level. Omitting all
        scopes targets the complete result tree. No rows are read until the
        relation is queried or materialized.
        """

        date_parts = _date_parts(result_date, dates, start_date, end_date)
        run_parts = _parts(run_id, run_ids, "run_id", _run_part)
        implementation_parts = _parts(
            implementation,
            implementations,
            "implementation",
            str,
        )
        patterns = [
            (
                self.root
                / date_part
                / run_part
                / implementation_part
                / "results.parquet"
            ).as_posix()
            for date_part, run_part, implementation_part in product(
                date_parts,
                run_parts,
                implementation_parts,
            )
        ]

        relation = self.connection.read_parquet(
            patterns[0] if len(patterns) == 1 else patterns,
            filename=True,
        )
        columns = """
            parse_path(filename)[-4]::DATE AS result_date,
            parse_path(filename)[-3] AS run_id,
            parse_path(filename)[-2] AS implementation,
            cycle_ordinal,
            row_id,
            row_ordinal,
            status,
            value_c
        """
        if include_filename:
            columns += ", filename"
        return relation.project(columns)


def _date_parts(
    result_date: str | date | None,
    dates: Sequence[str | date] | None,
    start_date: str | date | None,
    end_date: str | date | None,
) -> list[str]:
    modes = sum(
        (
            result_date is not None,
            dates is not None,
            start_date is not None or end_date is not None,
        )
    )
    if modes > 1:
        raise ValueError("use only one of result_date, dates, or start_date/end_date")
    if result_date is not None:
        return [_as_date(result_date).isoformat()]
    if dates is not None:
        return _many(dates, "dates", lambda value: _as_date(value).isoformat())
    if (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be provided together")
    if start_date is None:
        return ["*"]

    start = _as_date(start_date)
    end = _as_date(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    ]


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _run_part(value: str | int) -> str:
    return f"{value:04d}" if isinstance(value, int) else value


def _parts[T](
    value: T | None,
    values: Sequence[T] | None,
    name: str,
    render: Callable[[T], str],
) -> list[str]:
    if value is not None and values is not None:
        raise ValueError(f"{name} and {name}s cannot be combined")
    if values is not None:
        return _many(values, f"{name}s", render)
    return [render(value)] if value is not None else ["*"]


def _many[T](values: Sequence[T], name: str, render: Callable[[T], str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not a string")
    rendered = list(dict.fromkeys(render(value) for value in values))
    if not rendered:
        raise ValueError(f"{name} must not be empty")
    return rendered
