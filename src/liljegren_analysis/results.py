"""Expose HeatStressBench campaign results as lazy DuckDB relations."""

from __future__ import annotations

from datetime import date
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
        run_id: str | int | None = None,
        implementation: str | None = None,
    ) -> duckdb.DuckDBPyRelation:
        """Return a lazy relation over the selected filesystem scope."""

        date_part = result_date.isoformat() if isinstance(result_date, date) else result_date
        run_part = f"{run_id:04d}" if isinstance(run_id, int) else run_id
        pattern = (
            self.root
            / (date_part or "*")
            / (run_part or "*")
            / (implementation or "*")
            / "results.parquet"
        ).as_posix()

        return self.connection.read_parquet(pattern, filename=True).project(
            """
            parse_path(filename)[-4]::DATE AS result_date,
            parse_path(filename)[-3] AS run_id,
            parse_path(filename)[-2] AS implementation,
            cycle_ordinal,
            row_id,
            row_ordinal,
            status,
            value_c
            """
        )
