"""Persist daily ERA5 cell-area-weighted pairwise WBGT statistics."""

from pathlib import Path

from daily_pairwise import REPOSITORY, run_pipeline
from liljegren_analysis import (
    compute_daily_area_weighted_pairwise,
    persist_daily_area_weighted_pairwise,
)


DEFAULT_DATABASE = (
    REPOSITORY / "data" / "daily_pairwise_area_weighted_v1.duckdb"
)
DEFAULT_LOG = REPOSITORY / "data" / "daily_pairwise_area_weighted.log"


def main() -> int:
    return run_pipeline(
        compute=compute_daily_area_weighted_pairwise,
        persist=persist_daily_area_weighted_pairwise,
        description=(
            "Persist daily ERA5 cell-area-weighted pairwise statistics "
            "with isolated workers."
        ),
        default_database=Path(DEFAULT_DATABASE),
        default_log=Path(DEFAULT_LOG),
        logger_name="daily_pairwise_area_weighted",
    )


if __name__ == "__main__":
    raise SystemExit(main())
