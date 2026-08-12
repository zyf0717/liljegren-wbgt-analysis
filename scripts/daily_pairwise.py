"""Persist daily pairwise WBGT statistics with isolated date workers."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb


REPOSITORY = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from liljegren_analysis import (  # noqa: E402
    compute_daily_pairwise,
    persist_daily_pairwise,
)


DEFAULT_RUNS = (
    REPOSITORY / ".hsb" / "campaigns" / "era5-2025-all-six-m2-w20-v8",
    REPOSITORY
    / ".hsb"
    / "runs"
    / "era5-2025-meteor-m4-w18"
    / "wbgt-comparison-20260810-231240",
)
DEFAULT_DATABASE = REPOSITORY / "data" / "daily_pairwise.duckdb"
DEFAULT_LOG = REPOSITORY / "data" / "daily_pairwise.log"
SGT = ZoneInfo("Asia/Singapore")
LOGGER = logging.getLogger("daily_pairwise")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persist daily pairwise statistics with isolated workers.",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=date(2025, 1, 1),
        help="Inclusive ISO date (default: 2025-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=date(2025, 12, 31),
        help="Inclusive ISO date (default: 2025-12-31).",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        help="Campaign/run root; repeat for multiple roots.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"Output DuckDB file (default: {DEFAULT_DATABASE}).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG,
        help=f"Append progress logs to this file (default: {DEFAULT_LOG}).",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="Concurrent date workers (default: 4).",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=_positive_int,
        default=4,
        help="DuckDB threads per worker (default: 4).",
    )
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    return args


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from error


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _days(start: date, end: date) -> list[date]:
    return [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    ]


class _SgtFormatter(logging.Formatter):
    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        timestamp = datetime.fromtimestamp(record.created, SGT)
        return timestamp.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def _configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = _SgtFormatter("[%(asctime)s SGT] %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def main() -> int:
    args = parse_args()
    runs = tuple(
        path.expanduser().resolve()
        for path in (args.run if args.run is not None else DEFAULT_RUNS)
    )
    if not runs:
        raise ValueError("runs is empty")

    database = args.database.expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file.expanduser().resolve()
    _configure_logging(log_file)

    days = _days(args.start_date, args.end_date)
    LOGGER.info(
        "Processing %d dates with %d workers and %d threads per worker",
        len(days),
        args.workers,
        args.threads_per_worker,
    )
    LOGGER.info("Output: %s", database)
    LOGGER.info("Log: %s", log_file)

    # A fresh PID per date prevents the DuckDB process-state slowdown observed
    # when a worker handled successive dates.
    executor = ProcessPoolExecutor(
        max_workers=args.workers,
        max_tasks_per_child=1,
    )
    try:
        futures = [
            executor.submit(
                compute_daily_pairwise,
                result_date,
                runs,
                args.threads_per_worker,
            )
            for result_date in days
        ]

        # The controller is the only DuckDB writer; every completed date is
        # persisted synchronously before the next result is accepted.
        with duckdb.connect(str(database)) as writer:
            for completed, future in enumerate(as_completed(futures), start=1):
                result_date, result, worker_pid, compute_seconds = future.result()
                persist_daily_pairwise(writer, result_date, result)
                LOGGER.info(
                    "Processed %s worker_pid=%d compute=%.1fs [%d/%d]",
                    result_date,
                    worker_pid,
                    compute_seconds,
                    completed,
                    len(days),
                )
            writer.execute("CHECKPOINT")

        LOGGER.info("Saved daily_pairwise to %s", database)
        return 0
    except Exception:
        LOGGER.exception("daily_pairwise failed")
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
