"""Persist daily pairwise temperature statistics with isolated workers."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb


from liljegren_analysis import (
    compute_daily_pairwise,
    persist_daily_pairwise,
)
from liljegren_analysis.comparison.sql import SAMPLE_SCOPES


REPOSITORY = Path(__file__).parents[1]
DEFAULT_RUNS = (
    REPOSITORY / ".hsb" / "runs" / "wbgt-comparison-20260823-181813",
)
SGT = ZoneInfo("Asia/Singapore")
_OUTPUT_STEMS = {
    "pairwise": "daily_pairwise_v5",
    "all-implementations": "daily_pairwise_all_implementations_v5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Persist row- and area-weighted daily pairwise summaries with "
            "isolated date workers."
        ),
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
        help="Output DuckDB file (default: derived from the selected modes).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Progress log (default: derived from the selected modes).",
    )
    parser.add_argument(
        "--sample-scope",
        choices=SAMPLE_SCOPES,
        default="pairwise",
        help=(
            "Valid rows required per pair or across all seven "
            "implementations (default: pairwise)."
        ),
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
    stem = _OUTPUT_STEMS[args.sample_scope]
    if args.database is None:
        args.database = REPOSITORY / "data" / f"{stem}.duckdb"
    if args.log_file is None:
        args.log_file = REPOSITORY / "data" / f"{stem}.log"
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


def _configure_logging(log_file: Path, logger: logging.Logger) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = _SgtFormatter("[%(asctime)s SGT] %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def run_pipeline() -> int:
    args = parse_args()
    logger = logging.getLogger(
        f"daily_pairwise.{args.sample_scope}"
    )
    runs = tuple(
        path.expanduser().resolve()
        for path in (args.run if args.run is not None else DEFAULT_RUNS)
    )
    if not runs:
        raise ValueError("runs is empty")

    database = args.database.expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file.expanduser().resolve()
    _configure_logging(log_file, logger)

    days = _days(args.start_date, args.end_date)
    logger.info(
        "Processing %d dates quantities=wbgt,tg,tnwb quantity_scope=joint "
        "sample_scope=%s with %d workers "
        "and %d threads per worker",
        len(days),
        args.sample_scope,
        args.workers,
        args.threads_per_worker,
    )
    logger.info("Output: %s", database)
    logger.info("Log: %s", log_file)

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
                args.sample_scope,
            )
            for result_date in days
        ]

        # The controller is the only DuckDB writer; every completed date is
        # persisted synchronously before the next result is accepted.
        with duckdb.connect(str(database)) as writer:
            for completed, future in enumerate(as_completed(futures), start=1):
                result_date, summaries, worker_pid, compute_seconds = (
                    future.result()
                )
                persist_daily_pairwise(
                    writer,
                    result_date,
                    summaries,
                    args.sample_scope,
                )
                logger.info(
                    "Processed %s worker_pid=%d compute=%.1fs [%d/%d]",
                    result_date,
                    worker_pid,
                    compute_seconds,
                    completed,
                    len(days),
                )
            writer.execute("CHECKPOINT")

        logger.info("Saved pairwise statistics to %s", database)
        return 0
    except Exception:
        logger.exception("Pairwise processing failed")
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        logging.shutdown()


def main() -> int:
    return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
