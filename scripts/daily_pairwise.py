"""Persist daily pairwise temperature statistics with isolated workers."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from liljegren_analysis import (
    DEFAULT_PAIRWISE_OUTPUT_ROOT,
    HSBRunResults,
    completed_pairwise_dates,
    compute_daily_pairwise,
    initialize_pairwise_output,
    persist_daily_pairwise,
)
from liljegren_analysis.comparison.config import SAMPLE_SCOPES


REPOSITORY = Path(__file__).parents[1]
DEFAULT_RUNS = (
    REPOSITORY.parent
    / "HeatStressBench"
    / ".hsb"
    / "runs"
    / "wbgt-comparison-20260902-150355",
)
SGT = ZoneInfo("Asia/Singapore")


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
        help="HSB run root; repeat for multiple roots.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY / DEFAULT_PAIRWISE_OUTPUT_ROOT,
        help="Root directory for versioned Parquet date partitions.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Progress log (default: derived from the selected modes).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the selected versioned output scope before processing",
    )
    parser.add_argument(
        "--sample-scope",
        choices=SAMPLE_SCOPES,
        default="all-implementations",
        help=(
            "Valid rows required per pair or across all discovered "
            "implementations (default: all-implementations)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Concurrent date workers (default: 1).",
    )
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    if args.log_file is None:
        args.log_file = REPOSITORY / "data" / f"daily_pairwise_{args.sample_scope}.log"
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

    output_root = args.output_root.expanduser().resolve()
    log_file = args.log_file.expanduser().resolve()
    _configure_logging(log_file, logger)
    for run in runs:
        HSBRunResults.from_hsb_run(run)
    product_root = initialize_pairwise_output(
        output_root, args.sample_scope, overwrite=args.force
    )

    days = _days(args.start_date, args.end_date)
    complete_dates = completed_pairwise_dates(output_root, args.sample_scope)
    pending_days = [result_date for result_date in days if result_date not in complete_dates]
    logger.info(
        "Processing %d dates quantities=wbgt,tg,tnwb quantity_scope=joint "
        "sample_scope=%s with %d workers; pending=%d skipped=%d",
        len(days),
        args.sample_scope,
        args.workers,
        len(pending_days),
        len(days) - len(pending_days),
    )
    logger.info("Output: %s", product_root)
    logger.info("Log: %s", log_file)

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
                args.sample_scope,
            )
            for result_date in pending_days
        ]

        for completed, future in enumerate(as_completed(futures), start=1):
            result_date, summaries, worker_pid, compute_seconds = future.result()
            persist_daily_pairwise(output_root, result_date, summaries, args.sample_scope)
            logger.info(
                "Processed %s worker_pid=%d compute=%.1fs [%d/%d]",
                result_date, worker_pid, compute_seconds, completed, len(pending_days),
            )
        logger.info("Saved pairwise statistics to %s", product_root)
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
