"""End-to-end pairwise workflows shared by scripts and notebooks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from ..results import HSBRunResults
from .config import SAMPLE_SCOPES, SampleScope
from .pairwise import compute_daily_pairwise, persist_daily_pairwise
from .persistence import (
    DEFAULT_PAIRWISE_OUTPUT_ROOT,
    completed_pairwise_dates,
    initialize_pairwise_output,
)


def run_daily_pairwise(
    hsb_run_paths: str | Path | Sequence[str | Path],
    *,
    start_date: date,
    end_date: date,
    output_root: str | Path = DEFAULT_PAIRWISE_OUTPUT_ROOT,
    sample_scope: SampleScope = "all-implementations",
    workers: int = 1,
    overwrite: bool = False,
) -> Path:
    """Compute and atomically publish pending daily pairwise partitions."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    if workers < 1:
        raise ValueError("workers must be positive")
    raw_runs = (
        (hsb_run_paths,)
        if isinstance(hsb_run_paths, (str, Path))
        else tuple(hsb_run_paths)
    )
    if not raw_runs:
        raise ValueError("hsb_run_paths must not be empty")
    runs = tuple(Path(path).expanduser().resolve() for path in raw_runs)
    if len(runs) != len(set(runs)):
        raise ValueError("hsb_run_paths must be unique")
    for run in runs:
        HSBRunResults.from_hsb_run(run)

    output = Path(output_root).expanduser().resolve()
    product = initialize_pairwise_output(output, sample_scope, overwrite=overwrite)
    days = tuple(
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    )
    complete = completed_pairwise_dates(output, sample_scope)
    pending = tuple(day for day in days if day not in complete)
    logger = logging.getLogger(__name__)
    logger.info(
        "Pairwise dates=%d pending=%d skipped=%d workers=%d output=%s",
        len(days),
        len(pending),
        len(days) - len(pending),
        workers,
        product,
    )
    if not pending:
        return product
    executor = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1)
    try:
        futures = [
            executor.submit(compute_daily_pairwise, day, runs, sample_scope)
            for day in pending
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result_date, summaries, worker_pid, elapsed = future.result()
            persist_daily_pairwise(output, result_date, summaries, sample_scope)
            logger.info(
                "Processed %s worker_pid=%d compute=%.1fs [%d/%d]",
                result_date,
                worker_pid,
                elapsed,
                completed,
                len(pending),
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return product
