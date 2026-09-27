"""End-to-end pairwise workflows shared by scripts and notebooks."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from ..config import (
    DEFAULT_PAIRWISE_IMPLEMENTATIONS,
    SAMPLE_SCOPES,
    SampleScope,
    implementation_names,
)
from ..io.artifacts import atomic_json, input_fingerprint
from ..results import date_range, discover_results
from .pairwise import compute_daily_pairwise, persist_daily_pairwise
from .persistence import (
    completed_pairwise_dates,
    initialize_pairwise_output,
)


def run_daily_pairwise(
    hsb_run_paths: str | Path | Sequence[str | Path],
    *,
    start_date: date,
    end_date: date,
    output_root: str | Path,
    sample_scope: SampleScope = "all-implementations",
    implementations: Sequence[str] | None = None,
    workers: int = 1,
    overwrite: bool = False,
) -> Path:
    """Compute and atomically publish pending daily pairwise partitions."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    selected = (
        implementation_names(
            DEFAULT_PAIRWISE_IMPLEMENTATIONS
            if implementations is None
            else implementations
        )
        if sample_scope == "all-implementations" or implementations is not None
        else None
    )
    if workers < 1:
        raise ValueError("workers must be positive")
    hsb_run_paths = (
        (hsb_run_paths,) if isinstance(hsb_run_paths, (str, Path)) else hsb_run_paths
    )
    runs = tuple(Path(path).expanduser().resolve() for path in hsb_run_paths)
    days = date_range(start_date, end_date)
    inventory = discover_results(runs, days, selected)
    by_date = {
        day: tuple(item for item in inventory if item.result_date == day)
        for day in days
    }
    files = tuple(sorted(item.path for item in inventory))
    output = Path(output_root).expanduser().resolve()
    product = initialize_pairwise_output(
        output,
        sample_scope,
        implementations=selected,
        overwrite=overwrite,
    )
    request = {
        "runs": [str(run) for run in runs],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "sample_scope": sample_scope,
        "implementations": list(selected) if selected is not None else None,
        "input_fingerprint": input_fingerprint(files),
    }
    manifest = output / ".work" / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != request:
        raise ValueError(
            "Incompatible pairwise inputs or settings; use a new output or overwrite"
        )
    atomic_json(request, manifest)
    complete = completed_pairwise_dates(output)
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

    def publish(day, summaries):
        persist_daily_pairwise(output, day, summaries)
        logger.info("Processed %s", day)

    if workers == 1:
        for day in pending:
            publish(
                day,
                compute_daily_pairwise(
                    day, by_date[day], sample_scope, implementations=selected
                ),
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)), max_tasks_per_child=1
        ) as executor:
            futures = {
                executor.submit(
                    compute_daily_pairwise,
                    day,
                    by_date[day],
                    sample_scope,
                    implementations=selected,
                ): day
                for day in pending
            }
            for future in as_completed(futures):
                publish(futures[future], future.result())
    if input_fingerprint(files) != request["input_fingerprint"]:
        raise ValueError("Inputs changed during pairwise analysis")
    return product
