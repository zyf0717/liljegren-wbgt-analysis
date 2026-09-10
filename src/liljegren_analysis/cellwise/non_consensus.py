"""Non-consensus entry point for the shared hourly WBGT classification engine."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..comparison.config import SampleScope

from .iso7243 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATE_WORKERS,
    DEFAULT_OUTPUT_ROOT,
    analyze_cellwise_iso7243_classification,
)
from .joint_pipeline import DEFAULT_JOINT_IMPLEMENTATIONS


@dataclass(frozen=True)
class NonConsensusOutputs:
    cellwise_non_consensus: Path
    cell_lookup: Path
    classes: Path
    metadata: Path


def analyze_cellwise_iso7243_non_consensus(
    hsb_run_paths: str | Path | Sequence[str | Path] | None = None,
    *,
    country_cache: str | Path | None = None,
    start_date: str | date,
    end_date: str | date,
    thresholds_c: Sequence[float],
    implementations: Sequence[str] = DEFAULT_JOINT_IMPLEMENTATIONS,
    sample_scope: SampleScope = "all-implementations",
    output_root: str | Path | None = None,
    date_workers: int = DEFAULT_DATE_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> NonConsensusOutputs:
    """Compute/reuse both hourly products and return the non-consensus paths."""
    outputs = analyze_cellwise_iso7243_classification(
        hsb_run_paths, country_cache=country_cache,
        start_date=start_date, end_date=end_date,
        thresholds_c=thresholds_c, implementations=implementations, sample_scope=sample_scope,
        output_root=output_root, date_workers=date_workers,
        batch_size=batch_size, force=force,
    )
    return NonConsensusOutputs(outputs.cellwise_non_consensus, outputs.cell_lookup,
                               outputs.classes, outputs.metadata)
