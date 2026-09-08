"""Common arguments for the two entry points to hourly WBGT classification."""

import argparse
from pathlib import Path

from ..comparison.config import SAMPLE_SCOPES

from .iso7243 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATE_WORKERS,
    DEFAULT_OUTPUT_ROOT,
    analyze_cellwise_iso7243_classification,
)
from .joint_pipeline import DEFAULT_JOINT_IMPLEMENTATIONS


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def run(*, non_consensus: bool = False) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsb-run", action="append", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--thresholds-c", required=True, nargs="+", type=float)
    parser.add_argument("--implementation", action="append")
    parser.add_argument("--sample-scope", choices=SAMPLE_SCOPES, default="all-implementations",
                        help="validity for pairwise counts; non-consensus always uses all targets")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--date-workers", type=_positive_int, default=DEFAULT_DATE_WORKERS)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true",
                        help="rebuild final artifacts; allow a new checkpoint configuration")
    args = parser.parse_args()
    outputs = analyze_cellwise_iso7243_classification(
        args.hsb_run, start_date=args.start_date, end_date=args.end_date,
        thresholds_c=args.thresholds_c,
        implementations=args.implementation or DEFAULT_JOINT_IMPLEMENTATIONS,
        sample_scope=args.sample_scope,
        output_root=args.output_root, date_workers=args.date_workers,
        batch_size=args.batch_size, force=args.force,
    )
    print(outputs.cellwise_non_consensus if non_consensus else outputs.cell_pair_differences)
