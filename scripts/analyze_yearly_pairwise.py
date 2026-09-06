#!/usr/bin/env python3
"""Pool persisted daily pairwise summaries into annual outputs."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from liljegren_analysis.comparison import (
    DEFAULT_PAIRWISE_OUTPUT_ROOT,
    analyze_yearly_pairwise,
)
from liljegren_analysis.comparison.config import REFERENCE_IMPLEMENTATION, SAMPLE_SCOPES

REPOSITORY = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=REPOSITORY / DEFAULT_PAIRWISE_OUTPUT_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPOSITORY / "data/pairwise/yearly/2025"
    )
    parser.add_argument(
        "--start-date", type=date.fromisoformat, default=date(2025, 1, 1)
    )
    parser.add_argument(
        "--end-date", type=date.fromisoformat, default=date(2025, 12, 31)
    )
    parser.add_argument(
        "--sample-scope", choices=SAMPLE_SCOPES, default="all-implementations"
    )
    parser.add_argument("--reference", default=REFERENCE_IMPLEMENTATION)
    parser.add_argument("--implementation", action="append", dest="implementations")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    outputs = analyze_yearly_pairwise(
        arguments.input_root,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        output_root=arguments.output_root,
        sample_scope=arguments.sample_scope,
        reference_implementation=arguments.reference,
        export_implementations=(
            tuple(arguments.implementations) if arguments.implementations else None
        ),
    )
    print(outputs.pairwise)


if __name__ == "__main__":
    main()
