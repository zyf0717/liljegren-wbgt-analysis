#!/usr/bin/env python3
"""Compute joint-support cellwise WBGT, Tg and Tnwb quantiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise import (
    DEFAULT_DATE_WORKERS,
    DEFAULT_JOINT_IMPLEMENTATIONS,
    DEFAULT_REPLAY_WORKERS,
    compute_joint_cellwise_temperature_quantiles,
)
from liljegren_analysis.comparison.distributions import (
    DEFAULT_WBGT_HISTOGRAM,
    WBGTHistogramSpec,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hsb-run", action="append", type=Path)
    source.add_argument("--country-cache", type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing incompatible output product",
    )
    parser.add_argument(
        "--exact-validation-cells",
        type=int,
        default=100,
        help="deterministic exact-order-statistic validation sample; zero disables it",
    )
    parser.add_argument(
        "--temperature-bin-min-c",
        type=float,
        default=DEFAULT_WBGT_HISTOGRAM.minimum_c,
    )
    parser.add_argument(
        "--temperature-bin-max-c",
        type=float,
        default=DEFAULT_WBGT_HISTOGRAM.maximum_c,
    )
    parser.add_argument(
        "--temperature-bin-width-c",
        type=float,
        default=DEFAULT_WBGT_HISTOGRAM.width_c,
    )
    parser.add_argument(
        "--implementation",
        action="append",
        help=(
            "target implementation; repeat to override the default set: "
            + ", ".join(DEFAULT_JOINT_IMPLEMENTATIONS)
        ),
    )
    parser.add_argument(
        "--date-workers",
        type=_positive_int,
        default=DEFAULT_DATE_WORKERS,
        help=f"parallel joint-mask dates (default: {DEFAULT_DATE_WORKERS})",
    )
    parser.add_argument(
        "--replay-workers",
        type=_positive_int,
        default=DEFAULT_REPLAY_WORKERS,
        help=f"parallel implementation replays (default: {DEFAULT_REPLAY_WORKERS})",
    )
    arguments = parser.parse_args()
    if arguments.country_cache is not None and arguments.output_root is None:
        parser.error("--country-cache requires --output-root")
    try:
        histogram_spec = WBGTHistogramSpec(
            minimum_c=arguments.temperature_bin_min_c,
            maximum_c=arguments.temperature_bin_max_c,
            width_c=arguments.temperature_bin_width_c,
        )
    except ValueError as error:
        parser.error(str(error))
    output = compute_joint_cellwise_temperature_quantiles(
        arguments.hsb_run,
        country_cache=arguments.country_cache,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        output_root=arguments.output_root,
        histogram_spec=histogram_spec,
        exact_validation_cells=arguments.exact_validation_cells,
        implementations=(
            arguments.implementation
            if arguments.implementation is not None
            else DEFAULT_JOINT_IMPLEMENTATIONS
        ),
        date_workers=arguments.date_workers,
        replay_workers=arguments.replay_workers,
        overwrite=arguments.force,
    )
    print(output)


if __name__ == "__main__":
    main()
