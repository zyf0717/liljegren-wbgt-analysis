#!/usr/bin/env python3
"""Analyze persisted cellwise temperature quantile spread without shard scans."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise.spread import analyze_cellwise_quantile_spread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/cellwise_temperature_quantiles/2025-joint-v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cellwise_temperature_quantiles/analysis/wbgt"),
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--quantity", choices=("wbgt", "tg", "tnwb"), default="wbgt")
    arguments = parser.parse_args()
    outputs = analyze_cellwise_quantile_spread(
        arguments.input_root,
        output_root=arguments.output_root,
        quantity=arguments.quantity,
        sample_size=arguments.sample_size,
    )
    print(outputs.cellwise_spread)


if __name__ == "__main__":
    main()
