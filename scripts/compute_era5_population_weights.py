#!/usr/bin/env python3
"""Aggregate LandScan population counts onto the global ERA5 grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from liljegren_analysis.cellwise.population import (
    DEFAULT_LANDSCAN_PATH,
    DEFAULT_OUTPUT_ROOT,
    compute_era5_population_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-raster", type=Path, default=DEFAULT_LANDSCAN_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    outputs = compute_era5_population_weights(
        arguments.input_raster,
        output_root=arguments.output_root,
        force=arguments.force,
    )
    print(outputs.population)


if __name__ == "__main__":
    main()
