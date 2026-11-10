"""Restore Japan's original two-panel layout using current hourly results.

Run with --input-root <run>/country/jpn/hourly_classification and
--population-weights-path <run>/population/population.parquet.
"""

import argparse
from pathlib import Path

from analysis.cellwise.iso7243_plots import plot_country_non_consensus_stack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--population-weights-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    print(plot_country_non_consensus_stack(
        args.input_root, country_iso_a3="JPN",
        population_weights_path=args.population_weights_path,
        output_directory=args.output_directory,
    ))


if __name__ == "__main__":
    main()
