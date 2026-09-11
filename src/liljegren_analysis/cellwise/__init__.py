"""Streaming cellwise temporal temperature distribution summaries."""

from .joint_pipeline import (
    DEFAULT_DATE_WORKERS,
    DEFAULT_JOINT_IMPLEMENTATIONS,
    DEFAULT_REPLAY_WORKERS,
    compute_joint_cellwise_temperature_quantiles,
)
from .paths import (
    DEFAULT_ISO7243_OUTPUT_ROOT,
    DEFAULT_JOINT_OUTPUT_ROOT,
    DEFAULT_QUANTILE_SPREAD_INPUT_ROOT,
    DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT,
)
from .spread import (
    QuantileSpreadOutputs,
    analyze_cellwise_quantile_spread,
)
from .population import PopulationWeightOutputs, compute_era5_population_weights
from .iso7243 import (
    ISO7243ClassificationOutputs,
    analyze_cellwise_iso7243_classification,
)
from .iso7243_plots import (
    plot_cellwise_iso7243_classification,
    summarize_cellwise_iso7243_classification,
)
from .non_consensus import (
    NonConsensusOutputs,
    analyze_cellwise_iso7243_non_consensus,
)
from .non_consensus_plots import plot_cellwise_iso7243_non_consensus

__all__ = [
    "QuantileSpreadOutputs",
    "PopulationWeightOutputs",
    "ISO7243ClassificationOutputs",
    "NonConsensusOutputs",
    "DEFAULT_DATE_WORKERS",
    "DEFAULT_ISO7243_OUTPUT_ROOT",
    "DEFAULT_JOINT_IMPLEMENTATIONS",
    "DEFAULT_JOINT_OUTPUT_ROOT",
    "DEFAULT_QUANTILE_SPREAD_INPUT_ROOT",
    "DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT",
    "DEFAULT_REPLAY_WORKERS",
    "analyze_cellwise_quantile_spread",
    "compute_joint_cellwise_temperature_quantiles",
    "compute_era5_population_weights",
    "analyze_cellwise_iso7243_classification",
    "analyze_cellwise_iso7243_non_consensus",
    "plot_cellwise_iso7243_non_consensus",
    "plot_cellwise_iso7243_classification",
    "summarize_cellwise_iso7243_classification",
]
