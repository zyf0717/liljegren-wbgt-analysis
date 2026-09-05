"""Pairwise WBGT comparison, persistence, and pooling helpers."""

from .pairwise import (
    DailyPairwiseSummaries,
    compute_daily_pairwise,
    persist_daily_pairwise,
)
from .persistence import initialize_pairwise_output, pairwise_metadata
from .statistics import (
    histogram_quantile_bin_indices,
    histogram_quantile_statistics,
    histogram_quantiles,
    pooled_area_weighted_difference_statistics,
    pooled_difference_statistics,
    pooled_sufficient_statistics,
)

__all__ = [
    "DailyPairwiseSummaries",
    "compute_daily_pairwise",
    "histogram_quantile_bin_indices",
    "histogram_quantile_statistics",
    "histogram_quantiles",
    "initialize_pairwise_output",
    "pairwise_metadata",
    "persist_daily_pairwise",
    "pooled_area_weighted_difference_statistics",
    "pooled_difference_statistics",
    "pooled_sufficient_statistics",
]
