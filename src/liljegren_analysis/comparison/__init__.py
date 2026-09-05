"""Pairwise WBGT comparison, persistence, and pooling helpers."""

from .pairwise import (
    DailyPairwiseSummaries,
    compute_daily_pairwise,
    persist_daily_pairwise,
)
from .persistence import (
    DEFAULT_PAIRWISE_OUTPUT_ROOT,
    completed_pairwise_dates,
    initialize_pairwise_output,
    pairwise_metadata,
    pairwise_product_root,
)
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
    "DEFAULT_PAIRWISE_OUTPUT_ROOT",
    "compute_daily_pairwise",
    "completed_pairwise_dates",
    "histogram_quantile_bin_indices",
    "histogram_quantile_statistics",
    "histogram_quantiles",
    "initialize_pairwise_output",
    "pairwise_metadata",
    "pairwise_product_root",
    "persist_daily_pairwise",
    "pooled_area_weighted_difference_statistics",
    "pooled_difference_statistics",
    "pooled_sufficient_statistics",
]
