"""Analysis helpers for the Liljegren WBGT comparison project."""

from .pairwise import compute_daily_pairwise, persist_daily_pairwise
from .results import CampaignResults
from .statistics import (
    pooled_difference_statistics,
    pooled_mean,
    pooled_pearson_correlation,
    pooled_sample_covariance,
    pooled_sample_variance,
)

__all__ = [
    "CampaignResults",
    "compute_daily_pairwise",
    "persist_daily_pairwise",
    "pooled_difference_statistics",
    "pooled_mean",
    "pooled_pearson_correlation",
    "pooled_sample_covariance",
    "pooled_sample_variance",
]
