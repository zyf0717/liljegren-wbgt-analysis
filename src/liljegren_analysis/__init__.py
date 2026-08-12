"""Analysis helpers for the Liljegren WBGT comparison project."""

from .area_weighted import (
    compute_daily_area_weighted_pairwise,
    persist_daily_area_weighted_pairwise,
)
from .pairwise import compute_daily_pairwise, persist_daily_pairwise
from .results import CampaignResults
from .statistics import (
    pooled_area_weighted_difference_statistics,
    pooled_difference_statistics,
)

__all__ = [
    "CampaignResults",
    "compute_daily_area_weighted_pairwise",
    "compute_daily_pairwise",
    "persist_daily_area_weighted_pairwise",
    "persist_daily_pairwise",
    "pooled_area_weighted_difference_statistics",
    "pooled_difference_statistics",
]
