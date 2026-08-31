"""Analysis helpers for the Liljegren WBGT comparison project."""

from .country import (
    CountryExtraction,
    CountryExtractionProgress,
    extract_countries_observations,
    extract_country_observations,
)
from .comparison import (
    DailyPairwiseSummaries,
    compute_daily_pairwise,
    histogram_quantile_statistics,
    histogram_quantiles,
    persist_daily_pairwise,
    pooled_area_weighted_difference_statistics,
    pooled_difference_statistics,
    pooled_sufficient_statistics,
)
from .cellwise import (
    analyze_cellwise_quantile_spread,
    compute_cellwise_wbgt_quantiles,
    compute_joint_cellwise_temperature_quantiles,
)
from .results import (
    RESULT_QUANTITIES,
    RESULT_QUANTITY_SPECS,
    RESULT_SCHEMA_ID,
    HSBRunResults,
    ResultQuantity,
    ResultQuantitySpec,
)

__all__ = [
    "HSBRunResults",
    "CountryExtraction",
    "CountryExtractionProgress",
    "DailyPairwiseSummaries",
    "RESULT_QUANTITIES",
    "RESULT_QUANTITY_SPECS",
    "RESULT_SCHEMA_ID",
    "ResultQuantity",
    "ResultQuantitySpec",
    "compute_daily_pairwise",
    "compute_cellwise_wbgt_quantiles",
    "compute_joint_cellwise_temperature_quantiles",
    "analyze_cellwise_quantile_spread",
    "extract_countries_observations",
    "extract_country_observations",
    "histogram_quantile_statistics",
    "histogram_quantiles",
    "persist_daily_pairwise",
    "pooled_area_weighted_difference_statistics",
    "pooled_difference_statistics",
    "pooled_sufficient_statistics",
]
