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
    compute_era5_population_weights,
    analyze_cellwise_iso7243_classification,
    analyze_cellwise_iso7243_non_consensus,
    plot_cellwise_iso7243_non_consensus,
    plot_cellwise_iso7243_classification,
)
from .classification import (
    ClassificationAnalysis,
    Era5ClassificationAnalysis,
    NasaPowerClassificationAnalysis,
    analyze_era5_country_cache,
    analyze_nasa_power_run,
    analyze_wbgt_classification,
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
    "ClassificationAnalysis",
    "DailyPairwiseSummaries",
    "Era5ClassificationAnalysis",
    "RESULT_QUANTITIES",
    "RESULT_QUANTITY_SPECS",
    "RESULT_SCHEMA_ID",
    "ResultQuantity",
    "ResultQuantitySpec",
    "NasaPowerClassificationAnalysis",
    "analyze_era5_country_cache",
    "analyze_nasa_power_run",
    "analyze_wbgt_classification",
    "compute_daily_pairwise",
    "compute_cellwise_wbgt_quantiles",
    "compute_joint_cellwise_temperature_quantiles",
    "compute_era5_population_weights",
    "analyze_cellwise_iso7243_classification",
    "analyze_cellwise_iso7243_non_consensus",
    "plot_cellwise_iso7243_non_consensus",
    "plot_cellwise_iso7243_classification",
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
