def test_public_api_reexports_canonical_objects() -> None:
    from liljegren_analysis import (
        compute_daily_pairwise,
        extract_country_observations,
        pooled_sufficient_statistics,
    )
    from liljegren_analysis.comparison.pairwise import (
        compute_daily_pairwise as canonical_compute_daily_pairwise,
    )
    from liljegren_analysis.comparison.statistics import (
        pooled_sufficient_statistics as canonical_pooled_sufficient_statistics,
    )
    from liljegren_analysis.country.extraction import (
        extract_country_observations as canonical_extract_country_observations,
    )
    assert extract_country_observations is canonical_extract_country_observations
    assert compute_daily_pairwise is canonical_compute_daily_pairwise
    assert pooled_sufficient_statistics is canonical_pooled_sufficient_statistics
