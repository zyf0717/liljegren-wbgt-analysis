import liljegren_analysis


def test_package_root_has_no_cross_subsystem_reexports() -> None:
    assert not hasattr(liljegren_analysis, "compute_daily_pairwise")
    assert not hasattr(liljegren_analysis, "HSBRunResults")
    assert not hasattr(liljegren_analysis, "extract_country_observations")


def test_removed_wbgt_only_pipeline_is_absent() -> None:
    import liljegren_analysis.cellwise as cellwise

    assert not hasattr(cellwise, "compute_cellwise_wbgt_quantiles")
