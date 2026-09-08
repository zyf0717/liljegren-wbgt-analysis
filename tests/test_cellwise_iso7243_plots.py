import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import pytest

from iso7243_fixtures import CELLS, reference_pairwise_counts, write_days
from liljegren_analysis.cellwise.iso7243 import analyze_cellwise_iso7243_classification
from liljegren_analysis.cellwise.iso7243_plots import (
    _accumulate_population_affected,
    configure_percent_colorbar,
    plot_cellwise_iso7243_classification,
    summarize_cellwise_iso7243_classification,
)


def test_hourly_rates_weighting_and_endpoint_bins(tmp_path):
    run = tmp_path / "run"
    write_days(run, 12, complex_rows=False)
    out = analyze_cellwise_iso7243_classification(
        run, start_date="2025-01-01", end_date="2025-01-12",
        thresholds_c=[25., 30.], output_root=tmp_path / "out", date_workers=1)
    weights = tmp_path / "population.parquet"
    pq.write_table(pa.table({"source_cell_id": CELLS, "population": [10, 0, 1000]}), weights)
    summary = summarize_cellwise_iso7243_classification(
        out.metadata.parent, population_weights_path=weights)
    np.testing.assert_allclose(summary.changed_fraction, [1 / 3, 0, np.nan], equal_nan=True)
    assert summary.histogram_shares.shape == (3, 20)
    np.testing.assert_allclose(summary.histogram_shares.sum(axis=1), 1.)
    # Five of fifteen pairs disagree for the first cell; unsupported third cell excluded.
    assert summary.histogram_shares[0, -1] == pytest.approx(5 / 30)
    assert summary.histogram_shares[2, -1] == pytest.approx(5 / 15)
    np.testing.assert_allclose(summary.pair_changed_fraction[0], [1, 0, np.nan], equal_nan=True)
    np.testing.assert_array_equal(summary.population_affected[:5], 10)
    np.testing.assert_array_equal(summary.population_affected[5:-1], 0)
    np.testing.assert_array_equal(summary.population_affected[-1], 10)
    assert summary.total_population == 1010
    figures = plot_cellwise_iso7243_classification(out.metadata.parent, population_weights_path=weights)
    assert figures.png.stat().st_size > 0
    assert figures.pairwise_png.stat().st_size > 0
    assert figures.population_affected_png.stat().st_size > 0
    assert figures.non_consensus_population_png.stat().st_size > 0
    population_csv = pcsv.read_csv(figures.population_affected_csv)
    assert population_csv.column_names == [
        "comparison_type", "implementation_a", "implementation_b",
        "estimated_population_affected_ge_5_percent",
        "percent_total_population_ge_5_percent",
        "estimated_population_affected_ge_10_percent",
        "percent_total_population_ge_10_percent",
        "estimated_population_affected_ge_20_percent",
        "percent_total_population_ge_20_percent",
        "estimated_population_affected_ge_30_percent",
        "percent_total_population_ge_30_percent",
    ]
    assert population_csv.num_rows == 16
    for threshold in (5, 10, 20, 30):
        assert population_csv[
            f"estimated_population_affected_ge_{threshold}_percent"
        ][0].as_py() == 10
        assert population_csv[
            f"percent_total_population_ge_{threshold}_percent"
        ][0].as_py() == pytest.approx(1000 / 1010)


def test_legacy_product_rejected(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"quantiles": ["p50", "p90", "p95", "p99"]}))
    with pytest.raises(ValueError, match="hourly"):
        summarize_cellwise_iso7243_classification(tmp_path)


def test_pairwise_plot_uses_each_pairs_support(tmp_path):
    run = tmp_path / "run"
    write_days(run, 1)
    out = analyze_cellwise_iso7243_classification(
        run, start_date="2025-01-01", end_date="2025-01-01",
        thresholds_c=[25., 30.], sample_scope="pairwise",
        output_root=tmp_path / "out", date_workers=1)
    weights = tmp_path / "population.parquet"
    pq.write_table(pa.table({"source_cell_id": CELLS, "population": [10, 0, 1000]}), weights)
    summary = summarize_cellwise_iso7243_classification(out.metadata.parent, population_weights_path=weights)
    reference = reference_pairwise_counts(run, ["2025-01-01"])
    expected = np.divide(reference[:, 1] + reference[:, 2], reference[:, 0],
                         out=np.full((15, 3), np.nan), where=reference[:, 0] > 0)
    np.testing.assert_allclose(summary.pair_changed_fraction, expected, equal_nan=True)
    np.testing.assert_allclose(summary.changed_fraction, np.nanmean(expected, axis=0))
    assert summary.pair_changed_fraction[0, 0] == pytest.approx(3 / 24)
    # Cell 2 has no all-target support, but ten pairs have valid data there.
    assert np.isfinite(summary.pair_changed_fraction[:, 2]).sum() == 10
    assert summary.changed_fraction[2] == 0
    np.testing.assert_allclose(summary.histogram_shares.sum(axis=1), 1.)
    population = np.asarray([10, 0, 1000])
    expected_population = np.asarray([
        [
            population[(pair[0] > 0) & (100 * (pair[1] + pair[2]) >= threshold * pair[0])].sum()
            for threshold in (5, 10, 20, 30)
        ]
        for pair in reference
    ])
    np.testing.assert_array_equal(summary.population_affected[:-1], expected_population)
    output = plot_cellwise_iso7243_classification(out.metadata.parent, population_weights_path=weights)
    assert output.pairwise_png.stat().st_size > 0


def test_population_affected_thresholds_are_inclusive_and_exclude_unsupported_cells():
    totals = np.zeros((2, 4), dtype=np.int64)
    _accumulate_population_affected(
        totals,
        np.asarray([0, 0, 0, 1, 1]),
        np.asarray([1, 1, 2, 3, 0]),
        np.asarray([20, 0, 10, 10, 10]),
        np.asarray([10, 1000, 20, 30, 0]),
    )
    np.testing.assert_array_equal(totals, [[30, 20, 20, 0], [30, 30, 30, 30]])
    assert np.all(np.diff(totals, axis=1) <= 0)


def test_no_supported_cells_remain_undefined(tmp_path):
    from iso7243_fixtures import NAMES
    from liljegren_analysis.cellwise.non_consensus_plots import plot_cellwise_iso7243_non_consensus

    run = tmp_path / "run"
    write_days(run, 1)
    path = run / "results/2025-01-01/0001" / NAMES[0] / "results.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.schema.get_field_index("wbgt_status"), "wbgt_status",
                             pa.array(["invalid"] * len(table)))
    pq.write_table(table, path)
    out = analyze_cellwise_iso7243_classification(
        run, start_date="2025-01-01", end_date="2025-01-01",
        thresholds_c=[25.], output_root=tmp_path / "out", date_workers=1)
    weights = tmp_path / "population.parquet"
    pq.write_table(pa.table({"source_cell_id": CELLS, "population": [10, 0, 1000]}), weights)
    summary = summarize_cellwise_iso7243_classification(out.metadata.parent, population_weights_path=weights)
    assert np.isnan(summary.changed_fraction).all()
    assert np.isnan(summary.pair_changed_fraction).all()
    assert np.equal(summary.histogram_shares, 0).all()
    assert plot_cellwise_iso7243_non_consensus(out.metadata.parent).png.stat().st_size > 0
    assert plot_cellwise_iso7243_classification(
        out.metadata.parent, population_weights_path=weights).png.stat().st_size > 0


@pytest.mark.parametrize("upper,label", [(37., "≥ 37"), (41., "≥ 41"), (100., "100")])
def test_colorbar_endpoint(upper, label):
    figure, axis = plt.subplots()
    image = axis.scatter([0], [0], c=[0], vmin=0, vmax=upper)
    bar = figure.colorbar(image)
    configure_percent_colorbar(bar, upper)
    assert bar.get_ticks()[-1] == upper
    assert bar.ax.get_yticklabels()[-1].get_text() == label
    if upper == 41:
        assert 40 not in bar.get_ticks()
    plt.close(figure)
