import json

import matplotlib
from sample_settings import DEFAULT_IMPLEMENTATIONS

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import pytest
from iso7243_fixtures import (
    CELLS,
    NAMES,
    PAIR_COUNT,
    reference_pairwise_counts,
    write_days,
)

from analysis.cellwise.iso7243 import analyze_cellwise_iso7243_classification
from analysis.cellwise.iso7243_plots import (
    _format_percentage,
    _format_population,
    _format_two_significant_figures,
    non_consensus_population_figure,
    plot_cellwise_iso7243_classification,
    plot_global_iso7243_non_consensus_stack,
)
from analysis.cellwise.iso7243_summary import (
    ISO7243PlotSummary,
    _accumulate_population_affected,
    summarize_cellwise_iso7243_classification,
)
from analysis.cellwise.projection import (
    STACKED_MAP_WIDTH_PX,
    configure_percent_colorbar,
)


def test_hourly_rates_weighting_and_endpoint_bins(tmp_path):
    run = tmp_path / "run"
    write_days(run, 12, complex_rows=False)
    out = analyze_cellwise_iso7243_classification(
        run,
        start_date="2025-01-01",
        end_date="2025-01-12",
        thresholds_c=[25.0, 30.0],
        output_root=tmp_path / "out",
        date_workers=1,
        implementations=DEFAULT_IMPLEMENTATIONS,
    )
    weights = tmp_path / "population.parquet"
    pq.write_table(
        pa.table({"source_cell_id": CELLS, "population": [10, 0, 1000]}), weights
    )
    summary = summarize_cellwise_iso7243_classification(
        out, population_weights_path=weights
    )
    np.testing.assert_allclose(
        summary.changed_fraction, [2 / len(NAMES), 0, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        summary.non_consensus_fraction, [1, 0, np.nan], equal_nan=True
    )
    np.testing.assert_array_equal(summary.population, [10, 0, 1000])
    assert summary.histogram_shares.shape == (3, 20)
    np.testing.assert_allclose(summary.histogram_shares.sum(axis=1), 1.0)
    # Pairs containing Argonne disagree for the first cell; third cell is unsupported.
    assert summary.histogram_shares[0, -1] == pytest.approx(
        (len(NAMES) - 1) / (2 * PAIR_COUNT)
    )
    assert summary.histogram_shares[2, -1] == pytest.approx(
        (len(NAMES) - 1) / PAIR_COUNT
    )
    np.testing.assert_allclose(
        summary.pair_changed_fraction[0], [1, 0, np.nan], equal_nan=True
    )
    np.testing.assert_array_equal(summary.population_affected[: len(NAMES) - 1], 10)
    np.testing.assert_array_equal(summary.population_affected[len(NAMES) - 1 : -1], 0)
    np.testing.assert_array_equal(summary.population_affected[-1], 10)
    assert summary.total_population == 1010
    figures = plot_cellwise_iso7243_classification(out, population_weights_path=weights)
    assert (figures / "classification_changes.png").name == "classification_changes.png"
    assert (
        figures / "pairwise_classification_changes.png"
    ).name == "pairwise_classification_changes.png"
    assert (figures / "population_affected.png").name == "population_affected.png"
    assert (
        figures / "non_consensus_population_affected.png"
    ).name == "non_consensus_population_affected.png"
    assert (figures / "population_affected.csv").name == "population_affected.csv"
    assert (figures / "classification_changes.png").stat().st_size > 0
    assert (figures / "pairwise_classification_changes.png").stat().st_size > 0
    assert (figures / "population_affected.png").stat().st_size > 0
    assert (figures / "non_consensus_population_affected.png").stat().st_size > 0
    stacked = plot_global_iso7243_non_consensus_stack(
        out, population_weights_path=weights
    )
    assert plt.imread(stacked).shape[1] == STACKED_MAP_WIDTH_PX
    population_csv = pcsv.read_csv(figures / "population_affected.csv")
    assert population_csv.column_names == [
        "comparison_type",
        "implementation_a",
        "implementation_b",
        "5%",
        "10%",
        "20%",
        "30%",
    ]
    assert population_csv.num_rows == PAIR_COUNT + 1
    for threshold in (5, 10, 20, 30):
        assert population_csv[f"{threshold}%"][0].as_py() == "0.000000010 (0.99%)"
        assert population_csv[f"{threshold}%"][len(NAMES) - 1].as_py() == "-"


@pytest.mark.parametrize(
    ("population", "expected"),
    [
        (6_459_547_367, "6.5"),
        (96_742_045, "0.097"),
        (10_000_000, "0.010"),
        (9_999_999, "0.010"),
        (442_136, "0.00044"),
        (10, "0.000000010"),
        (0, "0"),
    ],
)
def test_population_display_uses_billions_without_suffix_and_exactly_two_sf(
    population, expected
):
    assert _format_population(population) == expected


@pytest.mark.parametrize(
    ("population", "expected"),
    [(96_742_045, "97"), (10_000_000, "10"), (442_136, "0.44"), (10, "0.000010")],
)
def test_country_population_display_uses_millions_without_suffix(population, expected):
    assert _format_population(population, regional=True) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (83.051, "83%"),
        (1.20065, "1.2%"),
        (0.0855, "0.086%"),
        (0.01, "0.010%"),
        (0.009, "0.0090%"),
        (0.0, "0%"),
    ],
)
def test_percentage_display_uses_exactly_two_significant_figures(value, expected):
    assert _format_percentage(value) == expected


def test_two_significant_figure_formatter_preserves_trailing_zeroes():
    assert _format_two_significant_figures(6.0) == "6.0"


def test_incompatible_product_rejected(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"quantiles": ["p50", "p90", "p95", "p99"]})
    )
    with pytest.raises(ValueError, match="hourly"):
        summarize_cellwise_iso7243_classification(
            tmp_path, population_weights_path=tmp_path / "absent"
        )


def test_pairwise_plot_uses_each_pairs_support(tmp_path):
    run = tmp_path / "run"
    write_days(run, 1)
    out = analyze_cellwise_iso7243_classification(
        run,
        start_date="2025-01-01",
        end_date="2025-01-01",
        thresholds_c=[25.0, 30.0],
        sample_scope="pairwise",
        output_root=tmp_path / "out",
        date_workers=1,
        implementations=DEFAULT_IMPLEMENTATIONS,
    )
    weights = tmp_path / "population.parquet"
    pq.write_table(
        pa.table({"source_cell_id": CELLS, "population": [10, 0, 1000]}), weights
    )
    summary = summarize_cellwise_iso7243_classification(
        out, population_weights_path=weights
    )
    reference = reference_pairwise_counts(run, ["2025-01-01"])
    expected = np.divide(
        reference[:, 1] + reference[:, 2],
        reference[:, 0],
        out=np.full((PAIR_COUNT, 3), np.nan),
        where=reference[:, 0] > 0,
    )
    np.testing.assert_allclose(summary.pair_changed_fraction, expected, equal_nan=True)
    np.testing.assert_allclose(summary.changed_fraction, np.nanmean(expected, axis=0))
    assert summary.pair_changed_fraction[0, 0] == pytest.approx(3 / 24)
    # Cell 2 has no all-target support, but pairs excluding Argonne remain valid.
    assert np.isfinite(summary.pair_changed_fraction[:, 2]).sum() == (
        (len(NAMES) - 1) * (len(NAMES) - 2) // 2
    )
    assert summary.changed_fraction[2] == 0
    np.testing.assert_allclose(summary.histogram_shares.sum(axis=1), 1.0)
    population = np.asarray([10, 0, 1000])
    expected_population = np.asarray(
        [
            [
                population[
                    (pair[0] > 0) & (100 * (pair[1] + pair[2]) >= threshold * pair[0])
                ].sum()
                for threshold in (5, 10, 20, 30)
            ]
            for pair in reference
        ]
    )
    np.testing.assert_array_equal(summary.population_affected[:-1], expected_population)
    output = plot_cellwise_iso7243_classification(out, population_weights_path=weights)
    assert (output / "pairwise_classification_changes.png").stat().st_size > 0


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


@pytest.mark.parametrize("region_label", [None, "Test region"])
def test_non_consensus_population_map_filters_cells(
    tmp_path, monkeypatch, region_label
):
    summary = ISO7243PlotSummary(
        latitude=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        longitude=np.asarray([10.0, 11.0, 12.0, 13.0, 14.0]),
        changed_fraction=np.zeros(5),
        non_consensus_fraction=np.asarray([0.20, 0.199, np.nan, 1.0, 0.3]),
        population=np.asarray([20, 999, 1000, 80, 0]),
        histogram_shares=np.zeros((3, 20)),
        implementation_pairs=(),
        pair_changed_fraction=np.empty((0, 5)),
        population_affected=np.asarray([[100, 100, 100, 80]]),
        total_population=2099,
        region_label=region_label,
        country_boundary=None,
    )
    monkeypatch.setattr(plt, "close", lambda figure: None)
    try:
        backend = matplotlib.get_backend()
        figure = non_consensus_population_figure(summary)
        assert matplotlib.get_backend() == backend
        assert plt.fignum_exists(figure.number)
        figure.canvas.draw()
        background, cells = figure.axes[0].collections
        np.testing.assert_array_equal(background.get_array(), [0])
        np.testing.assert_allclose(
            background.get_facecolors(), [[189 / 255, 189 / 255, 189 / 255, 1]]
        )
        # Exactly 20% qualifies; below-threshold and unsupported populations do not.
        # Zero-population affected cells use the low end of the gradient.
        np.testing.assert_array_equal(cells.get_array(), [20, 80, 0])
        assert cells.norm(0) == 0
        assert cells.get_cmap().name == "rocket_r"
        assert cells.get_clim() == pytest.approx((0.0, 74.0))
        assert cells.norm(37) == pytest.approx(0.5)
        np.testing.assert_array_equal(cells.to_rgba(80), cells.to_rgba(74))
        assert cells.colorbar.extend == "max"
        assert figure.axes[0].get_legend() is None
        assert figure.axes[1].get_ylabel() == "Population (thousands) of grid cell"
    finally:
        monkeypatch.undo()
        plt.close("all")


def test_no_supported_cells_remain_undefined(tmp_path):
    from iso7243_fixtures import NAMES

    from analysis.cellwise.iso7243_plots import (
        plot_cellwise_iso7243_non_consensus,
    )

    run = tmp_path / "run"
    write_days(run, 1)
    path = run / "results/2025-01-01/0001" / NAMES[0] / "results.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("wbgt_status"),
        "wbgt_status",
        pa.array(["invalid"] * len(table)),
    )
    pq.write_table(table, path)
    out = analyze_cellwise_iso7243_classification(
        run,
        start_date="2025-01-01",
        end_date="2025-01-01",
        thresholds_c=[25.0],
        output_root=tmp_path / "out",
        date_workers=1,
        implementations=DEFAULT_IMPLEMENTATIONS,
    )
    weights = tmp_path / "population.parquet"
    pq.write_table(
        pa.table({"source_cell_id": CELLS, "population": [10, 0, 1000]}), weights
    )
    summary = summarize_cellwise_iso7243_classification(
        out, population_weights_path=weights
    )
    assert np.isnan(summary.changed_fraction).all()
    assert np.isnan(summary.non_consensus_fraction).all()
    assert np.isnan(summary.pair_changed_fraction).all()
    assert np.equal(summary.histogram_shares, 0).all()
    non_consensus_figure = plot_cellwise_iso7243_non_consensus(out)
    assert non_consensus_figure.name == "non_consensus_percent.png"
    assert non_consensus_figure.stat().st_size > 0
    figures = plot_cellwise_iso7243_classification(out, population_weights_path=weights)
    assert (figures / "classification_changes.png").stat().st_size > 0


@pytest.mark.parametrize("upper,label", [(37.0, "37"), (41.0, "41"), (100.0, "100")])
def test_colorbar_endpoint(upper, label):
    from analysis.cellwise.projection import add_map_colorbar

    figure, axis = plt.subplots()
    image = axis.scatter([0], [0], c=[0], vmin=0, vmax=upper)
    bar = add_map_colorbar(figure, image, ax=axis, label="Percent", capped=upper < 100)
    configure_percent_colorbar(bar, upper)
    assert bar.get_ticks()[-1] == upper
    assert bar.ax.get_yticklabels()[-1].get_text() == label
    assert bar.extend == ("max" if upper < 100 else "neither")
    assert bar.extendfrac == 0.05
    if upper == 41:
        assert 40 not in bar.get_ticks()
    plt.close(figure)


def test_non_consensus_summary_does_not_read_pair_tables(tmp_path, monkeypatch):
    from analysis.cellwise.iso7243_summary import summarize_non_consensus

    run = tmp_path / "run"
    write_days(run, 1, complex_rows=False)
    output = analyze_cellwise_iso7243_classification(
        run,
        start_date="2025-01-01",
        end_date="2025-01-01",
        thresholds_c=[25, 30],
        output_root=tmp_path / "out",
        date_workers=1,
        implementations=DEFAULT_IMPLEMENTATIONS,
    )
    # A valid completion record still names this file, but map loading need not open it.
    (output / "cell_pair_differences.parquet").write_bytes(b"not needed by this figure")
    summary = summarize_non_consensus(output)
    np.testing.assert_allclose(summary.non_consensus_fraction, [1, 0, np.nan])
    assert summary.population is None
