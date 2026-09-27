"""Figures for persisted cellwise ISO 7243 classification differences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from ..io.artifacts import atomic_csv
from ..io.formatting import format_percentage as _format_percentage
from ..io.formatting import (
    format_two_significant_figures as _format_two_significant_figures,
)
from .iso7243_summary import (
    POPULATION_AFFECTED_THRESHOLDS_PERCENT,
    ISO7243PlotSummary,
    NonConsensusSummary,
    summarize_cellwise_iso7243_classification,
    summarize_non_consensus,
)
from .projection import (
    EQUAL_EARTH_X_LIMIT,
    EQUAL_EARTH_Y_LIMIT,
    PANEL_TITLE_FONT_SIZE_PT,
    STACKED_MAP_WIDTH_IN,
    add_map_colorbar,
    configure_map_axis,
    configure_percent_colorbar,
    percent_color_upper,
    plot_country_boundary,
    plot_major_cities,
    plot_map_cells,
    regional_map_figure_size,
    save_figure,
)

_WEIGHTINGS = ("Unweighted", "Geometric-weighted", "Population-weighted")


def plot_cellwise_iso7243_classification(
    input_root: str | Path,
    *,
    output_directory: str | Path | None = None,
    population_weights_path: str | Path,
) -> Path:
    """Plot cell disagreement frequency and three weighted count distributions."""

    root = Path(input_root).expanduser().resolve()
    target = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else root / "figures"
    )
    summary = summarize_cellwise_iso7243_classification(
        root, population_weights_path=population_weights_path
    )
    target.mkdir(parents=True, exist_ok=True)
    png = target / "classification_changes.png"
    pairwise_png = target / "pairwise_classification_changes.png"
    population_affected_png = target / "population_affected.png"
    non_consensus_population_png = target / "non_consensus_population_affected.png"
    population_affected_csv = target / "population_affected.csv"
    save_figure(classification_figure(summary), png)
    save_figure(pairwise_maps_figure(summary), pairwise_png)
    save_figure(population_affected_figure(summary), population_affected_png)
    save_figure(non_consensus_population_figure(summary), non_consensus_population_png)
    atomic_csv(_population_affected_table(summary), population_affected_csv)
    return target


def plot_global_iso7243_non_consensus_stack(
    input_root: str | Path,
    *,
    output_directory: str | Path | None = None,
    population_weights_path: str | Path,
) -> Path:
    """Render the global non-consensus maps on the shared stacked-map canvas."""

    root = Path(input_root).expanduser().resolve()
    target = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else root / "figures"
    )
    summary = summarize_non_consensus(
        root, population_weights_path=population_weights_path
    )
    if summary.region_label is not None:
        raise ValueError("The standardized vertical stack requires global input")
    target.mkdir(parents=True, exist_ok=True)
    output = target / "non_consensus_stacked.png"
    save_figure(global_non_consensus_figure(summary), output)
    return output


def _population_affected_table(summary: ISO7243PlotSummary) -> pa.Table:
    def row(
        comparison_type: str,
        implementation_a: str | None,
        implementation_b: str | None,
        populations: np.ndarray,
    ) -> dict[str, object]:
        values: dict[str, object] = {
            "comparison_type": comparison_type,
            "implementation_a": implementation_a,
            "implementation_b": implementation_b,
        }
        for threshold, population in zip(
            POPULATION_AFFECTED_THRESHOLDS_PERCENT, populations, strict=True
        ):
            if population == 0:
                values[f"{threshold}%"] = "-"
                continue
            population_percent = 100.0 * int(population) / summary.total_population
            values[f"{threshold}%"] = (
                f"{_format_population(int(population), regional=summary.region_label is not None)} "
                f"({_format_percentage(population_percent)})"
            )
        return values

    rows: list[dict[str, object]] = []
    for pair_index, (implementation_a, implementation_b) in enumerate(
        summary.implementation_pairs
    ):
        rows.append(
            row(
                "pairwise_disagreement",
                implementation_a,
                implementation_b,
                summary.population_affected[pair_index],
            )
        )
    rows.append(row("non_consensus", None, None, summary.population_affected[-1]))
    fields = [
        ("comparison_type", pa.string()),
        ("implementation_a", pa.string()),
        ("implementation_b", pa.string()),
    ]
    for threshold in POPULATION_AFFECTED_THRESHOLDS_PERCENT:
        fields.append((f"{threshold}%", pa.string()))
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(fields),
    )


def _format_population(population: int, *, regional: bool = False) -> str:
    divisor = 1e6 if regional else 1e9
    return _format_two_significant_figures(population / divisor)


def classification_figure(summary: ISO7243PlotSummary):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), layout="constrained")
    map_axis = axes[0, 0]
    changed_percent = 100.0 * summary.changed_fraction
    image = plot_map_cells(
        map_axis,
        summary.longitude,
        summary.latitude,
        changed_percent,
        regional=summary.region_label is not None,
        point_size=0.35,
        cmap="cividis",
        vmin=0.0,
        vmax=50.0,
    )
    configure_map_axis(
        map_axis,
        summary.longitude if summary.region_label else None,
        summary.latitude if summary.region_label else None,
    )
    plot_country_boundary(map_axis, summary.country_boundary)
    colorbar = add_map_colorbar(
        figure,
        image,
        ax=map_axis,
        label="Mean pairwise WBGT class disagreement\n(% valid hours)",
    )
    configure_percent_colorbar(colorbar, 50.0)
    x = np.arange(2.5, 100.0, 5.0)
    for axis, label, shares in zip(
        axes.flat[1:], _WEIGHTINGS, summary.histogram_shares, strict=True
    ):
        axis.bar(x, 100.0 * shares, width=5.0, color="#386cb0")
        axis.set(
            xlabel="WBGT class disagreement (% valid hours)",
            ylabel="Cell-pair share (%)",
            title=label,
            xticks=np.arange(0, 101, 25),
            xlim=(0.0, 100.0),
            ylim=(0.0, 100.0),
        )
    return figure


def pairwise_maps_figure(summary: ISO7243PlotSummary):
    import matplotlib.pyplot as plt

    color_upper_percent = percent_color_upper(100.0 * summary.pair_changed_fraction)
    column_count = min(3, len(summary.implementation_pairs))
    row_count = (len(summary.implementation_pairs) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.2 * column_count, 2.5 * row_count),
        layout="constrained",
        squeeze=False,
    )
    image = None
    for axis, (implementation_a, implementation_b), values in zip(
        axes.flat,
        summary.implementation_pairs,
        summary.pair_changed_fraction,
    ):
        image = plot_map_cells(
            axis,
            summary.longitude,
            summary.latitude,
            100.0 * values,
            regional=summary.region_label is not None,
            point_size=0.25,
            cmap="cividis",
            vmin=0.0,
            vmax=color_upper_percent,
        )
        configure_map_axis(
            axis,
            summary.longitude if summary.region_label else None,
            summary.latitude if summary.region_label else None,
        )
        plot_country_boundary(axis, summary.country_boundary)
        axis.set_title(f"{implementation_a} vs {implementation_b}")
    for axis in tuple(axes.flat)[len(summary.implementation_pairs) :]:
        axis.set_visible(False)
    assert image is not None
    colorbar = add_map_colorbar(
        figure,
        image,
        ax=axes.ravel().tolist(),
        label="WBGT class disagreement (% valid hours)",
        capped=color_upper_percent < 100.0,
        shrink=0.86,
    )
    configure_percent_colorbar(colorbar, color_upper_percent)
    return figure


def population_affected_figure(summary: ISO7243PlotSummary):
    import matplotlib.pyplot as plt

    labels = [
        f"{implementation_a} vs {implementation_b}"
        for implementation_a, implementation_b in summary.implementation_pairs
    ] + ["All-target non-consensus"]
    positions = np.arange(len(labels), dtype=np.float64)
    height = 0.18
    offsets = height * (
        np.arange(len(POPULATION_AFFECTED_THRESHOLDS_PERCENT))
        - (len(POPULATION_AFFECTED_THRESHOLDS_PERCENT) - 1) / 2
    )
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    figure, axis = plt.subplots(
        figsize=(12.0, max(6.0, 0.58 * len(labels))), layout="constrained"
    )
    population_percent = (
        100.0
        * summary.population_affected.astype(np.float64)
        / summary.total_population
    )
    for column, (threshold, offset, color) in enumerate(
        zip(POPULATION_AFFECTED_THRESHOLDS_PERCENT, offsets, colors, strict=True)
    ):
        axis.barh(
            positions + offset,
            population_percent[:, column],
            height=height,
            color=color,
            label=f"≥ {threshold}%",
        )
    axis.set(
        yticks=positions,
        yticklabels=labels,
        xlabel=(
            f"Affected population (% of {summary.region_label} input-region population)"
            if summary.region_label
            else "Affected population (% of total population)"
        ),
        ylabel="Comparison",
    )
    axis.invert_yaxis()
    axis.grid(axis="x", color="#d9d9d9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        title="Annual disagreement / non-consensus rate",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=4,
    )
    return figure


def _population_map(axis, summary: NonConsensusSummary):
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    affected = summary.non_consensus_fraction >= 0.20
    grey = summary.non_consensus_fraction < 0.20
    population = summary.population[affected]
    population_cap = float(np.quantile(population, 0.95)) if population.size else 0.0
    plot_map_cells(
        axis,
        summary.longitude[grey],
        summary.latitude[grey],
        np.zeros(np.count_nonzero(grey)),
        regional=summary.region_label is not None,
        point_size=0.35,
        cmap=ListedColormap(["#bdbdbd"]),
        vmin=0.0,
        vmax=1.0,
    )
    image = plot_map_cells(
        axis,
        summary.longitude[affected],
        summary.latitude[affected],
        population,
        regional=summary.region_label is not None,
        point_size=0.35,
        cmap=sns.color_palette("rocket_r", as_cmap=True),
        vmin=0.0,
        vmax=population_cap or 1.0,
    )
    return image


def non_consensus_population_figure(summary: ISO7243PlotSummary):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figure_size = (
        regional_map_figure_size(summary.longitude, summary.latitude)
        if summary.region_label is not None
        else (10.0, 4.8)
    )
    figure, axis = plt.subplots(figsize=figure_size, layout="constrained")
    image = _population_map(axis, summary)
    configure_map_axis(
        axis,
        summary.longitude if summary.region_label else None,
        summary.latitude if summary.region_label else None,
    )
    plot_country_boundary(axis, summary.country_boundary)
    plot_major_cities(axis, summary.country_iso_a3)
    colorbar = add_map_colorbar(
        figure, image, ax=axis, label="Population (thousands) of grid cell"
    )
    colorbar.formatter = FuncFormatter(lambda value, _: f"{value / 1_000:g}")
    colorbar.update_ticks()
    colorbar.ax.yaxis.labelpad = 10
    return figure


def global_non_consensus_figure(summary: ISO7243PlotSummary):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    percent = 100.0 * summary.non_consensus_fraction
    valid = np.isfinite(percent)
    upper = percent_color_upper(percent)
    affected_population = int(
        summary.population[summary.non_consensus_fraction >= 0.20].sum()
    )

    figure_width = STACKED_MAP_WIDTH_IN
    map_left = 0.08
    map_width = figure_width - 1.08
    map_height = map_width * EQUAL_EARTH_Y_LIMIT / EQUAL_EARTH_X_LIMIT
    colorbar_left = map_left + map_width + 0.12
    colorbar_width = 0.13
    bottom_margin = 0.16
    title_heights = (0.38, 0.58)
    panel_gap = 0.24
    figure_height = (
        bottom_margin + 2.0 * map_height + sum(title_heights) + panel_gap + 0.16
    )
    panel_bottoms = (
        bottom_margin + map_height + title_heights[1] + panel_gap,
        bottom_margin,
    )
    titles = (
        "A) Non-consensus in ISO 7243 WBGT classifications",
        (
            f"B) {_format_population(affected_population)} billion people lived in "
            "grid cells with ISO 7243 WBGT\n"
            "classification non-consensus in ≥20% of valid hours"
        ),
    )

    figure = plt.figure(figsize=(figure_width, figure_height))
    for index, (bottom, title_height, title) in enumerate(
        zip(panel_bottoms, title_heights, titles, strict=True)
    ):
        axis = figure.add_axes(
            [
                map_left / figure_width,
                bottom / figure_height,
                map_width / figure_width,
                map_height / figure_height,
            ]
        )
        colorbar_axis = figure.add_axes(
            [
                colorbar_left / figure_width,
                bottom / figure_height,
                colorbar_width / figure_width,
                map_height / figure_height,
            ]
        )
        if index == 0:
            image = plot_map_cells(
                axis,
                summary.longitude[valid],
                summary.latitude[valid],
                percent[valid],
                regional=False,
                point_size=0.35,
                cmap="cividis",
                vmin=0.0,
                vmax=upper,
            )
            colorbar_label = "WBGT classification non-consensus (% valid hours)"
        else:
            image = _population_map(axis, summary)
            colorbar_label = "Population (thousands) of grid cell"
        configure_map_axis(axis)
        colorbar = figure.colorbar(
            image,
            cax=colorbar_axis,
            label=colorbar_label,
            extend="max" if index or upper < 100 else "neither",
            extendfrac=0.05,
        )
        colorbar.ax.yaxis.labelpad = 8
        if index == 0:
            configure_percent_colorbar(colorbar, upper)
        else:
            colorbar.formatter = FuncFormatter(lambda value, _: f"{value / 1_000:g}")
            colorbar.update_ticks()
        figure.text(
            (map_left + map_width / 2.0) / figure_width,
            (bottom + map_height + title_height - 0.05) / figure_height,
            title,
            ha="center",
            va="top",
            fontsize=PANEL_TITLE_FONT_SIZE_PT,
            linespacing=1.2,
        )

    return figure


def plot_cellwise_iso7243_non_consensus(
    input_root: Path, *, output_directory: Path | None = None
) -> Path:
    root = Path(input_root)
    path = (
        Path(output_directory) / "non_consensus_percent.png"
        if output_directory is not None
        else root / "figures/non_consensus_percent.png"
    )
    save_figure(non_consensus_figure(summarize_non_consensus(root)), path)
    return path


def non_consensus_figure(summary: NonConsensusSummary):
    import matplotlib.pyplot as plt

    percent = 100.0 * summary.non_consensus_fraction
    upper = percent_color_upper(percent)
    regional = summary.region_label is not None
    size = (
        regional_map_figure_size(summary.longitude, summary.latitude)
        if regional
        else (10.0, 4.8)
    )
    figure, axis = plt.subplots(figsize=size, layout="constrained")
    valid = np.isfinite(percent)
    image = plot_map_cells(
        axis,
        summary.longitude[valid],
        summary.latitude[valid],
        percent[valid],
        regional=regional,
        point_size=0.35,
        cmap="cividis",
        vmin=0.0,
        vmax=upper,
    )
    configure_map_axis(
        axis,
        summary.longitude if regional else None,
        summary.latitude if regional else None,
    )
    plot_country_boundary(axis, summary.country_boundary)
    plot_major_cities(axis, summary.country_iso_a3)
    colorbar = add_map_colorbar(
        figure,
        image,
        ax=axis,
        label="WBGT classification non-consensus (% valid hours)",
        capped=upper < 100,
    )
    configure_percent_colorbar(colorbar, upper)
    return figure
