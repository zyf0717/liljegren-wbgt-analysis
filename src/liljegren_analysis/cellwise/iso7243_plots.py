"""Figures for persisted cellwise ISO 7243 classification differences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from ..comparison.area_weighted import _ERA5_LATITUDE_WEIGHTS
from ..io.artifacts import atomic_csv
from ..io.formatting import format_percentage as _format_percentage
from ..io.formatting import format_two_significant_figures as _format_two_significant_figures
from .paths import DEFAULT_ISO7243_OUTPUT_ROOT
from .iso7243 import load_hourly_metadata
from .population import DEFAULT_POPULATION_WEIGHTS_PATH, load_population_weights
from .projection import (
    add_map_colorbar,
    configure_equal_earth_axis,
    country_boundary_from_metadata,
    EQUAL_EARTH_X_LIMIT,
    EQUAL_EARTH_Y_LIMIT,
    PANEL_TITLE_FONT_SIZE_PT,
    plot_country_boundary,
    plot_equal_earth_cells,
    plot_major_cities,
    regional_map_figure_size,
    STACKED_MAP_DPI,
    STACKED_MAP_WIDTH_IN,
)


DEFAULT_INPUT_ROOT = DEFAULT_ISO7243_OUTPUT_ROOT
POPULATION_AFFECTED_THRESHOLDS_PERCENT = (5, 10, 20, 30)
_WEIGHTINGS = ("Unweighted", "Geometric-weighted", "Population-weighted")


@dataclass(frozen=True)
class ISO7243FigureOutputs:
    """Rendered figures and population-affected summary paths."""

    png: Path
    pairwise_png: Path
    population_affected_png: Path
    non_consensus_population_png: Path
    population_affected_csv: Path


@dataclass(frozen=True)
class ISO7243PlotSummary:
    """Compact aggregates used by the overview figure."""

    latitude: np.ndarray
    longitude: np.ndarray
    changed_fraction: np.ndarray
    non_consensus_fraction: np.ndarray
    population: np.ndarray
    histogram_shares: np.ndarray
    implementation_pairs: tuple[tuple[str, str], ...]
    pair_changed_fraction: np.ndarray
    population_affected: np.ndarray
    total_population: int
    region_label: str | None
    country_boundary: BaseGeometry | None
    country_iso_a3: str | None = None


def plot_cellwise_iso7243_classification(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    *,
    output_directory: str | Path | None = None,
    population_weights_path: str | Path = DEFAULT_POPULATION_WEIGHTS_PATH,
) -> ISO7243FigureOutputs:
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
    non_consensus_population_png = (
        target / "non_consensus_population_affected.png"
    )
    population_affected_csv = target / "population_affected.csv"
    _write_figure(summary, png)
    _write_pairwise_maps(summary, pairwise_png)
    _write_population_affected_figure(summary, population_affected_png)
    _write_non_consensus_population_figure(summary, non_consensus_population_png)
    atomic_csv(_population_affected_table(summary), population_affected_csv)
    return ISO7243FigureOutputs(
        png,
        pairwise_png,
        population_affected_png,
        non_consensus_population_png,
        population_affected_csv,
    )


def plot_global_iso7243_non_consensus_stack(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    *,
    output_directory: str | Path | None = None,
    population_weights_path: str | Path = DEFAULT_POPULATION_WEIGHTS_PATH,
) -> Path:
    """Render the global non-consensus maps on the shared stacked-map canvas."""

    root = Path(input_root).expanduser().resolve()
    target = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else root / "figures"
    )
    summary = summarize_cellwise_iso7243_classification(
        root, population_weights_path=population_weights_path
    )
    if summary.region_label is not None:
        raise ValueError("The standardized vertical stack requires global input")
    target.mkdir(parents=True, exist_ok=True)
    output = target / "non_consensus_stacked.png"
    _write_global_non_consensus_stack(summary, output)
    return output


def summarize_cellwise_iso7243_classification(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    *,
    population_weights_path: str | Path = DEFAULT_POPULATION_WEIGHTS_PATH,
) -> ISO7243PlotSummary:
    """Stream hourly counts into rates and weighted cell-pair distributions."""
    root = Path(input_root).expanduser().resolve()
    metadata = load_hourly_metadata(root)
    sample_scope = metadata["sample_scope"]
    implementation_pairs = _implementation_pairs(metadata)
    lookup = pq.read_table(root / "cell_lookup.parquet")
    cell_ids = _integer_column(lookup, "cell_id")
    if not np.array_equal(cell_ids, np.arange(len(cell_ids))) or len(cell_ids) != metadata["cell_count"]:
        raise ValueError("Cell lookup has invalid cell_id values")
    non_consensus = pq.read_table(root / "cellwise_non_consensus.parquet")
    if not np.array_equal(_integer_column(non_consensus, "cell_id"), cell_ids):
        raise ValueError("Non-consensus lookup differs")
    joint = _integer_column(non_consensus, "n_joint")
    non_unanimous = _integer_column(non_consensus, "n_non_consensus")
    if (joint < 0).any() or (non_unanimous < 0).any() or (non_unanimous > joint).any():
        raise ValueError("Invalid hourly support counts")
    if int(joint.sum()) != metadata["joint_observation_count"]:
        raise ValueError("Joint observation count differs from metadata")
    source_ids = lookup["source_cell_id"].combine_chunks().to_pylist()
    country_selection = metadata.get("country_selection")
    region_label = (
        str(country_selection.get("country_name"))
        if metadata.get("input_source") == "country-parquet"
        and isinstance(country_selection, dict)
        else None
    )
    raw_country_iso_a3 = (
        country_selection.get("country_iso_a3") if region_label is not None else None
    )
    population, total_population = _population_weights(
        source_ids, population_weights_path, regional=region_label is not None
    )
    weights = np.vstack((np.ones(len(cell_ids)), _geometry_weights(source_ids),
                         population.astype(np.float64)))
    population_affected = np.zeros(
        (len(implementation_pairs) + 1, len(POPULATION_AFFECTED_THRESHOLDS_PERCENT)),
        dtype=np.int64,
    )
    _accumulate_population_affected(
        population_affected[-1:],
        np.zeros(len(cell_ids), dtype=np.int64),
        non_unanimous,
        joint,
        population,
    )
    changed_sum = np.zeros(len(cell_ids), dtype=np.uint64)
    compared_sum = np.zeros(len(cell_ids), dtype=np.uint64)
    pair_changed_fraction = np.full(
        (len(implementation_pairs), len(cell_ids)), np.nan, dtype=np.float64,
    )
    seen = np.zeros(pair_changed_fraction.shape, dtype=bool)
    pair_index_by_code = _pair_index_by_code(metadata, implementation_pairs)
    implementation_count = len(metadata["implementation_ids"])
    histogram = np.zeros((3, 20), dtype=np.float64)
    totals = np.zeros(3, dtype=np.float64)
    product = pq.ParquetFile(root / "cell_pair_differences.parquet")
    for batch in product.iter_batches(
        columns=["cell_id", "n_compared", "n_higher", "n_lower",
                 "implementation_a_id", "implementation_b_id"],
        batch_size=250_000, use_threads=False,
    ):
        ids, compared, higher, lower, a, b = (
            np.asarray(batch.column(i), dtype=np.int64) for i in range(6)
        )
        if ((ids < 0).any() or (ids >= len(cell_ids)).any()
            or (a < 0).any() or (b < 0).any()
            or (a >= implementation_count).any() or (b >= implementation_count).any()):
            raise ValueError("Cell-pair IDs are out of range")
        pair_indices = pair_index_by_code[a * implementation_count + b]
        changed = higher + lower
        if ((pair_indices < 0).any() or (compared < joint[ids]).any()
            or (higher < 0).any() or (lower < 0).any() or (changed > compared).any()):
            raise ValueError("Cell-pair counts disagree with hourly support")
        if sample_scope == "all-implementations" and (
            (compared != joint[ids]).any() or (changed > non_unanimous[ids]).any()
        ):
            raise ValueError("Cell-pair counts disagree with shared hourly support")
        keys = pair_indices * len(cell_ids) + ids
        if len(np.unique(keys)) != len(keys) or seen[pair_indices, ids].any():
            raise ValueError("Duplicate cell-pair counts")
        seen[pair_indices, ids] = True
        np.add.at(changed_sum, ids, changed.astype(np.uint64))
        np.add.at(compared_sum, ids, compared.astype(np.uint64))
        supported = compared > 0
        fractions = np.divide(changed, compared, out=np.full(len(ids), np.nan),
                              where=supported)
        pair_changed_fraction[pair_indices, ids] = fractions
        _accumulate_population_affected(
            population_affected[:-1],
            pair_indices,
            changed,
            compared,
            population[ids],
        )
        for index in range(3):
            mass = weights[index, ids[supported]]
            histogram[index] += np.histogram(
                100.0 * fractions[supported], bins=np.arange(0.0, 105.0, 5.0),
                weights=mass,
            )[0]
            totals[index] += mass.sum()
    if (product.metadata.num_rows != metadata["row_count"]
        or product.metadata.num_rows != len(cell_ids) * len(implementation_pairs)
        or not seen.all()):
        raise ValueError("Cell-pair product is incomplete")
    if np.any(changed_sum < non_unanimous):
        raise ValueError("Non-consensus exceeds all pair disagreements")
    if int(compared_sum.sum()) != metadata["pair_observation_count"]:
        raise ValueError("Pair observation count differs from metadata")
    supported_pairs = np.isfinite(pair_changed_fraction).sum(axis=0)
    mean = np.divide(np.nansum(pair_changed_fraction, axis=0), supported_pairs,
                     out=np.full(len(cell_ids), np.nan), where=supported_pairs > 0)
    return ISO7243PlotSummary(
        latitude=_float_column(lookup, "latitude"),
        longitude=_float_column(lookup, "longitude"),
        changed_fraction=mean,
        non_consensus_fraction=np.divide(
            non_unanimous, joint, out=np.full(len(cell_ids), np.nan), where=joint > 0
        ),
        population=population,
        histogram_shares=np.divide(histogram, totals[:, None],
                                  out=np.zeros_like(histogram), where=totals[:, None] > 0),
        implementation_pairs=implementation_pairs,
        pair_changed_fraction=pair_changed_fraction,
        population_affected=population_affected,
        total_population=total_population,
        region_label=region_label,
        country_boundary=country_boundary_from_metadata(metadata),
        country_iso_a3=raw_country_iso_a3 if isinstance(raw_country_iso_a3, str) else None,
    )


def _implementation_pairs(metadata: dict[str, object]) -> tuple[tuple[str, str], ...]:
    ids = metadata["implementation_ids"]
    if not isinstance(ids, dict) or len(ids) < 2 or set(ids.values()) != set(range(len(ids))):
        raise ValueError("ISO 7243 implementation-ID mapping is invalid")
    implementations = tuple(name for name, _ in sorted(ids.items(), key=lambda item: item[1]))
    pairs = tuple(
        (implementations[implementation_a], implementations[implementation_b])
        for implementation_a in range(len(implementations))
        for implementation_b in range(implementation_a + 1, len(implementations))
    )
    if len(pairs) != int(metadata["implementation_pair_count"]):
        raise ValueError("ISO 7243 implementation pair count is invalid")
    return pairs


def _pair_index_by_code(
    metadata: dict[str, object], implementation_pairs: tuple[tuple[str, str], ...]
) -> np.ndarray:
    ids = metadata["implementation_ids"]
    assert isinstance(ids, dict)
    implementation_count = len(ids)
    result = np.full(implementation_count**2, -1, dtype=np.int32)
    for pair_index, (implementation_a, implementation_b) in enumerate(implementation_pairs):
        result[
            int(ids[implementation_a]) * implementation_count
            + int(ids[implementation_b])
        ] = pair_index
    return result


def _geometry_weights(source_ids: list[str]) -> np.ndarray:
    table = _ERA5_LATITUDE_WEIGHTS
    lookup = dict(
        zip(
            table["latitude_code"].to_pylist(),
            table["area_weight"].to_pylist(),
            strict=True,
        )
    )
    weights = np.asarray([lookup.get(source_id[12:19], np.nan) for source_id in source_ids])
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("Cell lookup has invalid ERA5 geometry weights")
    return weights


def _population_weights(
    source_ids: list[str], path: str | Path, *, regional: bool = False
) -> tuple[np.ndarray, int]:
    weights, global_population = load_population_weights(source_ids, path)
    if global_population <= 0:
        raise ValueError("Population weights have no positive total population")
    total_population = (
        int(weights.sum(dtype=np.int64)) if regional else global_population
    )
    if total_population <= 0:
        raise ValueError("Analysis grid has no positive population")
    return weights, total_population


def _accumulate_population_affected(
    totals: np.ndarray,
    group_indices: np.ndarray,
    changed: np.ndarray,
    compared: np.ndarray,
    population: np.ndarray,
) -> None:
    """Add each qualifying cell population once per group and threshold."""
    if not (
        len(group_indices) == len(changed) == len(compared) == len(population)
        and totals.shape[1] == len(POPULATION_AFFECTED_THRESHOLDS_PERCENT)
    ):
        raise ValueError("Population-affected inputs have incompatible shapes")
    supported = compared > 0
    for column, threshold in enumerate(POPULATION_AFFECTED_THRESHOLDS_PERCENT):
        affected = supported & (100 * changed >= threshold * compared)
        np.add.at(totals[:, column], group_indices[affected], population[affected])


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


def _write_figure(summary: ISO7243PlotSummary, png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), layout="constrained")
    map_axis = axes[0, 0]
    changed_percent = 100.0 * summary.changed_fraction
    image = plot_equal_earth_cells(
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
    configure_equal_earth_axis(
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
    figure.savefig(png, dpi=220)
    plt.close(figure)


def _write_pairwise_maps(summary: ISO7243PlotSummary, png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_upper_percent = _dynamic_color_upper_percent(
        100.0 * summary.pair_changed_fraction
    )
    column_count = min(3, len(summary.implementation_pairs))
    row_count = (
        len(summary.implementation_pairs) + column_count - 1
    ) // column_count
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
        image = plot_equal_earth_cells(
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
        configure_equal_earth_axis(
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
    figure.savefig(png, dpi=220)
    plt.close(figure)


def _write_population_affected_figure(
    summary: ISO7243PlotSummary, png: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
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
        100.0 * summary.population_affected.astype(np.float64) / summary.total_population
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
    figure.savefig(png, dpi=220)
    plt.close(figure)


def _write_non_consensus_population_figure(
    summary: ISO7243PlotSummary, png: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import FuncFormatter
    import seaborn as sns

    affected = summary.non_consensus_fraction >= 0.20
    grey = summary.non_consensus_fraction < 0.20
    population = summary.population[affected]
    population_cap = float(np.quantile(population, 0.95)) if population.size else 0.0
    figure_size = (
        regional_map_figure_size(summary.longitude, summary.latitude)
        if summary.region_label is not None
        else (10.0, 4.8)
    )
    figure, axis = plt.subplots(figsize=figure_size, layout="constrained")
    plot_equal_earth_cells(
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
    image = plot_equal_earth_cells(
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
    configure_equal_earth_axis(
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
    figure.savefig(png, dpi=220)
    plt.close(figure)


def _write_global_non_consensus_stack(
    summary: ISO7243PlotSummary, png: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import FuncFormatter
    import seaborn as sns

    percent = 100.0 * summary.non_consensus_fraction
    valid = np.isfinite(percent)
    upper = _dynamic_color_upper_percent(percent)
    affected = summary.non_consensus_fraction >= 0.20
    grey = summary.non_consensus_fraction < 0.20
    population = summary.population[affected]
    population_cap = (
        float(np.quantile(population, 0.95)) if population.size else 0.0
    )
    affected_population = int(population.sum(dtype=np.int64))

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
        bottom_margin
        + 2.0 * map_height
        + sum(title_heights)
        + panel_gap
        + 0.16
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
            image = plot_equal_earth_cells(
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
            plot_equal_earth_cells(
                axis,
                summary.longitude[grey],
                summary.latitude[grey],
                np.zeros(np.count_nonzero(grey)),
                regional=False,
                point_size=0.35,
                cmap=ListedColormap(["#bdbdbd"]),
                vmin=0.0,
                vmax=1.0,
            )
            image = plot_equal_earth_cells(
                axis,
                summary.longitude[affected],
                summary.latitude[affected],
                population,
                regional=False,
                point_size=0.35,
                cmap=sns.color_palette("rocket_r", as_cmap=True),
                vmin=0.0,
                vmax=population_cap or 1.0,
            )
            colorbar_label = "Population (thousands) of grid cell"
        configure_equal_earth_axis(axis)
        colorbar = figure.colorbar(
            image,
            cax=colorbar_axis,
            label=colorbar_label,
            extend="max",
            extendfrac=0.05,
        )
        colorbar.ax.yaxis.labelpad = 8
        if index == 0:
            configure_percent_colorbar(colorbar, upper)
        else:
            colorbar.formatter = FuncFormatter(
                lambda value, _: f"{value / 1_000:g}"
            )
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

    figure.savefig(png, dpi=STACKED_MAP_DPI, facecolor="white")
    plt.close(figure)


def _dynamic_color_upper_percent(values: np.ndarray) -> float:
    """Choose a shared robust limit without letting rare extremes flatten every map."""

    finite = values[np.isfinite(values)]
    if not len(finite):
        return 1.0
    return max(1.0, float(np.ceil(np.quantile(finite, 0.99))))


def configure_percent_colorbar(colorbar, upper: float) -> None:
    """Include the numeric endpoint without crowding adjacent percentage ticks."""
    ticks = colorbar.get_ticks()
    ticks = ticks[(ticks >= 0) & (ticks < upper)]
    if len(ticks) > 1:
        spacing = float(np.median(np.diff(ticks)))
        ticks = ticks[upper - ticks >= spacing / 2]
    ticks = np.unique(np.append(ticks, upper))
    colorbar.set_ticks(ticks, labels=[f"{tick:g}" for tick in ticks])


def _integer_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64
    )


def _float_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    )
