"""Figures for persisted cellwise ISO 7243 classification differences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..comparison.area_weighted import _ERA5_LATITUDE_WEIGHTS
from ..io.artifacts import atomic_csv
from .paths import DEFAULT_ISO7243_OUTPUT_ROOT
from .iso7243 import load_hourly_metadata
from .population import DEFAULT_OUTPUT_ROOT as DEFAULT_POPULATION_OUTPUT_ROOT
from .projection import configure_equal_earth_axis, equal_earth_coordinates


DEFAULT_INPUT_ROOT = DEFAULT_ISO7243_OUTPUT_ROOT
DEFAULT_POPULATION_WEIGHTS_PATH = DEFAULT_POPULATION_OUTPUT_ROOT / "population.parquet"
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
    histogram_shares: np.ndarray
    implementation_pairs: tuple[tuple[str, str], ...]
    pair_changed_fraction: np.ndarray
    population_affected: np.ndarray
    total_population: int


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
    png = target / "iso7243_classification_changes.png"
    pairwise_png = target / "iso7243_pairwise_classification_changes.png"
    population_affected_png = target / "iso7243_population_affected.png"
    non_consensus_population_png = (
        target / "iso7243_non_consensus_population_affected.png"
    )
    population_affected_csv = target / "iso7243_population_affected.csv"
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


def summarize_cellwise_iso7243_classification(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
    *,
    population_weights_path: str | Path = DEFAULT_POPULATION_WEIGHTS_PATH,
) -> ISO7243PlotSummary:
    """Stream hourly counts into rates and weighted cell-pair distributions."""
    root = Path(input_root).expanduser().resolve()
    metadata = load_hourly_metadata(root)
    sample_scope = metadata.get("sample_scope", "all-implementations")
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
    population, total_population = _population_weights(
        source_ids, population_weights_path
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
        histogram_shares=np.divide(histogram, totals[:, None],
                                  out=np.zeros_like(histogram), where=totals[:, None] > 0),
        implementation_pairs=implementation_pairs,
        pair_changed_fraction=pair_changed_fraction,
        population_affected=population_affected,
        total_population=total_population,
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
    source_ids: list[str], path: str | Path
) -> tuple[np.ndarray, int]:
    population_path = Path(path).expanduser().resolve()
    if not population_path.is_file():
        raise FileNotFoundError(
            f"Population weights do not exist: {population_path}; run "
            "scripts/compute_era5_population_weights.py first"
        )
    table = pq.read_table(population_path, columns=["source_cell_id", "population"])
    population_ids = table["source_cell_id"].combine_chunks().to_pylist()
    population = _integer_column(table, "population")
    if len(population_ids) != len(set(population_ids)) or (population < 0).any():
        raise ValueError("Population weights contain invalid source_cell_id or population values")
    total_population = int(population.sum(dtype=np.int64))
    if total_population <= 0:
        raise ValueError("Population weights have no positive total population")
    lookup = dict(zip(population_ids, population, strict=True))
    weights = np.asarray([lookup.get(source_id, -1) for source_id in source_ids], dtype=np.int64)
    if (weights < 0).any():
        raise ValueError("Population weights are missing a cell from the ISO 7243 lookup")
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
    rows = []
    for pair_index, (implementation_a, implementation_b) in enumerate(
        summary.implementation_pairs
    ):
        for threshold, population in zip(
            POPULATION_AFFECTED_THRESHOLDS_PERCENT,
            summary.population_affected[pair_index],
            strict=True,
        ):
            rows.append(
                {
                    "comparison_type": "pairwise_disagreement",
                    "implementation_a": implementation_a,
                    "implementation_b": implementation_b,
                    "minimum_rate_percent": threshold,
                    "estimated_population_affected": int(population),
                    "percent_total_population": (
                        100.0 * int(population) / summary.total_population
                    ),
                }
            )
    for threshold, population in zip(
        POPULATION_AFFECTED_THRESHOLDS_PERCENT,
        summary.population_affected[-1],
        strict=True,
    ):
        rows.append(
            {
                "comparison_type": "non_consensus",
                "implementation_a": None,
                "implementation_b": None,
                "minimum_rate_percent": threshold,
                "estimated_population_affected": int(population),
                "percent_total_population": (
                    100.0 * int(population) / summary.total_population
                ),
            }
        )
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("comparison_type", pa.string()),
                ("implementation_a", pa.string()),
                ("implementation_b", pa.string()),
                ("minimum_rate_percent", pa.uint8()),
                ("estimated_population_affected", pa.int64()),
                ("percent_total_population", pa.float64()),
            ]
        ),
    )


def _write_figure(summary: ISO7243PlotSummary, png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), layout="constrained")
    map_axis = axes[0, 0]
    changed_percent = 100.0 * summary.changed_fraction
    map_x, map_y = equal_earth_coordinates(summary.longitude, summary.latitude)
    image = map_axis.scatter(
        map_x,
        map_y,
        c=changed_percent,
        s=0.35,
        marker="s",
        linewidths=0,
        cmap="magma",
        vmin=0.0,
        vmax=50.0,
        rasterized=True,
    )
    configure_equal_earth_axis(map_axis)
    # map_axis.set_title("Mean classification disagreement across implementation pairs")
    colorbar = figure.colorbar(
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
    map_x, map_y = equal_earth_coordinates(summary.longitude, summary.latitude)
    for axis, (implementation_a, implementation_b), values in zip(
        axes.flat,
        summary.implementation_pairs,
        summary.pair_changed_fraction,
    ):
        image = axis.scatter(
            map_x,
            map_y,
            c=100.0 * values,
            s=0.25,
            marker="s",
            linewidths=0,
            cmap="magma",
            vmin=0.0,
            vmax=color_upper_percent,
            rasterized=True,
        )
        configure_equal_earth_axis(axis)
        axis.set_title(f"{implementation_a} vs {implementation_b}")
    for axis in tuple(axes.flat)[len(summary.implementation_pairs) :]:
        axis.set_visible(False)
    assert image is not None
    colorbar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="WBGT class disagreement (% valid hours)",
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
        xlabel="Affected population (% of total population)",
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

    thresholds = np.asarray(POPULATION_AFFECTED_THRESHOLDS_PERCENT)
    population_percent = (
        100.0 * summary.population_affected[-1].astype(np.float64)
        / summary.total_population
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
    axis.plot(thresholds, population_percent, color="#4c78a8", marker="o", linewidth=2)
    axis.set(
        xlabel="Non-consensus rate (% valid hours)",
        ylabel="Population affected (% of total population)",
        xticks=thresholds,
        xlim=(thresholds[0] - 1, thresholds[-1] + 1),
        ylim=(0.0, 100.0),
    )
    axis.grid(color="#d9d9d9", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.savefig(png, dpi=220)
    plt.close(figure)


def _dynamic_color_upper_percent(values: np.ndarray) -> float:
    """Choose a shared robust limit without letting rare extremes flatten every map."""

    finite = values[np.isfinite(values)]
    if not len(finite):
        return 1.0
    return max(1.0, float(np.ceil(np.quantile(finite, 0.99))))


def configure_percent_colorbar(colorbar, upper: float) -> None:
    """Include the endpoint and explicitly mark saturated percentage colours."""
    ticks = colorbar.get_ticks()
    ticks = ticks[(ticks >= 0) & (ticks < upper)]
    if len(ticks) > 1:
        spacing = float(np.median(np.diff(ticks)))
        ticks = ticks[upper - ticks >= spacing / 2]
    ticks = np.unique(np.append(ticks, upper))
    colorbar.set_ticks(ticks, labels=[
        f"≥ {tick:g}" if tick == upper and upper < 100 else f"{tick:g}"
        for tick in ticks
    ])


def _integer_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64
    )


def _float_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    )
