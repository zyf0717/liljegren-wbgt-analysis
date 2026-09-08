"""Figures for persisted cellwise ISO 7243 classification differences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..comparison.area_weighted import _ERA5_LATITUDE_WEIGHTS
from .paths import DEFAULT_ISO7243_OUTPUT_ROOT
from .iso7243 import load_hourly_metadata
from .population import DEFAULT_OUTPUT_ROOT as DEFAULT_POPULATION_OUTPUT_ROOT
from .projection import configure_equal_earth_axis, equal_earth_coordinates


DEFAULT_INPUT_ROOT = DEFAULT_ISO7243_OUTPUT_ROOT
DEFAULT_POPULATION_WEIGHTS_PATH = DEFAULT_POPULATION_OUTPUT_ROOT / "population.parquet"
_WEIGHTINGS = ("Unweighted", "Geometric-weighted", "Population-weighted")


@dataclass(frozen=True)
class ISO7243FigureOutputs:
    """Rendered overview and pairwise PNG paths."""

    png: Path
    pairwise_png: Path


@dataclass(frozen=True)
class ISO7243PlotSummary:
    """Compact aggregates used by the overview figure."""

    latitude: np.ndarray
    longitude: np.ndarray
    changed_fraction: np.ndarray
    histogram_shares: np.ndarray
    implementation_pairs: tuple[tuple[str, str], ...]
    pair_changed_fraction: np.ndarray


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
    _write_figure(summary, png)
    _write_pairwise_maps(summary, pairwise_png)
    return ISO7243FigureOutputs(png, pairwise_png)


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
    weights = np.vstack((np.ones(len(cell_ids)), _geometry_weights(source_ids),
                         _population_weights(source_ids, population_weights_path)))
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


def _population_weights(source_ids: list[str], path: str | Path) -> np.ndarray:
    population_path = Path(path).expanduser().resolve()
    if not population_path.is_file():
        raise FileNotFoundError(
            f"Population weights do not exist: {population_path}; run "
            "scripts/compute_era5_population_weights.py first"
        )
    table = pq.read_table(population_path, columns=["source_cell_id", "population"])
    population_ids = table["source_cell_id"].combine_chunks().to_pylist()
    population = _integer_column(table, "population").astype(np.float64)
    if len(population_ids) != len(set(population_ids)) or (population < 0.0).any():
        raise ValueError("Population weights contain invalid source_cell_id or population values")
    lookup = dict(zip(population_ids, population, strict=True))
    weights = np.asarray([lookup.get(source_id, np.nan) for source_id in source_ids])
    if not np.isfinite(weights).all():
        raise ValueError("Population weights are missing a cell from the ISO 7243 lookup")
    return weights


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
        label="WBGT class disagreement (% valid hours) — shared P99 cutoff",
        shrink=0.86,
    )
    configure_percent_colorbar(colorbar, color_upper_percent)
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
