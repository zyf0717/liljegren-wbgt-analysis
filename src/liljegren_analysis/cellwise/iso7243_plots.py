"""Figures for persisted cellwise ISO 7243 classification differences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..comparison.area_weighted import _ERA5_LATITUDE_WEIGHTS
from .population import DEFAULT_OUTPUT_ROOT as DEFAULT_POPULATION_OUTPUT_ROOT


DEFAULT_INPUT_ROOT = Path("data/cellwise_iso7243_classification_changes")
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
    """Stream the product into map values and unweighted/weighted histograms."""

    root = Path(input_root).expanduser().resolve()
    metadata = _load_metadata(root / "metadata.json")
    implementation_pairs = _implementation_pairs(metadata)
    lookup = pq.read_table(
        root / "cell_lookup.parquet",
        columns=["cell_id", "source_cell_id", "latitude", "longitude"],
    )
    cell_ids = _integer_column(lookup, "cell_id")
    if not len(cell_ids) or len(cell_ids) != len(np.unique(cell_ids)) or (cell_ids < 0).any():
        raise ValueError("Cell lookup has invalid cell_id values")
    source_ids = lookup["source_cell_id"].combine_chunks().to_pylist()
    geometry_weights = _geometry_weights(source_ids)
    population_weights = _population_weights(source_ids, population_weights_path)
    max_cell_id = int(cell_ids.max())
    weights_by_cell = np.full((3, max_cell_id + 1), np.nan, dtype=np.float64)
    weights_by_cell[0, cell_ids] = 1.0
    weights_by_cell[1, cell_ids] = geometry_weights
    weights_by_cell[2, cell_ids] = population_weights

    changed_sum = np.zeros(max_cell_id + 1, dtype=np.float64)
    compared_sum = np.zeros(max_cell_id + 1, dtype=np.float64)
    lookup_positions = np.full(max_cell_id + 1, -1, dtype=np.int32)
    lookup_positions[cell_ids] = np.arange(len(cell_ids), dtype=np.int32)
    pair_changed_fraction = np.full(
        (len(implementation_pairs), len(cell_ids)), np.nan, dtype=np.float32
    )
    pair_index_by_code = _pair_index_by_code(metadata, implementation_pairs)
    histogram = np.zeros((3, 5), dtype=np.float64)
    totals = np.zeros(3, dtype=np.float64)
    product = pq.ParquetFile(root / "cell_pair_differences.parquet")
    for batch in product.iter_batches(
        columns=[
            "cell_id",
            "n_compared",
            "n_higher",
            "n_lower",
            "implementation_a_id",
            "implementation_b_id",
        ],
        batch_size=1_000_000,
    ):
        batch_ids = np.asarray(batch.column(0), dtype=np.int64)
        n_compared = np.asarray(batch.column(1), dtype=np.uint8)
        n_changed = np.asarray(batch.column(2), dtype=np.uint8) + np.asarray(
            batch.column(3), dtype=np.uint8
        )
        implementation_a = np.asarray(batch.column(4), dtype=np.uint8)
        implementation_b = np.asarray(batch.column(5), dtype=np.uint8)
        positions = np.full(len(batch_ids), -1, dtype=np.int32)
        in_range = (batch_ids >= 0) & (batch_ids <= max_cell_id)
        positions[in_range] = lookup_positions[batch_ids[in_range]]
        implementation_count = len(metadata["implementation_ids"])
        pair_indices = pair_index_by_code[
            implementation_a * implementation_count + implementation_b
        ]
        if (
            (batch_ids < 0).any()
            or (batch_ids > max_cell_id).any()
            or (positions < 0).any()
            or (pair_indices < 0).any()
            or (n_compared != 4).any()
            or (n_changed > 4).any()
            or (n_changed > n_compared).any()
        ):
            raise ValueError("Cell-pair differences do not match the lookup or count schema")
        np.add.at(changed_sum, batch_ids, n_changed)
        np.add.at(compared_sum, batch_ids, n_compared)
        for pair_index in np.unique(pair_indices):
            pair_rows = pair_indices == pair_index
            pair_positions = positions[pair_rows]
            if np.isfinite(pair_changed_fraction[pair_index, pair_positions]).any():
                raise ValueError("Cell-pair differences contain duplicate implementation pairs")
            pair_changed_fraction[pair_index, pair_positions] = (
                n_changed[pair_rows] / n_compared[pair_rows]
            )
        for index in range(3):
            weights = weights_by_cell[index, batch_ids]
            histogram[index] += np.bincount(n_changed, weights=weights, minlength=5)
            totals[index] += weights.sum()
    if product.metadata.num_rows != int(metadata["row_count"]):
        raise ValueError("Cell-pair row count disagrees with metadata")
    if not np.isfinite(totals).all() or (totals <= 0.0).any():
        raise ValueError("No positive mass available for one or more histogram weightings")
    if (compared_sum[cell_ids] <= 0.0).any():
        raise ValueError("One or more lookup cells have no cell-pair differences")
    expected_compared = 4 * int(metadata["implementation_pair_count"])
    if not np.equal(compared_sum[cell_ids], expected_compared).all():
        raise ValueError("Cell-pair differences are incomplete for one or more lookup cells")
    if not np.isfinite(pair_changed_fraction).all():
        raise ValueError("Cell-pair differences are incomplete for one or more implementation pairs")
    return ISO7243PlotSummary(
        latitude=_float_column(lookup, "latitude"),
        longitude=_float_column(lookup, "longitude"),
        changed_fraction=changed_sum[cell_ids] / compared_sum[cell_ids],
        histogram_shares=histogram / totals[:, np.newaxis],
        implementation_pairs=implementation_pairs,
        pair_changed_fraction=pair_changed_fraction,
    )


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ISO 7243 metadata: {path}")
    metadata = json.loads(path.read_text())
    required = ("row_count", "implementation_pair_count", "quantiles", "implementation_ids")
    if any(key not in metadata for key in required):
        raise ValueError("ISO 7243 metadata is incomplete")
    if metadata["quantiles"] != ["p50", "p90", "p95", "p99"]:
        raise ValueError("ISO 7243 product does not contain the expected quantiles")
    return metadata


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
    image = map_axis.scatter(
        summary.longitude,
        summary.latitude,
        c=changed_percent,
        s=0.35,
        marker="s",
        linewidths=0,
        cmap="magma",
        vmin=0.0,
        vmax=50.0,
        rasterized=True,
    )
    map_axis.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title="Mean classification disagreement across implementation pairs",
        xlim=(-180, 180),
        ylim=(-90, 90),
    )
    figure.colorbar(
        image,
        ax=map_axis,
        label="Changed classifications (%) — capped at 50%",
    )
    x = np.arange(5)
    for axis, label, shares in zip(
        axes.flat[1:], _WEIGHTINGS, summary.histogram_shares, strict=True
    ):
        axis.bar(x, 100.0 * shares, color="#386cb0")
        axis.set(
            xlabel="Classification differences per cell-pair",
            ylabel="Cell-pair share (%)",
            title=label,
            xticks=x,
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
        image = axis.scatter(
            summary.longitude,
            summary.latitude,
            c=100.0 * values,
            s=0.25,
            marker="s",
            linewidths=0,
            cmap="magma",
            vmin=0.0,
            vmax=color_upper_percent,
            rasterized=True,
        )
        axis.set(
            title=f"{implementation_a} vs {implementation_b}",
            xlim=(-180, 180),
            ylim=(-90, 90),
            xticks=(-180, 0, 180),
            yticks=(-90, 0, 90),
        )
    for axis in tuple(axes.flat)[len(summary.implementation_pairs) :]:
        axis.set_visible(False)
    assert image is not None
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label=f"Classification differences (%) — shared P99 scale ({color_upper_percent:g}%)",
        shrink=0.86,
    )
    figure.savefig(png, dpi=220)
    plt.close(figure)


def _dynamic_color_upper_percent(values: np.ndarray) -> float:
    """Choose a shared robust limit without letting rare extremes flatten every map."""

    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("Pairwise map values are not finite")
    return max(1.0, float(np.ceil(np.quantile(finite, 0.99))))


def _integer_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64
    )


def _float_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    )
