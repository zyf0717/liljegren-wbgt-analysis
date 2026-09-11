"""Map persisted cellwise ISO 7243 non-consensus percentages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from .non_consensus import DEFAULT_OUTPUT_ROOT
from .iso7243 import load_hourly_metadata
from .iso7243_plots import configure_percent_colorbar, _dynamic_color_upper_percent
from .projection import (
    configure_equal_earth_axis,
    country_boundary_from_metadata,
    plot_country_boundary,
    plot_equal_earth_cells,
    plot_major_cities,
    regional_map_figure_size,
)


@dataclass(frozen=True)
class NonConsensusFigureOutputs:
    """Rendered non-consensus percentage map."""

    png: Path


def plot_cellwise_iso7243_non_consensus(
    input_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    output_directory: str | Path | None = None,
) -> NonConsensusFigureOutputs:
    """Write a map of annual joint-valid rows lacking ISO class unanimity."""

    root = Path(input_root).expanduser().resolve()
    target = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else root / "figures"
    )
    metadata = load_hourly_metadata(root)
    lookup = pq.read_table(root / "cell_lookup.parquet")
    values = pq.read_table(root / "cellwise_non_consensus.parquet")
    cell_ids = _integers(lookup, "cell_id")
    if not np.array_equal(_integers(values, "cell_id"), cell_ids):
        raise ValueError("Non-consensus values do not match the cell lookup")
    percent = _floats(values, "non_consensus_percent")
    valid = np.isfinite(percent)
    joint = _integers(values, "n_joint")
    changed = _integers(values, "n_non_consensus")
    expected = np.divide(100.0 * changed, joint, out=np.full(len(joint), np.nan), where=joint > 0)
    if ((joint < 0).any() or (changed < 0).any() or (changed > joint).any()
        or not np.allclose(percent, expected, equal_nan=True)
        or (percent[valid] < 0.0).any() or (percent[valid] > 100.0).any()):
        raise ValueError("Non-consensus percentages are invalid")
    target.mkdir(parents=True, exist_ok=True)
    png = target / "non_consensus_percent.png"
    country_selection = metadata.get("country_selection")
    is_country_input = (
        metadata.get("input_source") == "country-parquet"
        and isinstance(country_selection, dict)
    )
    region_label = (
        str(country_selection.get("country_name"))
        if is_country_input
        else None
    )
    raw_country_iso_a3 = (
        country_selection.get("country_iso_a3") if is_country_input else None
    )
    country_iso_a3 = (
        raw_country_iso_a3 if isinstance(raw_country_iso_a3, str) else None
    )
    _write_map(
        _floats(lookup, "latitude"),
        _floats(lookup, "longitude"),
        percent,
        png,
        region_label=region_label,
        country_boundary=country_boundary_from_metadata(metadata),
        country_iso_a3=country_iso_a3,
    )
    return NonConsensusFigureOutputs(png)


def _write_map(latitude: np.ndarray, longitude: np.ndarray, percent: np.ndarray,
               png: Path, *, region_label: str | None = None,
               country_boundary: BaseGeometry | None = None,
               country_iso_a3: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = np.isfinite(percent)
    upper = _dynamic_color_upper_percent(percent)
    figure_size = (
        regional_map_figure_size(longitude, latitude)
        if region_label is not None
        else (10.0, 4.8)
    )
    figure, axis = plt.subplots(figsize=figure_size, layout="constrained")
    image = plot_equal_earth_cells(
        axis,
        longitude[valid],
        latitude[valid],
        percent[valid],
        regional=region_label is not None,
        point_size=0.35,
        cmap="magma",
        vmin=0.0,
        vmax=upper,
    )
    configure_equal_earth_axis(
        axis, longitude if region_label else None, latitude if region_label else None
    )
    plot_country_boundary(axis, country_boundary)
    plot_major_cities(axis, country_iso_a3)
    colorbar = figure.colorbar(
        image,
        ax=axis,
        label="WBGT classification non-consensus (% valid hours)",
    )
    configure_percent_colorbar(colorbar, upper)
    figure.savefig(png, dpi=220)
    plt.close(figure)


def _integers(table: object, column: str) -> np.ndarray:
    return np.asarray(table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)


def _floats(table: object, column: str) -> np.ndarray:
    return np.asarray(table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64)
