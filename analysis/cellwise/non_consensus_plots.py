"""Map persisted cellwise ISO 7243 non-consensus percentages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from ..config import DEFAULT_ISO7243_OUTPUT_ROOT
from .iso7243 import load_hourly_metadata
from .projection import (
    add_map_colorbar,
    configure_map_axis,
    configure_percent_colorbar,
    country_boundary_from_metadata,
    percent_color_upper,
    plot_country_boundary,
    plot_major_cities,
    plot_map_cells,
    regional_map_figure_size,
)


def plot_cellwise_iso7243_non_consensus(
    input_root: str | Path = DEFAULT_ISO7243_OUTPUT_ROOT,
    *,
    output_directory: str | Path | None = None,
) -> Path:
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
    cell_ids = np.asarray(lookup["cell_id"], dtype=np.int64)
    if not np.array_equal(np.asarray(values["cell_id"], dtype=np.int64), cell_ids):
        raise ValueError("Non-consensus values do not match the cell lookup")
    percent = np.asarray(values["non_consensus_percent"], dtype=np.float64)
    target.mkdir(parents=True, exist_ok=True)
    png = target / "non_consensus_percent.png"
    country_selection = metadata.get("country_selection", {})
    region_label = country_selection.get("country_name")
    country_iso_a3 = country_selection.get("country_iso_a3")
    _write_map(
        np.asarray(lookup["latitude"], dtype=np.float64),
        np.asarray(lookup["longitude"], dtype=np.float64),
        percent,
        png,
        region_label=region_label,
        country_boundary=country_boundary_from_metadata(metadata),
        country_iso_a3=country_iso_a3,
    )
    return png


def _write_map(
    latitude: np.ndarray,
    longitude: np.ndarray,
    percent: np.ndarray,
    png: Path,
    *,
    region_label: str | None = None,
    country_boundary: BaseGeometry | None = None,
    country_iso_a3: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    valid = np.isfinite(percent)
    upper = percent_color_upper(percent)
    figure_size = (
        regional_map_figure_size(longitude, latitude)
        if region_label is not None
        else (10.0, 4.8)
    )
    figure, axis = plt.subplots(figsize=figure_size, layout="constrained")
    image = plot_map_cells(
        axis,
        longitude[valid],
        latitude[valid],
        percent[valid],
        regional=region_label is not None,
        point_size=0.35,
        cmap="cividis",
        vmin=0.0,
        vmax=upper,
    )
    configure_map_axis(
        axis, longitude if region_label else None, latitude if region_label else None
    )
    plot_country_boundary(axis, country_boundary)
    plot_major_cities(axis, country_iso_a3)
    colorbar = add_map_colorbar(
        figure,
        image,
        ax=axis,
        label="WBGT classification non-consensus (% valid hours)",
        capped=upper < 100.0,
    )
    configure_percent_colorbar(colorbar, upper)
    figure.savefig(png, dpi=220)
    plt.close(figure)
