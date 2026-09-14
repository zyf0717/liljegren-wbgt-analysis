"""Rendering for cellwise quantile-spread products."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
from shapely.geometry.base import BaseGeometry

from ._pipeline import DEFAULT_PROBABILITIES
from .projection import (
    configure_equal_earth_axis,
    plot_country_boundary,
    plot_equal_earth_cells,
    regional_map_figure_size,
)

PERCENTILES = tuple(
    f"p{int(probability * 100):02d}" for probability in DEFAULT_PROBABILITIES
)
SPREAD_METRICS = tuple(f"spread_{percentile}_c" for percentile in PERCENTILES)
SPREAD_MAP_COLOR_LIMITS_C = (0.0, 8.0)
Weighting = Literal["none", "geometry", "population"]


def write_figures(
    summary: pa.Table,
    cellwise: pa.Table,
    directory: Path,
    weighting: Weighting,
    region_label: str | None = None,
    country_boundary: BaseGeometry | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = summary["metric"].to_pylist()
    rows = [metric_names.index(f"spread_{percentile}_c") for percentile in PERCENTILES]
    x = np.asarray(DEFAULT_PROBABILITIES) * 100
    figure, axis = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    weight_label = {
        "none": "Unweighted",
        "geometry": "Geometric-weighted",
        "population": "Population-weighted",
    }[weighting]
    for column, label in (
        ("weighted_median_c", f"{weight_label} median"),
        ("weighted_mean_c", f"{weight_label} mean"),
        ("weighted_p95_c", f"{weight_label} P95"),
    ):
        axis.plot(
            x, [summary[column][row].as_py() for row in rows], marker="o", label=label
        )
    axis.set(
        xlabel="Percentile",
        ylabel="Range of WBGT (°C) estimates",
        xticks=x,
    )
    axis.legend()
    figure.savefig(directory / "spread_by_percentile.png", dpi=200)
    plt.close(figure)
    _write_spread_maps(
        cellwise,
        directory,
        region_label=region_label,
        country_boundary=country_boundary,
    )


def _write_spread_maps(
    cellwise: pa.Table,
    directory: Path,
    *,
    region_label: str | None = None,
    country_boundary: BaseGeometry | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    longitude = _float_column(cellwise, "longitude")
    latitude = _float_column(cellwise, "latitude")
    for metric in SPREAD_METRICS:
        figure_size = (
            regional_map_figure_size(longitude, latitude)
            if region_label is not None
            else (10.0, 4.8)
        )
        figure, axis = plt.subplots(figsize=figure_size, layout="constrained")
        values = _float_column(cellwise, metric)
        image = plot_equal_earth_cells(
            axis,
            longitude,
            latitude,
            values,
            regional=region_label is not None,
            point_size=0.35,
            cmap="magma",
            vmin=SPREAD_MAP_COLOR_LIMITS_C[0],
            vmax=SPREAD_MAP_COLOR_LIMITS_C[1],
        )
        configure_equal_earth_axis(
            axis,
            longitude if region_label else None,
            latitude if region_label else None,
        )
        plot_country_boundary(axis, country_boundary)
        colorbar = figure.colorbar(image, ax=axis, label="Spread (°C)")
        ticks = colorbar.get_ticks()
        colorbar.set_ticks(
            ticks,
            labels=[
                f"≥ {tick:g}"
                if np.isclose(tick, SPREAD_MAP_COLOR_LIMITS_C[1])
                else f"{tick:g}"
                for tick in ticks
            ],
        )
        figure.savefig(
            directory / f"{metric.replace('_c', '')}_map.png", dpi=220
        )
        plt.close(figure)


def _float_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    )
