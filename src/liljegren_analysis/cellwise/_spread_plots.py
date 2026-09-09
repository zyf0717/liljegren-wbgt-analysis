"""Rendering for cellwise quantile-spread products."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa

from .projection import configure_equal_earth_axis, equal_earth_coordinates

PERCENTILES = ("p50", "p90", "p95", "p99")
SPREAD_METRICS = tuple(f"spread_{percentile}_c" for percentile in PERCENTILES)
SPREAD_MAP_COLOR_LIMITS_C = (0.0, 8.0)
Weighting = Literal["none", "geometry", "population"]


def write_figures(
    summary: pa.Table,
    cellwise: pa.Table,
    directory: Path,
    weighting: Weighting,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = summary["metric"].to_pylist()
    rows = [metric_names.index(f"spread_{percentile}_c") for percentile in PERCENTILES]
    x = np.asarray([50, 90, 95, 99])
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
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"spread_by_percentile.{suffix}", dpi=200)
    plt.close(figure)
    _write_spread_maps(cellwise, directory)


def _write_spread_maps(cellwise: pa.Table, directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    map_x, map_y = equal_earth_coordinates(
        _float_column(cellwise, "longitude"),
        _float_column(cellwise, "latitude"),
    )
    for metric in SPREAD_METRICS:
        figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
        values = _float_column(cellwise, metric)
        image = axis.scatter(
            map_x,
            map_y,
            c=values,
            s=0.35,
            marker="s",
            linewidths=0,
            cmap="magma",
            vmin=SPREAD_MAP_COLOR_LIMITS_C[0],
            vmax=SPREAD_MAP_COLOR_LIMITS_C[1],
            rasterized=True,
        )
        configure_equal_earth_axis(axis)
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
        for suffix in ("png", "pdf"):
            figure.savefig(
                directory / f"{metric.replace('_c', '')}_map.{suffix}", dpi=220
            )
        plt.close(figure)


def _float_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(
        table[column].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    )
