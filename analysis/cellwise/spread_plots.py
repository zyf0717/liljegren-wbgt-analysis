"""Rendering for cellwise quantile-spread products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from .projection import (
    PANEL_TITLE_FONT_SIZE_PT,
    STACKED_MAP_DPI,
    add_map_colorbar,
    configure_map_axis,
    country_boundary_from_metadata,
    plot_country_boundary,
    plot_map_cells,
    regional_map_figure_size,
    save_figure,
)
from .spread import PERCENTILES, SPREAD_METRICS, Weighting

SPREAD_MAP_COLOR_LIMITS_C = (0.0, 10.0)
STACKED_QUANTITIES = (
    ("wbgt", "WBGT"),
    ("tnwb", r"$T_{\mathrm{nwb}}$"),
    ("tg", r"$T_g$"),
)


def plot_quantile_spread(input_root: Path) -> None:
    root = Path(input_root)
    metadata = json.loads((root / "metadata.json").read_text())
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    write_figures(
        pq.read_table(root / "global_quantile_spread_summary.parquet"),
        pq.read_table(root / "cellwise_quantile_spread.parquet"),
        directory,
        metadata["weighting"],
        region_label=metadata.get("country_selection", {}).get("country_name"),
        country_boundary=country_boundary_from_metadata(metadata),
    )


def plot_stacked_cellwise_quantile_spread(
    output_root: str | Path,
) -> tuple[Path, ...]:
    """Plot separate P50/P95 WBGT, Tnwb, and Tg map stacks."""

    root = Path(output_root).expanduser().resolve()
    columns = ["latitude", "longitude", *SPREAD_METRICS]
    tables = {
        quantity: pq.read_table(
            root / quantity / "cellwise_quantile_spread.parquet",
            columns=columns,
        )
        for quantity, _ in STACKED_QUANTITIES
    }
    return write_stacked_spread_maps(tables, root / "figures")


def write_figures(
    summary: pa.Table,
    cellwise: pa.Table,
    directory: Path,
    weighting: Weighting,
    region_label: str | None = None,
    country_boundary: BaseGeometry | None = None,
) -> None:
    import matplotlib.pyplot as plt

    metric_names = summary["metric"].to_pylist()
    rows = [metric_names.index(f"spread_{percentile}_c") for percentile in PERCENTILES]
    x = [int(p[1:]) for p in PERCENTILES]
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
        ylabel="Range of estimates (°C)",
        xticks=x,
    )
    axis.legend()
    save_figure(figure, directory / "spread_by_percentile.png", dpi=200)
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
    import matplotlib.pyplot as plt

    longitude = np.asarray(cellwise["longitude"], dtype=np.float64)
    latitude = np.asarray(cellwise["latitude"], dtype=np.float64)
    for metric in SPREAD_METRICS:
        figure_size = (
            regional_map_figure_size(longitude, latitude)
            if region_label is not None
            else (10.0, 4.8)
        )
        figure, axis = plt.subplots(figsize=figure_size, layout="constrained")
        values = np.asarray(cellwise[metric], dtype=np.float64)
        image = plot_map_cells(
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
        configure_map_axis(
            axis,
            longitude if region_label else None,
            latitude if region_label else None,
        )
        plot_country_boundary(axis, country_boundary)
        colorbar = add_map_colorbar(figure, image, ax=axis, label="Spread (°C)")
        ticks = np.arange(
            SPREAD_MAP_COLOR_LIMITS_C[0], SPREAD_MAP_COLOR_LIMITS_C[1] + 1, 2
        )
        colorbar.set_ticks(ticks, labels=[f"{tick:g}" for tick in ticks])
        save_figure(figure, directory / f"{metric.replace('_c', '')}_map.png")


def write_stacked_spread_maps(
    cellwise_by_quantity: Mapping[str, pa.Table], directory: Path
) -> tuple[Path, ...]:
    """Render one vertical global map stack per percentile."""

    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for percentile in PERCENTILES:
        figure, axes = plt.subplots(
            len(STACKED_QUANTITIES),
            1,
            figsize=(10.0, 10.5),
            layout="constrained",
        )
        figure.get_layout_engine().set(h_pad=0.08, hspace=0.04)
        image = None
        for panel_label, axis, (quantity, quantity_label) in zip(
            ("A", "B", "C"), axes, STACKED_QUANTITIES, strict=True
        ):
            cellwise = cellwise_by_quantity[quantity]
            longitude = np.asarray(cellwise["longitude"], dtype=np.float64)
            latitude = np.asarray(cellwise["latitude"], dtype=np.float64)
            image = plot_map_cells(
                axis,
                longitude,
                latitude,
                np.asarray(cellwise[f"spread_{percentile}_c"], dtype=np.float64),
                regional=False,
                point_size=0.35,
                cmap="magma",
                vmin=SPREAD_MAP_COLOR_LIMITS_C[0],
                vmax=SPREAD_MAP_COLOR_LIMITS_C[1],
            )
            configure_map_axis(axis)
            axis.set_title(
                f"{panel_label}) {quantity_label}",
                fontsize=PANEL_TITLE_FONT_SIZE_PT,
            )
        if image is None:
            raise ValueError("No spread maps were rendered")
        colorbar = add_map_colorbar(
            figure,
            image,
            ax=axes,
            label="Range (°C)",
            shrink=0.8,
        )
        ticks = np.arange(
            SPREAD_MAP_COLOR_LIMITS_C[0], SPREAD_MAP_COLOR_LIMITS_C[1] + 1, 2
        )
        colorbar.set_ticks(ticks, labels=[f"{tick:g}" for tick in ticks])
        output = directory / f"stacked_spread_{percentile}.png"
        save_figure(
            figure,
            output,
            dpi=STACKED_MAP_DPI,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        outputs.append(output)
    return tuple(outputs)
