"""Shared WBGT threshold-classification analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationAnalysis:
    """Classified observations, implementation pairs, and tabular summaries."""

    classes: pd.DataFrame
    classified: pd.DataFrame
    paired: pd.DataFrame
    outputs: Mapping[str, pd.DataFrame]


def validate_thresholds(thresholds_c: Sequence[float]) -> tuple[float, ...]:
    """Return finite, strictly increasing WBGT thresholds."""

    thresholds = tuple(float(value) for value in thresholds_c)
    if (
        not thresholds
        or any(not math.isfinite(value) for value in thresholds)
        or any(left >= right for left, right in zip(thresholds, thresholds[1:]))
    ):
        raise ValueError("thresholds_c must be finite and strictly increasing")
    return thresholds


def class_definitions(thresholds_c: Sequence[float]) -> pd.DataFrame:
    """Build stable numeric IDs and labels for threshold bands."""

    thresholds = validate_thresholds(thresholds_c)
    bounds = (-np.inf, *thresholds, np.inf)
    labels = [
        f"< {upper:g} C"
        if index == 0
        else f">= {lower:g} C"
        if index == len(thresholds)
        else f"[{lower:g}, {upper:g}) C"
        for index, (lower, upper) in enumerate(zip(bounds, bounds[1:]))
    ]
    return pd.DataFrame({"class_id": range(len(labels)), "class_label": labels})


def analyze_wbgt_classification(
    observations: pd.DataFrame,
    thresholds_c: Sequence[float],
    *,
    pair_keys: Sequence[str],
    month_column: str,
    summary_dimensions: Sequence[str] = (),
) -> ClassificationAnalysis:
    """Classify valid WBGT values and compare every implementation pair.

    ``observations`` must contain one valid WBGT row for each
    ``pair_keys``/``implementation`` combination.  ``summary_dimensions`` are
    emitted as both change and transition summaries in addition to overall and
    monthly summaries.
    """

    required = {*pair_keys, "implementation", "wbgt_c", month_column}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"observations is missing columns: {missing}")
    if observations.empty:
        raise ValueError("observations is empty")

    thresholds = validate_thresholds(thresholds_c)
    classes = class_definitions(thresholds)
    classified = observations.copy()
    values = pd.to_numeric(classified["wbgt_c"], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError("observations contains missing or non-finite wbgt_c values")
    duplicate_keys = [*pair_keys, "implementation"]
    if classified.duplicated(duplicate_keys).any():
        raise ValueError("observations contains duplicate implementation rows")
    classified["class_id"] = np.searchsorted(thresholds, values, side="right").astype(
        "int8"
    )
    labels = classes.set_index("class_id")["class_label"]
    classified["class_label"] = classified["class_id"].map(labels)
    classified["result_month"] = pd.to_datetime(
        classified[month_column], utc=True
    ).dt.strftime("%Y-%m")

    pair_values = [*pair_keys, "result_month", "implementation", "wbgt_c", "class_id"]
    paired = classified[pair_values].merge(
        classified[pair_values],
        on=list(pair_keys),
        how="inner",
        suffixes=("_a", "_b"),
        validate="many_to_many",
    )
    paired = paired.loc[
        paired["implementation_a"] < paired["implementation_b"]
    ].copy()
    if paired.empty:
        raise ValueError("fewer than two implementations have matched observations")
    paired["wbgt_difference_c"] = paired["wbgt_c_a"] - paired["wbgt_c_b"]
    if not paired["result_month_a"].eq(paired["result_month_b"]).all():
        raise ValueError("matched observations disagree on result month")
    paired["result_month"] = paired.pop("result_month_a")
    paired.pop("result_month_b")
    paired["class_difference"] = paired["class_id_a"] - paired["class_id_b"]
    paired["classification_changed"] = paired["class_difference"].ne(0)

    pair_columns = ["implementation_a", "implementation_b"]
    outputs: dict[str, pd.DataFrame] = {"classes.csv": classes}
    _add_summaries(outputs, paired, classes, pair_columns, "pair")
    _add_summaries(
        outputs, paired, classes, ["result_month", *pair_columns], "monthly_pair"
    )
    for dimension in summary_dimensions:
        if dimension not in paired.columns:
            raise ValueError(f"summary dimension is missing from pair keys: {dimension}")
        _add_summaries(
            outputs, paired, classes, [dimension, *pair_columns], f"{dimension}_pair"
        )

    return ClassificationAnalysis(classes, classified, paired, outputs)


def write_outputs(outputs: Mapping[str, pd.DataFrame], output_directory: str | Path) -> None:
    """Persist all analysis tables as CSV files."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(directory / filename, index=False)


def _add_summaries(
    outputs: dict[str, pd.DataFrame],
    paired: pd.DataFrame,
    classes: pd.DataFrame,
    group_columns: list[str],
    prefix: str,
) -> None:
    outputs[f"{prefix}_change_summary.csv"] = _summarize_changes(paired, group_columns)
    outputs[f"{prefix}_transition_counts.csv"] = _count_transitions(
        paired, classes, group_columns
    )


def _summarize_changes(paired: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        paired.assign(
            a_higher_class=paired["class_difference"].gt(0),
            b_higher_class=paired["class_difference"].lt(0),
        )
        .groupby(group_columns, as_index=False, sort=True)
        .agg(
            n_paired=("classification_changed", "size"),
            n_changed=("classification_changed", "sum"),
            a_higher_class=("a_higher_class", "sum"),
            b_higher_class=("b_higher_class", "sum"),
        )
    )
    summary["percent_changed"] = (
        100.0 * summary["n_changed"] / summary["n_paired"]
    ).round(4)
    return summary[
        [
            *group_columns,
            "n_paired",
            "n_changed",
            "percent_changed",
            "a_higher_class",
            "b_higher_class",
        ]
    ]


def _count_transitions(
    paired: pd.DataFrame, classes: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    columns = [*group_columns, "class_id_a", "class_id_b"]
    counts = (
        paired.groupby(columns, as_index=False, sort=True)
        .size()
        .rename(columns={"size": "n"})
    )
    labels = classes.set_index("class_id")["class_label"]
    counts["class_a_label"] = counts["class_id_a"].map(labels)
    counts["class_b_label"] = counts["class_id_b"].map(labels)
    counts["classification_changed"] = counts["class_id_a"].ne(counts["class_id_b"])
    return counts
