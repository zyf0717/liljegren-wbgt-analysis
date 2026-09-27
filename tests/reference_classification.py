"""Small in-memory reference used to check streamed ERA5 classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationAnalysis:
    """Optional row-level results and tabular classification summaries."""

    classes: pd.DataFrame
    classified: pd.DataFrame | None
    paired: pd.DataFrame | None
    outputs: Mapping[str, pd.DataFrame]


def class_definitions(thresholds_c: Sequence[float]) -> pd.DataFrame:
    """Build stable numeric IDs and labels for threshold bands."""

    thresholds = tuple(thresholds_c)
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
    implementations: Sequence[str] | None = None,
) -> ClassificationAnalysis:
    """Classify valid WBGT values and compare every implementation pair.

    ``observations`` must contain one valid WBGT row for each
    ``pair_keys``/``implementation`` combination.
    """

    thresholds = tuple(thresholds_c)
    selected = tuple(implementations or sorted(observations["implementation"].unique()))
    observations = observations.loc[
        observations["implementation"].isin(selected)
    ].copy()
    classes = class_definitions(thresholds)
    classified = observations.copy()
    values = pd.to_numeric(classified["wbgt_c"], errors="coerce")
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
    paired = paired.loc[paired["implementation_a"] < paired["implementation_b"]].copy()
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
    complete = _complete_implementation_observations(
        classified, pair_keys, len(selected)
    )
    outputs["non_consensus_summary.csv"] = _summarize_non_consensus(complete, [])
    outputs["monthly_non_consensus_summary.csv"] = _summarize_non_consensus(
        complete, ["result_month"]
    )
    return ClassificationAnalysis(classes, classified, paired, outputs)


def _complete_implementation_observations(
    classified: pd.DataFrame,
    pair_keys: Sequence[str],
    implementation_count: int,
) -> pd.DataFrame:
    dimensions = [column for column in pair_keys if column in classified.columns]
    complete = classified.groupby(dimensions, as_index=False, sort=True).agg(
        result_month=("result_month", "first"),
        n_implementations=("implementation", "nunique"),
        n_classes=("class_id", "nunique"),
    )
    complete = complete.loc[
        complete["n_implementations"].eq(implementation_count)
    ].copy()
    complete["non_consensus"] = complete["n_classes"].gt(1)
    return complete


def _summarize_non_consensus(
    complete: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    if group_columns:
        summary = complete.groupby(group_columns, as_index=False, sort=True).agg(
            n_joint=("non_consensus", "size"),
            n_non_consensus=("non_consensus", "sum"),
        )
    else:
        summary = pd.DataFrame(
            {
                "n_joint": [len(complete)],
                "n_non_consensus": [int(complete["non_consensus"].sum())],
            }
        )
    summary["non_consensus_percent"] = (
        100.0 * summary["n_non_consensus"] / summary["n_joint"]
    ).round(4)
    return summary[
        [*group_columns, "n_joint", "n_non_consensus", "non_consensus_percent"]
    ]


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
