"""Render bounded-memory classification counters as stable CSV tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def change_summary(
    monthly_counts: dict[tuple[str, str, str], np.ndarray], *, monthly: bool
) -> pd.DataFrame:
    counts = monthly_counts if monthly else _sum_without_month(monthly_counts)
    rows = []
    for key, values in sorted(counts.items()):
        n_paired, n_changed, a_higher, b_higher = map(int, values)
        rows.append(
            (
                *key,
                n_paired,
                n_changed,
                round(100.0 * n_changed / n_paired, 4),
                a_higher,
                b_higher,
            )
        )
    groups = ["result_month"] if monthly else []
    return pd.DataFrame(
        rows,
        columns=[
            *groups,
            "implementation_a",
            "implementation_b",
            "n_paired",
            "n_changed",
            "percent_changed",
            "a_higher_class",
            "b_higher_class",
        ],
    )


def transition_summary(
    monthly_counts: dict[tuple[str, str, str, int, int], int],
    classes: pd.DataFrame,
    *,
    monthly: bool,
) -> pd.DataFrame:
    counts = monthly_counts if monthly else _sum_without_month(monthly_counts)
    labels = classes.set_index("class_id")["class_label"]
    rows = []
    for key, count in sorted(counts.items()):
        *groups, class_a, class_b = key
        rows.append(
            (
                *groups,
                class_a,
                class_b,
                count,
                labels[class_a],
                labels[class_b],
                class_a != class_b,
            )
        )
    groups = ["result_month"] if monthly else []
    return pd.DataFrame(
        rows,
        columns=[
            *groups,
            "implementation_a",
            "implementation_b",
            "class_id_a",
            "class_id_b",
            "n",
            "class_a_label",
            "class_b_label",
            "classification_changed",
        ],
    )


def non_consensus_summary(
    monthly_counts: dict[str, np.ndarray], *, monthly: bool
) -> pd.DataFrame:
    if monthly:
        items = sorted(monthly_counts.items())
    else:
        total = sum(monthly_counts.values(), start=np.zeros(2, dtype=np.uint64))
        items = [(None, total)]
    rows = []
    for month, values in items:
        n_joint, n_non_consensus = map(int, values)
        percent = round(100.0 * n_non_consensus / n_joint, 4) if n_joint else np.nan
        row = (n_joint, n_non_consensus, percent)
        rows.append((month, *row) if monthly else row)
    groups = ["result_month"] if monthly else []
    return pd.DataFrame(
        rows,
        columns=[*groups, "n_joint", "n_non_consensus", "non_consensus_percent"],
    )


def _sum_without_month(
    counts: dict[tuple, int | np.ndarray],
) -> dict[tuple, int | np.ndarray]:
    totals: dict[tuple, int | np.ndarray] = {}
    for key, value in counts.items():
        group = key[1:]
        if group not in totals:
            totals[group] = value.copy() if isinstance(value, np.ndarray) else value
        else:
            totals[group] += value
    return totals
