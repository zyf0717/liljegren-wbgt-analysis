"""Shared WBGT threshold-classification analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


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
    """Describe lower-inclusive threshold bands; thresholds are validated at entry."""
    bounds = (-np.inf, *thresholds_c, np.inf)
    rows = []
    for index, (lower, upper) in enumerate(zip(bounds, bounds[1:])):
        if index == 0:
            label, definition = f"< {upper:g} C", f"wbgt_c < {upper:g}"
        elif index == len(thresholds_c):
            label, definition = f">= {lower:g} C", f"wbgt_c >= {lower:g}"
        else:
            label, definition = (
                f"[{lower:g}, {upper:g}) C",
                f"{lower:g} <= wbgt_c < {upper:g}",
            )
        rows.append((index, label, definition))
    return pd.DataFrame(
        rows, columns=["class_id", "class_label", "threshold_definition"]
    )
