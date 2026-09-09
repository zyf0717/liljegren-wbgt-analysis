"""ERA5 country-cache classification analysis backed by Arrow Parquet reads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .core import (
    ClassificationAnalysis,
    analyze_wbgt_classification,
    classification_implementations,
)


@dataclass(frozen=True)
class Era5ClassificationAnalysis:
    classification: ClassificationAnalysis
    observation_summary: pd.DataFrame
    quantity_coverage: pd.DataFrame

    @property
    def outputs(self) -> dict[str, pd.DataFrame]:
        return {
            "observation_summary.csv": self.observation_summary,
            "quantity_coverage.csv": self.quantity_coverage,
            **self.classification.outputs,
        }


def analyze_era5_country_cache(
    cache_path: str | Path,
    start_date: date,
    end_date: date,
    thresholds_c: Sequence[float],
    *,
    implementations: Sequence[str] | None = None,
) -> Era5ClassificationAnalysis:
    """Analyze one preselected country Parquet cache without SQL or spill state."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"country cache does not exist: {path}")
    observations = pq.read_table(path).to_pandas()
    observations["result_date"] = pd.to_datetime(observations["result_date"]).dt.date
    observations = observations.loc[
        observations["result_date"].between(start_date, end_date)
    ].copy()
    if observations.empty:
        raise ValueError("country cache has no rows in the requested date range")
    selected = classification_implementations(implementations)
    missing = sorted(set(selected) - set(observations["implementation"]))
    if missing:
        raise ValueError(f"classification implementations are missing: {missing}")
    observations = observations.loc[
        observations["implementation"].isin(selected)
    ].copy()
    identity = ["result_date", "run_id", "implementation", "cycle_ordinal", "row_id"]
    duplicates = int(observations.duplicated(identity).sum())
    if duplicates:
        raise ValueError(f"country cache contains {duplicates} duplicate result rows")
    summary = pd.DataFrame({
        "result_rows": [len(observations)],
        "valid_wbgt_observations": [int(_valid(observations, "wbgt").sum())],
        "implementations": [observations["implementation"].nunique()],
        "cells": [observations["cell_id"].nunique()],
        "first_date": [observations["result_date"].min()],
        "last_date": [observations["result_date"].max()],
    })
    coverage = pd.concat([
        observations.assign(
            quantity=quantity,
            quantity_status=observations[f"{quantity}_status"],
            quantity_value_present=_valid(observations, quantity),
        )[["quantity", "implementation", "quantity_status", "quantity_value_present"]]
        for quantity in ("wbgt", "tg", "tnwb")
    ]).groupby(
        ["quantity", "implementation", "quantity_status", "quantity_value_present"],
        as_index=False, sort=True,
    ).size().rename(columns={"size": "n"})
    classification = analyze_wbgt_classification(
        observations.loc[_valid(observations, "wbgt")].copy(), thresholds_c,
        pair_keys=["result_date", "run_id", "cycle_ordinal", "row_id", "cell_id"],
        month_column="result_date",
        implementations=selected,
    )
    return Era5ClassificationAnalysis(classification, summary, coverage)


def _valid(observations: pd.DataFrame, quantity: str) -> pd.Series:
    values = pd.to_numeric(observations[f"{quantity}_c"], errors="coerce")
    return observations[f"{quantity}_status"].eq("ok") & np.isfinite(values)
