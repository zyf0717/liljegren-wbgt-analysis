"""NASA POWER workload WBGT classification analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from shapely.geometry import Point

from ..country.cells import cached_natural_earth_boundaries, load_countries
from .core import (
    ClassificationAnalysis,
    analyze_wbgt_classification,
    classification_implementations,
)


@dataclass(frozen=True)
class NasaPowerClassificationAnalysis:
    """NASA POWER classifications with workload and status diagnostics."""

    classification: ClassificationAnalysis
    workload_summary: pd.DataFrame
    status_summary: pd.DataFrame

    @property
    def outputs(self) -> dict[str, pd.DataFrame]:
        outputs = {
            "workload_summary.csv": self.workload_summary,
            "wbgt_status_summary.csv": self.status_summary,
            **self.classification.outputs,
        }
        return {
            filename.replace("location_id_", "location_").replace(
                "country_iso_a3_", "country_"
            ): frame
            for filename, frame in outputs.items()
        }


def analyze_nasa_power_run(
    run_directory: str | Path,
    start_date: date,
    end_date: date,
    thresholds_c: Sequence[float],
    *,
    shard_ordinal: int = 1,
    cycle_ordinal: int = 1,
    country_codes: Sequence[str] | None = None,
    location_ids: Sequence[str] | None = None,
    boundary_cache: str | Path | None = None,
    implementations: Sequence[str] | None = None,
) -> NasaPowerClassificationAnalysis:
    """Analyze selected NASA POWER workload observations from one HSB run."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if shard_ordinal < 1 or cycle_ordinal < 1:
        raise ValueError("shard_ordinal and cycle_ordinal must be positive")
    root = Path(run_directory).expanduser().resolve()
    workload_path = root / "dataset" / "workload.csv"
    if not workload_path.is_file():
        raise FileNotFoundError(f"NASA POWER workload does not exist: {workload_path}")

    workload = _load_workload(workload_path)
    workload, has_country_scope = _filter_workload(
        workload,
        start_date,
        end_date,
        country_codes=country_codes,
        location_ids=location_ids,
        boundary_cache=boundary_cache,
    )
    selected = classification_implementations(implementations)
    result_paths = _result_paths(
        root / "results", shard_ordinal, workload, selected
    )
    results = _load_results(result_paths, workload, cycle_ordinal)
    status_summary = (
        results.groupby(["implementation", "wbgt_status"], as_index=False, sort=True)
        .size()
        .rename(columns={"size": "n"})
    )
    valid = results["wbgt_status"].eq("ok") & np.isfinite(
        pd.to_numeric(results["wbgt_c"], errors="coerce")
    )
    observations = (
        results.loc[valid]
        .rename(columns={"row_id": "result_row_id"})
        .merge(
            workload,
            on=["result_partition", "row_ordinal"],
            how="inner",
            validate="many_to_one",
        )
    )
    if observations.empty:
        raise ValueError("no successful observations remain after filtering")

    pair_keys = [
        "result_partition",
        "cycle_ordinal",
        "row_ordinal",
        "result_row_id",
        "source_row_id",
        "location_id",
        "interval_start_utc",
        "longitude_deg_east",
        "latitude_deg_north",
    ]
    dimensions = ["location_id"]
    if has_country_scope:
        pair_keys.extend(["country_iso_a2", "country_iso_a3", "country_name"])
        dimensions.append("country_iso_a3")
    classification = analyze_wbgt_classification(
        observations,
        thresholds_c,
        pair_keys=pair_keys,
        month_column="interval_start_utc",
        summary_dimensions=dimensions,
        implementations=selected,
    )
    workload_summary = pd.DataFrame(
        {
            "selected_locations": [workload["location_id"].nunique()],
            "workload_rows": [len(workload)],
            "first_timestamp": [workload["interval_start_utc"].min()],
            "last_timestamp": [workload["interval_start_utc"].max()],
        }
    )
    return NasaPowerClassificationAnalysis(classification, workload_summary, status_summary)


def _load_workload(path: Path) -> pd.DataFrame:
    columns = [
        "row_id",
        "location_id",
        "interval_start_utc",
        "longitude_deg_east",
        "latitude_deg_north",
    ]
    workload = pd.read_csv(path, usecols=columns, parse_dates=["interval_start_utc"])
    workload = workload.rename(columns={"row_id": "source_row_id"})
    workload.insert(0, "result_partition", workload["interval_start_utc"].dt.year)
    workload.insert(
        1,
        "row_ordinal",
        workload.groupby("result_partition", sort=False).cumcount() + 1,
    )
    if workload["source_row_id"].duplicated().any():
        raise ValueError("workload.csv contains duplicate row_id values")
    return workload


def _filter_workload(
    workload: pd.DataFrame,
    start_date: date,
    end_date: date,
    *,
    country_codes: Sequence[str] | None,
    location_ids: Sequence[str] | None,
    boundary_cache: str | Path | None,
) -> tuple[pd.DataFrame, bool]:
    filtered = workload.copy()
    has_country_scope = country_codes is not None
    if country_codes is not None:
        requested = _unique_codes(country_codes)
        if not requested:
            raise ValueError("country_codes must be None or contain at least one code")
        if boundary_cache is None:
            raise ValueError("boundary_cache is required when filtering by country")
        boundaries = cached_natural_earth_boundaries(boundary_cache)
        countries = load_countries(boundaries, requested)
        coordinates = filtered[
            ["location_id", "longitude_deg_east", "latitude_deg_north"]
        ].drop_duplicates()
        if coordinates["location_id"].duplicated(keep=False).any():
            raise ValueError("locations have multiple coordinates")
        records = []
        for row in coordinates.itertuples(index=False):
            matches = [
                country
                for country in countries
                if country.geometry.covers(Point(row.longitude_deg_east, row.latitude_deg_north))
            ]
            if len(matches) > 1:
                raise ValueError(f"location {row.location_id!r} matches multiple countries")
            if matches:
                country = matches[0]
                records.append(
                    {
                        "location_id": row.location_id,
                        "country_iso_a2": country.iso_a2,
                        "country_iso_a3": country.iso_a3,
                        "country_name": country.name,
                    }
                )
        filtered = filtered.merge(
            pd.DataFrame(
                records,
                columns=[
                    "location_id",
                    "country_iso_a2",
                    "country_iso_a3",
                    "country_name",
                ],
            ),
            on="location_id",
            how="inner",
            validate="many_to_one",
        )
    if location_ids is not None:
        selected = {value.strip().lower() for value in location_ids}
        available = set(filtered["location_id"].str.lower())
        unknown = sorted(selected - available)
        if unknown:
            raise ValueError(f"unknown location_ids: {unknown}")
        filtered = filtered.loc[filtered["location_id"].str.lower().isin(selected)].copy()
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    filtered = filtered.loc[
        filtered["interval_start_utc"].ge(start) & filtered["interval_start_utc"].lt(end)
    ].copy()
    if filtered.empty:
        raise ValueError("no workload rows match the requested filters")
    return filtered, has_country_scope


def _result_paths(
    results_directory: Path,
    shard_ordinal: int,
    workload: pd.DataFrame,
    implementations: Sequence[str],
) -> dict[tuple[int, str], Path]:
    selected_partitions = set(workload["result_partition"].astype(int))
    paths = {
        (int(path.parents[2].name), path.parent.name): path
        for path in results_directory.glob(f"*/{shard_ordinal:04d}/*/results.parquet")
        if path.parents[2].name.isdigit()
        and int(path.parents[2].name) in selected_partitions
        and path.parent.name in implementations
    }
    missing_implementations = sorted(set(implementations) - {
        implementation for _, implementation in paths
    })
    if missing_implementations:
        raise FileNotFoundError(
            f"classification implementations are missing: {missing_implementations}"
        )
    missing = {
        (partition, implementation)
        for partition in selected_partitions
        for implementation in implementations
    } - set(paths)
    if missing:
        raise FileNotFoundError(f"missing partition/implementation results: {sorted(missing)}")
    return paths


def _load_results(
    paths: dict[tuple[int, str], Path], workload: pd.DataFrame, cycle_ordinal: int
) -> pd.DataFrame:
    selected_ordinals = {
        int(partition): set(frame["row_ordinal"])
        for partition, frame in workload.groupby("result_partition", sort=False)
    }
    frames = []
    columns = ["cycle_ordinal", "row_id", "row_ordinal", "wbgt_status", "wbgt_c"]
    for (partition, implementation), path in paths.items():
        result = pd.read_parquet(path, columns=columns)
        result = result.loc[
            result["cycle_ordinal"].eq(cycle_ordinal)
            & result["row_ordinal"].isin(selected_ordinals[partition])
        ].copy()
        if result["row_ordinal"].duplicated().any():
            raise ValueError(f"{path} contains duplicate row_ordinal values")
        result.insert(0, "implementation", implementation)
        result.insert(0, "result_partition", partition)
        frames.append(result)
    results = pd.concat(frames, ignore_index=True)
    row_ids = results.groupby(["result_partition", "cycle_ordinal", "row_ordinal"])["row_id"].nunique()
    if row_ids.gt(1).any():
        raise ValueError("implementations disagree on result row_id")
    return results


def _unique_codes(codes: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(code.strip().upper() for code in codes if code.strip()))
