"""Annual pairwise aggregation from persisted daily summaries."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..io.artifacts import atomic_json as _atomic_json
from ..io.artifacts import atomic_parquet as _atomic_parquet
from ..results import RESULT_QUANTITIES
from .config import REFERENCE_IMPLEMENTATION, SampleScope
from .persistence import (
    DEFAULT_PAIRWISE_OUTPUT_ROOT,
    completed_pairwise_dates,
    pairwise_metadata,
    pairwise_product_root,
)
from .statistics import histogram_quantile_statistics, pooled_sufficient_statistics

QUANTITY_ORDER = ("wbgt", "tnwb", "tg")
STATISTIC_SPECS = {
    "n": {"signed": False},
    "mean_difference_c": {"signed": True},
    "stddev_difference_c": {"signed": False},
    "mean_absolute_difference_c": {"signed": False},
    "rmse_c": {"signed": False},
    "loa_lower_c": {"signed": True, "reverse": "loa_upper_c"},
    "loa_upper_c": {"signed": True, "reverse": "loa_lower_c"},
    "median_absolute_difference_c": {"signed": False},
    "p95_absolute_difference_c": {"signed": False},
    "p99_absolute_difference_c": {"signed": False},
    "max_absolute_difference_c": {"signed": False},
    "area_weighted_mean_difference_c": {"signed": True},
    "area_weighted_stddev_difference_c": {"signed": False},
    "area_weighted_mean_absolute_difference_c": {"signed": False},
    "area_weighted_rmse_c": {"signed": False},
    "area_weighted_loa_lower_c": {
        "signed": True,
        "reverse": "area_weighted_loa_upper_c",
    },
    "area_weighted_loa_upper_c": {
        "signed": True,
        "reverse": "area_weighted_loa_lower_c",
    },
    "area_weighted_median_absolute_difference_c": {"signed": False},
    "area_weighted_p95_absolute_difference_c": {"signed": False},
    "area_weighted_p99_absolute_difference_c": {"signed": False},
}


@dataclass(frozen=True)
class YearlyPairwiseOutputs:
    pairwise: Path
    quantity_coverage: Path
    reference_xlsx: Path
    metadata: Path


def analyze_yearly_pairwise(
    input_root: str | Path = DEFAULT_PAIRWISE_OUTPUT_ROOT,
    *,
    start_date: date,
    end_date: date,
    output_root: str | Path = Path("data/pairwise/yearly"),
    sample_scope: SampleScope = "all-implementations",
    reference_implementation: str = REFERENCE_IMPLEMENTATION,
    export_implementations: tuple[str, ...] | None = None,
) -> YearlyPairwiseOutputs:
    """Pool daily summaries and persist annual tables and reference workbook."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    product = pairwise_product_root(input_root, sample_scope)
    metadata = json.loads((product / "metadata.json").read_text())
    if metadata != pairwise_metadata(sample_scope):
        raise ValueError("Unexpected pairwise product metadata")
    dates = tuple(
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    )
    completed = completed_pairwise_dates(input_root, sample_scope)
    missing_dates = set(dates) - completed
    if missing_dates:
        raise ValueError(
            f"Incomplete pairwise dates: {[value.isoformat() for value in sorted(missing_dates)]}"
        )
    directories = tuple(product / "dates" / day.isoformat() for day in dates)
    daily = _load(directories, "pairwise")
    histogram = (
        _load(directories, "abs_histogram")
        .groupby(
            ["quantity", "implementation_a", "implementation_b", "abs_bin_id"],
            as_index=False,
            sort=True,
        )
        .agg(
            abs_bin_lower_c=("abs_bin_lower_c", "min"),
            abs_bin_upper_c=("abs_bin_upper_c", "min"),
            n=("n", "sum"),
            area_weight_sum=("area_weight_sum", "sum"),
        )
    )
    status = _load(directories, "quantity_status")
    implementations = tuple(sorted(status["implementation"].unique()))
    _validate_coverage(daily, status, dates, implementations)

    records = []
    keys = ["quantity", "implementation_a", "implementation_b"]
    for (quantity, implementation_a, implementation_b), pair in daily.groupby(
        keys, sort=True
    ):
        statistics = pooled_sufficient_statistics(
            pair["n"],
            sum_differences=pair["sum_difference_c"],
            sum_squared_differences=pair["sum_squared_difference_c2"],
            sum_absolute_differences=pair["sum_absolute_difference_c"],
            min_differences=pair["min_difference_c"],
            max_differences=pair["max_difference_c"],
            max_absolute_differences=pair["max_absolute_difference_c"],
            area_weight_sums=pair["area_weight_sum"],
            area_weighted_sum_differences=pair["area_weighted_difference_sum_c"],
            area_weighted_sum_squared_differences=pair[
                "area_weighted_squared_difference_sum_c2"
            ],
            area_weighted_sum_absolute_differences=pair[
                "area_weighted_absolute_difference_sum_c"
            ],
            exceedance_counts_0_1=pair["n_absolute_difference_gt_0_1_c"],
            exceedance_counts_0_2=pair["n_absolute_difference_gt_0_2_c"],
            exceedance_counts_0_5=pair["n_absolute_difference_gt_0_5_c"],
            exceedance_area_weights_0_1=pair[
                "area_weight_absolute_difference_gt_0_1_c"
            ],
            exceedance_area_weights_0_2=pair[
                "area_weight_absolute_difference_gt_0_2_c"
            ],
            exceedance_area_weights_0_5=pair[
                "area_weight_absolute_difference_gt_0_5_c"
            ],
        )
        bins = histogram.loc[
            histogram["quantity"].eq(quantity)
            & histogram["implementation_a"].eq(implementation_a)
            & histogram["implementation_b"].eq(implementation_b)
        ]
        quantiles = histogram_quantile_statistics(
            bins["abs_bin_id"],
            bin_lowers=bins["abs_bin_lower_c"],
            bin_uppers=bins["abs_bin_upper_c"],
            counts=bins["n"],
            area_weight_sums=bins["area_weight_sum"],
            expected_count=statistics["n"],
            expected_area_weight_sum=statistics["area_weight_sum"],
        )
        records.append(
            {
                "quantity": quantity,
                "implementation_a": implementation_a,
                "implementation_b": implementation_b,
                **statistics,
                **quantiles,
            }
        )
    yearly = pd.DataFrame.from_records(records)
    coverage = _quantity_coverage(status)
    if reference_implementation not in implementations:
        raise ValueError(
            f"Reference implementation is absent: {reference_implementation}"
        )
    exports = export_implementations or tuple(
        implementation
        for implementation in implementations
        if implementation != reference_implementation
    )
    missing = set(exports) - set(implementations)
    if missing:
        raise ValueError(f"Export implementations are absent: {sorted(missing)}")

    root = Path(output_root).expanduser().resolve()
    outputs = YearlyPairwiseOutputs(
        root / "yearly_pairwise.parquet",
        root / "quantity_coverage.parquet",
        root / f"pairwise_metrics_vs_{reference_implementation}.xlsx",
        root / "metadata.json",
    )
    _atomic_parquet(
        pa.Table.from_pandas(yearly, preserve_index=False), outputs.pairwise
    )
    _atomic_parquet(
        pa.Table.from_pandas(coverage, preserve_index=False), outputs.quantity_coverage
    )
    _write_reference_xlsx(
        yearly, reference_implementation, exports, outputs.reference_xlsx
    )
    _atomic_json(
        {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sample_scope": sample_scope,
            "reference_implementation": reference_implementation,
            "implementations": list(implementations),
            "quantities": list(QUANTITY_ORDER),
        },
        outputs.metadata,
    )
    return outputs


def pairwise_matrices(yearly: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """Build presentation matrices from an annual pairwise table."""

    implementations = tuple(
        sorted(set(yearly["implementation_a"]) | set(yearly["implementation_b"]))
    )
    result = {}
    for quantity in QUANTITY_ORDER:
        matrices = {
            statistic: pd.DataFrame(
                np.nan, index=implementations, columns=implementations
            )
            for statistic in STATISTIC_SPECS
        }
        for row in yearly.loc[yearly["quantity"].eq(quantity)].itertuples(index=False):
            for statistic, spec in STATISTIC_SPECS.items():
                value = getattr(row, statistic)
                matrices[statistic].loc[row.implementation_a, row.implementation_b] = (
                    value
                )
                reverse = getattr(row, spec.get("reverse", statistic))
                matrices[statistic].loc[row.implementation_b, row.implementation_a] = (
                    -reverse if spec["signed"] else reverse
                )
        result[quantity] = matrices
    return result


def _load(directories: tuple[Path, ...], name: str) -> pd.DataFrame:
    return pd.concat(
        [
            pq.read_table(directory / f"{name}.parquet").to_pandas()
            for directory in directories
        ],
        ignore_index=True,
    )


def _validate_coverage(
    daily: pd.DataFrame,
    status: pd.DataFrame,
    dates: tuple[date, ...],
    implementations: tuple[str, ...],
) -> None:
    if set(daily["quantity"].unique()) != set(RESULT_QUANTITIES):
        raise ValueError("Unexpected quantity inventory")
    pairs = set(combinations(implementations, 2))
    for quantity in RESULT_QUANTITIES:
        observed = set(
            daily.loc[
                daily["quantity"].eq(quantity),
                ["implementation_a", "implementation_b"],
            ].itertuples(index=False, name=None)
        )
        if observed != pairs:
            raise ValueError(f"Incomplete implementation pairs for {quantity}")
    expected_dates = len(dates)
    pair_dates = daily.groupby(["quantity", "implementation_a", "implementation_b"])[
        "result_date"
    ].nunique()
    if (
        len(pair_dates) != len(RESULT_QUANTITIES) * len(pairs)
        or not pair_dates.eq(expected_dates).all()
    ):
        raise ValueError("Incomplete daily pair coverage")
    status_dates = status.groupby(["quantity", "implementation"])[
        "result_date"
    ].nunique()
    if (
        len(status_dates) != len(RESULT_QUANTITIES) * len(implementations)
        or not status_dates.eq(expected_dates).all()
    ):
        raise ValueError("Incomplete implementation status coverage")


def _quantity_coverage(status: pd.DataFrame) -> pd.DataFrame:
    valid = status["status"].eq("ok") & status["value_present"]
    result = (
        status.assign(valid_n=status["n"].where(valid, 0))
        .groupby(["quantity", "implementation"], as_index=False, sort=True)
        .agg(total_n=("n", "sum"), valid_n=("valid_n", "sum"))
    )
    result["valid_percent"] = 100.0 * result["valid_n"] / result["total_n"]
    return result


def _write_reference_xlsx(
    yearly: pd.DataFrame,
    reference: str,
    implementations: tuple[str, ...],
    path: Path,
) -> None:
    matrices = pairwise_matrices(yearly)
    metrics = (
        "n",
        *(name for name in STATISTIC_SPECS if name.startswith("area_weighted_")),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".xlsx")
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
            for metric in metrics:
                comparison = pd.DataFrame(
                    {
                        quantity.upper(): matrices[quantity][metric][reference]
                        for quantity in QUANTITY_ORDER
                    }
                ).loc[list(implementations)]
                sheet = (
                    metric.replace("area_weighted_", "aw_")
                    .replace("absolute", "abs")
                    .replace("difference", "diff")
                )
                comparison.rename_axis("implementation").to_excel(
                    writer, sheet_name=sheet
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
