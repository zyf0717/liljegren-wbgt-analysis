"""Annual pairwise aggregation from persisted daily summaries."""

from __future__ import annotations

import json
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import REFERENCE_IMPLEMENTATION
from ..io.artifacts import atomic_json, atomic_parquet, atomic_path
from ..results import RESULT_QUANTITIES, date_range
from .persistence import (
    completed_pairwise_dates,
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


def analyze_yearly_pairwise(
    input_root: str | Path,
    *,
    start_date: date,
    end_date: date,
    output_root: str | Path,
    reference_implementation: str = REFERENCE_IMPLEMENTATION,
    export_implementations: tuple[str, ...] | None = None,
) -> Path:
    """Pool daily summaries and persist annual tables and comparison workbooks."""

    product = Path(input_root).expanduser().resolve()
    metadata = json.loads((product / "metadata.json").read_text())
    implementations = tuple(sorted(metadata["implementations"]))
    dates = date_range(start_date, end_date)
    completed = completed_pairwise_dates(input_root)
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
    _validate_coverage(daily, status, dates, implementations)

    records = []
    keys = ["quantity", "implementation_a", "implementation_b"]
    for (quantity, implementation_a, implementation_b), pair in daily.groupby(
        keys, sort=True
    ):
        statistics = pooled_sufficient_statistics(pair)
        bins = histogram.loc[
            histogram["quantity"].eq(quantity)
            & histogram["implementation_a"].eq(implementation_a)
            & histogram["implementation_b"].eq(implementation_b)
        ]
        quantiles = histogram_quantile_statistics(bins)
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
    atomic_parquet(
        pa.Table.from_pandas(yearly, preserve_index=False),
        (root / "yearly_pairwise.parquet"),
    )
    atomic_parquet(
        pa.Table.from_pandas(coverage, preserve_index=False),
        (root / "quantity_coverage.parquet"),
    )
    _write_reference_xlsx(
        yearly,
        reference_implementation,
        exports,
        (root / f"pairwise_metrics_vs_{reference_implementation}.xlsx"),
    )
    _write_area_weighted_matrices_xlsx(
        yearly,
        (reference_implementation, *exports),
        root / "pairwise_area_weighted_matrices.xlsx",
    )
    atomic_json(
        {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sample_scope": metadata["sample_scope"],
            "reference_implementation": reference_implementation,
            "implementations": list(implementations),
            "quantities": list(QUANTITY_ORDER),
        },
        (root / "metadata.json"),
    )
    return root


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


def _write_area_weighted_matrices_xlsx(
    yearly: pd.DataFrame,
    implementations: tuple[str, ...],
    path: Path,
) -> None:
    """Export the notebook's area-weighted matrices, preserving blanks and signs."""
    from openpyxl.comments import Comment
    from openpyxl.utils import get_column_letter

    matrices = pairwise_matrices(yearly)
    metrics = {
        "mean_diff": "area_weighted_mean_difference_c",
        "mae": "area_weighted_mean_absolute_difference_c",
        "rmse": "area_weighted_rmse_c",
    }
    with (
        atomic_path(path) as temporary_path,
        pd.ExcelWriter(temporary_path, engine="openpyxl") as writer,
    ):
        for quantity in QUANTITY_ORDER:
            for name, metric in metrics.items():
                matrix = matrices[quantity][metric].reindex(
                    index=implementations, columns=implementations,
                )
                sheet = f"{quantity}_aw_{name}_c"
                matrix.rename_axis("implementation_a").to_excel(
                    writer, sheet_name=sheet, freeze_panes=(1, 1),
                )
                worksheet = writer.sheets[sheet]
                worksheet["A1"].comment = Comment(
                    f"{quantity.upper()}: {metric}. Units: degrees Celsius. "
                    "Rows are implementation_a; columns are implementation_b. "
                    "Signed mean differences are row minus column. "
                    "Diagonal and unavailable comparisons are blank.",
                    "liljegren-wbgt-analysis",
                )
                worksheet.column_dimensions["A"].width = 27
                for column in range(2, len(implementations) + 2):
                    worksheet.column_dimensions[get_column_letter(column)].width = 24
                for row in worksheet.iter_rows(min_row=2, min_col=2):
                    for cell in row:
                        cell.number_format = "0.00"


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
    with (
        atomic_path(path) as temporary_path,
        pd.ExcelWriter(temporary_path, engine="openpyxl") as writer,
    ):
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
            comparison.rename_axis("implementation").to_excel(writer, sheet_name=sheet)
