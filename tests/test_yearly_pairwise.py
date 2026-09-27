from __future__ import annotations

from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.comparison.pairwise import DailyPairwiseSummaries, persist_daily_pairwise
from analysis.comparison.persistence import initialize_pairwise_output
from analysis.comparison.yearly import analyze_yearly_pairwise, pairwise_matrices


def _daily_tables(result_date: date) -> DailyPairwiseSummaries:
    pairwise = []
    histogram = []
    status = []
    for quantity in ("wbgt", "tg", "tnwb"):
        pairwise.append(
            {
                "result_date": result_date,
                "quantity": quantity,
                "implementation_a": "a",
                "implementation_b": "b",
                "n": 2,
                "sum_difference_c": 4.0,
                "sum_squared_difference_c2": 10.0,
                "sum_absolute_difference_c": 4.0,
                "min_difference_c": 1.0,
                "max_difference_c": 3.0,
                "max_absolute_difference_c": 3.0,
                "area_weight_sum": 2.0,
                "area_weighted_difference_sum_c": 4.0,
                "area_weighted_squared_difference_sum_c2": 10.0,
                "area_weighted_absolute_difference_sum_c": 4.0,
                "n_absolute_difference_gt_0_1_c": 2,
                "n_absolute_difference_gt_0_2_c": 2,
                "n_absolute_difference_gt_0_5_c": 2,
                "area_weight_absolute_difference_gt_0_1_c": 2.0,
                "area_weight_absolute_difference_gt_0_2_c": 2.0,
                "area_weight_absolute_difference_gt_0_5_c": 2.0,
            }
        )
        for bin_id, lower in ((50, 1.0), (150, 3.0)):
            histogram.append(
                {
                    "result_date": result_date,
                    "quantity": quantity,
                    "implementation_a": "a",
                    "implementation_b": "b",
                    "abs_bin_id": bin_id,
                    "abs_bin_lower_c": lower,
                    "abs_bin_upper_c": lower + 0.02,
                    "n": 1,
                    "area_weight_sum": 1.0,
                }
            )
        for implementation in ("a", "b"):
            status.append(
                {
                    "result_date": result_date,
                    "quantity": quantity,
                    "implementation": implementation,
                    "status": "ok",
                    "value_present": True,
                    "n": 2,
                    "area_weight_sum": 2.0,
                }
            )
    return DailyPairwiseSummaries(
        pa.Table.from_pylist(pairwise),
        pa.Table.from_pylist(histogram),
        pa.Table.from_pylist(status),
    )


def test_yearly_api_pools_and_persists_outputs(tmp_path) -> None:
    input_root = tmp_path / "daily"
    result_date = date(2025, 1, 1)
    initialize_pairwise_output(
        input_root,
        "pairwise",
        implementations=("a", "b"),
    )
    persist_daily_pairwise(input_root, result_date, _daily_tables(result_date))

    outputs = analyze_yearly_pairwise(
        input_root,
        start_date=result_date,
        end_date=result_date,
        output_root=tmp_path / "yearly",
        reference_implementation="a",
        export_implementations=("b",),
    )

    yearly = pq.read_table((outputs / "yearly_pairwise.parquet")).to_pandas()
    assert yearly["mean_difference_c"].tolist() == [2.0, 2.0, 2.0]
    assert pq.read_table((outputs / "quantity_coverage.parquet")).num_rows == 6
    assert (outputs / "pairwise_metrics_vs_a.xlsx").is_file()
    assert (outputs / "metadata.json").is_file()
    matrices = pairwise_matrices(yearly)
    assert matrices["wbgt"]["mean_difference_c"].loc["a", "b"] == 2.0
    assert matrices["wbgt"]["mean_difference_c"].loc["b", "a"] == -2.0


def test_yearly_api_requires_complete_date_coverage(tmp_path) -> None:
    input_root = tmp_path / "daily"
    result_date = date(2025, 1, 1)
    initialize_pairwise_output(
        input_root,
        "pairwise",
        implementations=("a", "b"),
    )
    persist_daily_pairwise(input_root, result_date, _daily_tables(result_date))

    try:
        analyze_yearly_pairwise(
            input_root,
            start_date=result_date,
            end_date=date(2025, 1, 2),
            output_root=tmp_path / "yearly",
            reference_implementation="a",
        )
    except ValueError as error:
        assert "Incomplete pairwise dates" in str(error)
    else:
        raise AssertionError("missing daily partition was accepted")
