"""Build the single-pass daily pairwise summary query."""

from __future__ import annotations

from itertools import combinations
from typing import Literal

from .distributions import (
    ABS_DIFFERENCE_BIN_SQL,
    absolute_difference_bounds_sql,
)


SampleScope = Literal["pairwise", "all-implementations"]
SAMPLE_SCOPES: tuple[SampleScope, ...] = (
    "pairwise",
    "all-implementations",
)
IMPLEMENTATION_SET_VERSION = 1
EXPECTED_IMPLEMENTATIONS = (
    "heatstress",
    "heatstress-jl",
    "heatstress-r",
    "liljegren-c",
    "meteor",
    "pywbgt",
    "thermofeel",
)
_IMPLEMENTATION_COLUMNS = tuple(
    (implementation, implementation.replace("-", "_"))
    for implementation in EXPECTED_IMPLEMENTATIONS
)


_PAIRWISE_PAIR_CTES = """
sampled_results AS (
    SELECT *
    FROM valid_results
),
paired_values AS (
    SELECT
        left_result.result_date,
        left_result.implementation AS implementation_a,
        right_result.implementation AS implementation_b,
        left_result.row_id,
        left_result.value_c AS value_a_c,
        right_result.value_c AS value_b_c,
        left_result.value_c - right_result.value_c AS difference_c
    FROM sampled_results AS left_result
    INNER JOIN sampled_results AS right_result
        ON left_result.result_date = right_result.result_date
       AND left_result.run_id = right_result.run_id
       AND left_result.cycle_ordinal = right_result.cycle_ordinal
       AND left_result.row_id = right_result.row_id
       AND left_result.implementation < right_result.implementation
)"""


def _all_implementations_pair_ctes() -> str:
    implementation_literals = ", ".join(
        f"'{implementation}'" for implementation in EXPECTED_IMPLEMENTATIONS
    )
    value_columns = ",\n        ".join(
        "max(value_c) FILTER "
        f"(WHERE implementation = '{implementation}') AS {column}"
        for implementation, column in _IMPLEMENTATION_COLUMNS
    )
    completeness_checks = "\n       AND ".join(
        "count(*) FILTER "
        f"(WHERE implementation = '{implementation}') = 1"
        for implementation in EXPECTED_IMPLEMENTATIONS
    )
    pairs = tuple(combinations(_IMPLEMENTATION_COLUMNS, 2))
    implementations_a = ", ".join(
        f"'{implementation_a}'"
        for (implementation_a, _), _ in pairs
    )
    implementations_b = ", ".join(
        f"'{implementation_b}'"
        for _, (implementation_b, _) in pairs
    )
    values_a = ", ".join(
        f"complete.{column_a}" for (_, column_a), _ in pairs
    )
    values_b = ", ".join(
        f"complete.{column_b}" for _, (_, column_b) in pairs
    )
    return f"""
complete_observations AS (
    SELECT
        result_date,
        run_id,
        cycle_ordinal,
        row_id,
        {value_columns}
    FROM valid_results
    WHERE implementation IN ({implementation_literals})
    GROUP BY result_date, run_id, cycle_ordinal, row_id
    HAVING {completeness_checks}
),
paired_values AS (
    SELECT
        complete.result_date,
        unnest([{implementations_a}]) AS implementation_a,
        unnest([{implementations_b}]) AS implementation_b,
        complete.row_id,
        unnest([{values_a}]) AS value_a_c,
        unnest([{values_b}]) AS value_b_c,
        value_a_c - value_b_c AS difference_c
    FROM complete_observations AS complete
)"""


def build_daily_pairwise_sql(sample_scope: SampleScope) -> str:
    """Return one query producing pair totals and histogram rows."""

    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    pair_sql = (
        _PAIRWISE_PAIR_CTES
        if sample_scope == "pairwise"
        else _all_implementations_pair_ctes()
    )
    abs_lower, abs_upper = absolute_difference_bounds_sql("abs_bin_id")

    return f"""
WITH valid_results AS (
    SELECT
        result_date,
        run_id,
        implementation,
        cycle_ordinal,
        row_id,
        value_c
    FROM results
    WHERE status = 'ok'
      AND value_c IS NOT NULL
),
{pair_sql},
weighted_values AS (
    SELECT
        paired.*,
        abs(paired.difference_c) AS abs_difference_c,
        CASE
            WHEN regexp_full_match(
                paired.row_id,
                '^era5-global-[ns][0-9]{{6}}-.+$'
            ) AND latitude.area_weight IS NOT NULL
            THEN latitude.area_weight
            ELSE error('Invalid ERA5 row_id: ' || paired.row_id)
        END AS area_weight
    FROM paired_values AS paired
    LEFT JOIN _era5_latitude_weights AS latitude
        ON substring(paired.row_id, 13, 7) = latitude.latitude_code
),
binned_values AS (
    SELECT
        *,
        {ABS_DIFFERENCE_BIN_SQL} AS abs_bin_id
    FROM weighted_values
),
summaries AS (
    SELECT
        result_date,
        implementation_a,
        implementation_b,
        abs_bin_id,
        grouping(abs_bin_id) AS abs_bin_grouped,
        count(*) AS n,
        fsum(difference_c) AS sum_difference_c,
        fsum(difference_c * difference_c) AS sum_squared_difference_c2,
        fsum(abs_difference_c) AS sum_absolute_difference_c,
        min(difference_c) AS min_difference_c,
        max(difference_c) AS max_difference_c,
        max(abs_difference_c) AS max_absolute_difference_c,
        fsum(area_weight) AS area_weight_sum,
        fsum(area_weight * difference_c)
            AS area_weighted_difference_sum_c,
        fsum(area_weight * difference_c * difference_c)
            AS area_weighted_squared_difference_sum_c2,
        fsum(area_weight * abs_difference_c)
            AS area_weighted_absolute_difference_sum_c,
        count(*) FILTER (WHERE abs_difference_c > 0.1)
            AS n_absolute_difference_gt_0_1_c,
        count(*) FILTER (WHERE abs_difference_c > 0.2)
            AS n_absolute_difference_gt_0_2_c,
        count(*) FILTER (WHERE abs_difference_c > 0.5)
            AS n_absolute_difference_gt_0_5_c,
        fsum(area_weight) FILTER (WHERE abs_difference_c > 0.1)
            AS area_weight_absolute_difference_gt_0_1_c,
        fsum(area_weight) FILTER (WHERE abs_difference_c > 0.2)
            AS area_weight_absolute_difference_gt_0_2_c,
        fsum(area_weight) FILTER (WHERE abs_difference_c > 0.5)
            AS area_weight_absolute_difference_gt_0_5_c
    FROM binned_values
    GROUP BY GROUPING SETS (
        (result_date, implementation_a, implementation_b),
        (result_date, implementation_a, implementation_b, abs_bin_id)
    )
)
SELECT
    CASE
        WHEN abs_bin_grouped = 0 THEN 'abs_histogram'
        ELSE 'pairwise'
    END AS summary_kind,
    result_date,
    implementation_a,
    implementation_b,
    abs_bin_id,
    CASE WHEN abs_bin_grouped = 0 THEN {abs_lower} END AS abs_bin_lower_c,
    CASE WHEN abs_bin_grouped = 0 THEN {abs_upper} END AS abs_bin_upper_c,
    n,
    sum_difference_c,
    sum_squared_difference_c2,
    sum_absolute_difference_c,
    min_difference_c,
    max_difference_c,
    max_absolute_difference_c,
    area_weight_sum,
    area_weighted_difference_sum_c,
    area_weighted_squared_difference_sum_c2,
    area_weighted_absolute_difference_sum_c,
    n_absolute_difference_gt_0_1_c,
    n_absolute_difference_gt_0_2_c,
    n_absolute_difference_gt_0_5_c,
    coalesce(area_weight_absolute_difference_gt_0_1_c, 0.0)
        AS area_weight_absolute_difference_gt_0_1_c,
    coalesce(area_weight_absolute_difference_gt_0_2_c, 0.0)
        AS area_weight_absolute_difference_gt_0_2_c,
    coalesce(area_weight_absolute_difference_gt_0_5_c, 0.0)
        AS area_weight_absolute_difference_gt_0_5_c
FROM summaries
ORDER BY
    summary_kind,
    implementation_a,
    implementation_b,
    abs_bin_id
"""
