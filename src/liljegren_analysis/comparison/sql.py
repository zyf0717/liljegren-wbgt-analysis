"""Build single-scan daily pairwise summaries for all result quantities."""

from __future__ import annotations

from itertools import combinations
from typing import Literal

from ..results import RESULT_QUANTITIES
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
QUANTITY_SET_VERSION = 1
QUANTITY_SCOPE = "joint"
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


def _pairwise_pair_ctes() -> str:
    differences = ",\n        ".join(
        f"left_result.{quantity}_c - right_result.{quantity}_c "
        f"AS {quantity}_difference_c"
        for quantity in RESULT_QUANTITIES
    )
    quantities = ", ".join(f"'{quantity}'" for quantity in RESULT_QUANTITIES)
    difference_columns = ", ".join(
        f"{quantity}_difference_c" for quantity in RESULT_QUANTITIES
    )
    return f"""
paired_wide AS (
    SELECT
        left_result.result_date,
        left_result.implementation AS implementation_a,
        right_result.implementation AS implementation_b,
        left_result.row_id,
        left_result.area_weight,
        {differences}
    FROM joint_valid_results AS left_result
    INNER JOIN joint_valid_results AS right_result
        ON left_result.result_date = right_result.result_date
       AND left_result.run_id = right_result.run_id
       AND left_result.cycle_ordinal = right_result.cycle_ordinal
       AND left_result.row_id = right_result.row_id
       AND left_result.implementation < right_result.implementation
),
paired_values AS (
    SELECT
        result_date,
        implementation_a,
        implementation_b,
        row_id,
        area_weight,
        unnest([{quantities}]) AS quantity,
        unnest([{difference_columns}]) AS difference_c
    FROM paired_wide
)"""


def _all_implementations_pair_ctes() -> str:
    implementation_literals = ", ".join(
        f"'{implementation}'" for implementation in EXPECTED_IMPLEMENTATIONS
    )
    value_columns = ",\n        ".join(
        f"max({quantity}_c) FILTER "
        f"(WHERE implementation = '{implementation}') "
        f"AS {column}_{quantity}"
        for quantity in RESULT_QUANTITIES
        for implementation, column in _IMPLEMENTATION_COLUMNS
    )
    completeness_checks = "\n       AND ".join(
        "count(*) FILTER "
        f"(WHERE implementation = '{implementation}') = 1"
        for implementation in EXPECTED_IMPLEMENTATIONS
    )
    expanded = tuple(
        (quantity, implementation_a, column_a, implementation_b, column_b)
        for quantity in RESULT_QUANTITIES
        for (implementation_a, column_a), (implementation_b, column_b)
        in combinations(_IMPLEMENTATION_COLUMNS, 2)
    )
    quantities = ", ".join(f"'{quantity}'" for quantity, *_ in expanded)
    implementations_a = ", ".join(f"'{item[1]}'" for item in expanded)
    implementations_b = ", ".join(f"'{item[3]}'" for item in expanded)
    differences = ", ".join(
        f"complete.{column_a}_{quantity} - complete.{column_b}_{quantity}"
        for quantity, _, column_a, _, column_b in expanded
    )
    return f"""
complete_observations AS (
    SELECT
        result_date,
        run_id,
        cycle_ordinal,
        row_id,
        max(area_weight) AS area_weight,
        {value_columns}
    FROM joint_valid_results
    WHERE implementation IN ({implementation_literals})
    GROUP BY result_date, run_id, cycle_ordinal, row_id
    HAVING {completeness_checks}
),
paired_values AS (
    SELECT
        complete.result_date,
        unnest([{quantities}]) AS quantity,
        unnest([{implementations_a}]) AS implementation_a,
        unnest([{implementations_b}]) AS implementation_b,
        complete.row_id,
        complete.area_weight,
        unnest([{differences}]) AS difference_c
    FROM complete_observations AS complete
)"""


def build_daily_pairwise_sql(sample_scope: SampleScope) -> str:
    """Return one source-scan query for agreement and status summaries."""

    if sample_scope not in SAMPLE_SCOPES:
        raise ValueError(f"Unsupported sample scope: {sample_scope}")
    pair_sql = (
        _pairwise_pair_ctes()
        if sample_scope == "pairwise"
        else _all_implementations_pair_ctes()
    )
    abs_lower, abs_upper = absolute_difference_bounds_sql("abs_bin_id")
    quantity_literals = ", ".join(
        f"'{quantity}'" for quantity in RESULT_QUANTITIES
    )
    status_columns = ", ".join(
        f"{quantity}_status" for quantity in RESULT_QUANTITIES
    )
    value_columns = ", ".join(
        f"{quantity}_c" for quantity in RESULT_QUANTITIES
    )
    joint_validity = "\n      AND ".join(
        f"{quantity}_status = 'ok' AND {quantity}_c IS NOT NULL"
        for quantity in RESULT_QUANTITIES
    )

    return f"""
WITH base AS MATERIALIZED (
    SELECT
        source.result_date,
        source.run_id,
        source.implementation,
        source.cycle_ordinal,
        source.row_id,
        source.row_ordinal,
        source.wbgt_status,
        source.wbgt_c,
        source.tg_status,
        source.tg_c,
        source.tnwb_status,
        source.tnwb_c,
        CASE
            WHEN regexp_full_match(
                source.row_id,
                '^era5-global-[ns][0-9]{{6}}-.+$'
            ) AND latitude.area_weight IS NOT NULL
            THEN latitude.area_weight
            ELSE error('Invalid ERA5 row_id: ' || source.row_id)
        END AS area_weight
    FROM results AS source
    LEFT JOIN _era5_latitude_weights AS latitude
        ON substring(source.row_id, 13, 7) = latitude.latitude_code
),
quantity_status_values AS (
    SELECT
        result_date,
        implementation,
        area_weight,
        unnest([{quantity_literals}]) AS quantity,
        unnest([{status_columns}]) AS status,
        unnest([{value_columns}]) IS NOT NULL AS value_present
    FROM base
),
quantity_status AS (
    SELECT
        result_date,
        quantity,
        implementation,
        status,
        value_present,
        count(*)::BIGINT AS n,
        fsum(area_weight) AS area_weight_sum
    FROM quantity_status_values
    GROUP BY result_date, quantity, implementation, status, value_present
),
joint_valid_results AS (
    SELECT *
    FROM base
    WHERE {joint_validity}
),
{pair_sql},
weighted_values AS (
    SELECT
        paired.*,
        abs(paired.difference_c) AS abs_difference_c
    FROM paired_values AS paired
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
        quantity,
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
        (result_date, quantity, implementation_a, implementation_b),
        (result_date, quantity, implementation_a, implementation_b, abs_bin_id)
    )
),
agreement_output AS (
    SELECT
        CASE
            WHEN abs_bin_grouped = 0 THEN 'abs_histogram'
            ELSE 'pairwise'
        END AS summary_kind,
        result_date,
        quantity,
        NULL::VARCHAR AS implementation,
        implementation_a,
        implementation_b,
        NULL::VARCHAR AS status,
        NULL::BOOLEAN AS value_present,
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
),
status_output AS (
    SELECT
        'quantity_status' AS summary_kind,
        result_date,
        quantity,
        implementation,
        NULL::VARCHAR AS implementation_a,
        NULL::VARCHAR AS implementation_b,
        status,
        value_present,
        NULL::INTEGER AS abs_bin_id,
        NULL::DOUBLE AS abs_bin_lower_c,
        NULL::DOUBLE AS abs_bin_upper_c,
        n,
        NULL::DOUBLE AS sum_difference_c,
        NULL::DOUBLE AS sum_squared_difference_c2,
        NULL::DOUBLE AS sum_absolute_difference_c,
        NULL::DOUBLE AS min_difference_c,
        NULL::DOUBLE AS max_difference_c,
        NULL::DOUBLE AS max_absolute_difference_c,
        area_weight_sum,
        NULL::DOUBLE AS area_weighted_difference_sum_c,
        NULL::DOUBLE AS area_weighted_squared_difference_sum_c2,
        NULL::DOUBLE AS area_weighted_absolute_difference_sum_c,
        NULL::BIGINT AS n_absolute_difference_gt_0_1_c,
        NULL::BIGINT AS n_absolute_difference_gt_0_2_c,
        NULL::BIGINT AS n_absolute_difference_gt_0_5_c,
        NULL::DOUBLE AS area_weight_absolute_difference_gt_0_1_c,
        NULL::DOUBLE AS area_weight_absolute_difference_gt_0_2_c,
        NULL::DOUBLE AS area_weight_absolute_difference_gt_0_5_c
    FROM quantity_status
)
SELECT * FROM agreement_output
UNION ALL
SELECT * FROM status_output
ORDER BY
    summary_kind,
    quantity,
    implementation_a,
    implementation_b,
    status,
    value_present,
    abs_bin_id
"""
