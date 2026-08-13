"""Versioned fixed-bin summaries for pairwise WBGT differences."""

from __future__ import annotations


ABS_HISTOGRAM_VERSION = 1

ABS_DIFFERENCE_BIN_SQL = """
CASE
    WHEN abs_difference_c < 2.0
        THEN floor(abs_difference_c / 0.01)::INTEGER
    WHEN abs_difference_c < 5.0
        THEN 200 + floor((abs_difference_c - 2.0) / 0.05)::INTEGER
    WHEN abs_difference_c < 20.0
        THEN 260 + floor((abs_difference_c - 5.0) / 0.25)::INTEGER
    WHEN abs_difference_c < 100.0
        THEN 320 + floor((abs_difference_c - 20.0) / 1.0)::INTEGER
    ELSE 400
END
""".strip()

def absolute_difference_bin_bounds(bin_id: int) -> tuple[float, float | None]:
    """Return lower-inclusive and upper-exclusive absolute-difference bounds."""

    if 0 <= bin_id < 200:
        return bin_id * 0.01, (bin_id + 1) * 0.01
    if 200 <= bin_id < 260:
        offset = bin_id - 200
        return 2.0 + offset * 0.05, 2.0 + (offset + 1) * 0.05
    if 260 <= bin_id < 320:
        offset = bin_id - 260
        return 5.0 + offset * 0.25, 5.0 + (offset + 1) * 0.25
    if 320 <= bin_id < 400:
        offset = bin_id - 320
        return 20.0 + offset, 21.0 + offset
    if bin_id == 400:
        return 100.0, None
    raise ValueError(f"Invalid absolute-difference bin ID: {bin_id}")


def absolute_difference_bounds_sql(bin_expression: str) -> tuple[str, str]:
    """Return SQL expressions for absolute-difference bin bounds."""

    lower = f"""
    CASE
        WHEN {bin_expression} < 200 THEN {bin_expression} * 0.01
        WHEN {bin_expression} < 260
            THEN 2.0 + ({bin_expression} - 200) * 0.05
        WHEN {bin_expression} < 320
            THEN 5.0 + ({bin_expression} - 260) * 0.25
        WHEN {bin_expression} < 400
            THEN 20.0 + ({bin_expression} - 320) * 1.0
        ELSE 100.0
    END
    """.strip()
    upper = f"""
    CASE
        WHEN {bin_expression} < 200 THEN ({bin_expression} + 1) * 0.01
        WHEN {bin_expression} < 260
            THEN 2.0 + ({bin_expression} - 199) * 0.05
        WHEN {bin_expression} < 320
            THEN 5.0 + ({bin_expression} - 259) * 0.25
        WHEN {bin_expression} < 400
            THEN 20.0 + ({bin_expression} - 319) * 1.0
        ELSE NULL
    END
    """.strip()
    return f"CAST(({lower}) AS DOUBLE)", f"CAST(({upper}) AS DOUBLE)"
