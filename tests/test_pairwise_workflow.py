from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from liljegren_analysis.comparison import run_daily_pairwise


def test_daily_workflow_returns_without_starting_workers_when_complete(
    tmp_path,
) -> None:
    result_date = date(2025, 1, 1)
    product = tmp_path / "product"
    with (
        patch("liljegren_analysis.comparison.workflows.HSBRunResults.from_hsb_run"),
        patch(
            "liljegren_analysis.comparison.workflows.initialize_pairwise_output",
            return_value=product,
        ),
        patch(
            "liljegren_analysis.comparison.workflows.completed_pairwise_dates",
            return_value=frozenset({result_date}),
        ),
        patch(
            "liljegren_analysis.comparison.workflows.ProcessPoolExecutor"
        ) as executor,
    ):
        output = run_daily_pairwise(
            Path("run"),
            start_date=result_date,
            end_date=result_date,
            output_root=tmp_path,
        )

    assert output == product
    executor.assert_not_called()


def test_daily_workflow_validates_cheap_options_before_io(tmp_path) -> None:
    with pytest.raises(ValueError, match="start_date"):
        run_daily_pairwise(
            Path("run"),
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 1),
            output_root=tmp_path,
        )
    with pytest.raises(ValueError, match="workers"):
        run_daily_pairwise(
            Path("run"),
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 1),
            output_root=tmp_path,
            workers=0,
        )
