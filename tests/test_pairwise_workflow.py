from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from sample_settings import DEFAULT_PAIRWISE_IMPLEMENTATIONS

from analysis.comparison.workflows import run_daily_pairwise


def test_daily_workflow_returns_without_starting_workers_when_complete(
    tmp_path,
) -> None:
    result_date = date(2025, 1, 1)
    product = tmp_path / "product"
    with (
        patch(
            "analysis.comparison.workflows.initialize_pairwise_output",
            return_value=product,
        ),
        patch(
            "analysis.comparison.workflows.completed_pairwise_dates",
            return_value=frozenset({result_date}),
        ),
        patch("analysis.comparison.workflows.ProcessPoolExecutor") as executor,
    ):
        output = run_daily_pairwise(
            Path("run"),
            start_date=result_date,
            end_date=result_date,
            output_root=tmp_path,
            implementations=DEFAULT_PAIRWISE_IMPLEMENTATIONS,
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
            implementations=DEFAULT_PAIRWISE_IMPLEMENTATIONS,
        )
    with pytest.raises(ValueError, match="workers"):
        run_daily_pairwise(
            Path("run"),
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 1),
            output_root=tmp_path,
            workers=0,
            implementations=DEFAULT_PAIRWISE_IMPLEMENTATIONS,
        )


def test_daily_serial_parallel_and_single_discovery(tmp_path):
    import pyarrow.parquet as pq
    from sample_settings import DEFAULT_IMPLEMENTATIONS
    from test_joint_cellwise_temperature_pipeline import _write_joint_day

    from analysis.comparison import workflows

    run = tmp_path / "run"
    _write_joint_day(run, "2025-01-01", 1)
    _write_joint_day(run, "2025-01-02", 2)
    settings = dict(
        hsb_run_paths=(run,),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        implementations=DEFAULT_IMPLEMENTATIONS,
    )
    with patch.object(
        workflows, "discover_results", wraps=workflows.discover_results
    ) as discover:
        serial = run_daily_pairwise(
            **settings, output_root=tmp_path / "serial", workers=1
        )
    assert discover.call_count == 1
    parallel = run_daily_pairwise(
        **settings, output_root=tmp_path / "parallel", workers=2
    )
    for path in serial.glob("dates/*/*.parquet"):
        assert pq.read_table(path).equals(
            pq.read_table(parallel / path.relative_to(serial))
        )
