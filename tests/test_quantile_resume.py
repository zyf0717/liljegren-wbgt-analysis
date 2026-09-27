"""Interrupted disk-backed replay must restart without double counting."""

from unittest.mock import patch

import pyarrow.parquet as pq
import pytest
from sample_settings import DEFAULT_IMPLEMENTATIONS, DEFAULT_PROBABILITIES
from test_joint_cellwise_temperature_pipeline import _write_joint_day

from analysis.cellwise import _joint_masks
from analysis.cellwise.accumulator import CellwiseHistogramAccumulator
from analysis.cellwise.joint_pipeline import (
    compute_joint_cellwise_temperature_quantiles,
)
from analysis.comparison.distributions import WBGTHistogramSpec


def test_interrupted_replay_reuses_masks_and_completed_implementations(tmp_path):
    run = tmp_path / "run"
    _write_joint_day(run, "2025-01-01", 1)
    _write_joint_day(run, "2025-01-02", 2)
    settings = dict(
        start_date="2025-01-01",
        end_date="2025-01-02",
        histogram_spec=WBGTHistogramSpec(minimum_c=-70, maximum_c=70, width_c=1),
        exact_validation_cells=3,
        date_workers=1,
        replay_workers=1,
    )
    reference = compute_joint_cellwise_temperature_quantiles(
        run,
        output_root=tmp_path / "reference",
        **settings,
        implementations=DEFAULT_IMPLEMENTATIONS,
        probabilities=DEFAULT_PROBABILITIES,
    )
    output = tmp_path / "resumed"
    update = CellwiseHistogramAccumulator.update
    calls = 0

    def interrupt(self, cells, values):
        nonlocal calls
        calls += 1
        update(self, cells, values)
        if calls == 8:
            raise RuntimeError("interrupted after partial histogram update")

    with patch.object(CellwiseHistogramAccumulator, "update", interrupt):
        with pytest.raises(RuntimeError, match="partial histogram"):
            compute_joint_cellwise_temperature_quantiles(
                run,
                output_root=output,
                **settings,
                implementations=DEFAULT_IMPLEMENTATIONS,
                probabilities=DEFAULT_PROBABILITIES,
            )
    assert not (output / "metadata.json").exists()
    completed = output / "implementations" / f"{DEFAULT_IMPLEMENTATIONS[0]}.parquet"
    before = completed.stat().st_mtime_ns
    with patch.object(
        _joint_masks,
        "scan_joint_day_mask",
        side_effect=AssertionError("must reuse masks"),
    ):
        compute_joint_cellwise_temperature_quantiles(
            run,
            output_root=output,
            **settings,
            implementations=DEFAULT_IMPLEMENTATIONS,
            probabilities=DEFAULT_PROBABILITIES,
        )
    assert completed.stat().st_mtime_ns == before
    for implementation in DEFAULT_IMPLEMENTATIONS:
        name = f"implementations/{implementation}.parquet"
        left, right = pq.read_table(reference / name), pq.read_table(output / name)
        assert left.equals(right)
