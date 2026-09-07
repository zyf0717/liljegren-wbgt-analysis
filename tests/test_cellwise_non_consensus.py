import numpy as np
import pyarrow.parquet as pq

from iso7243_fixtures import write_days
from liljegren_analysis.cellwise.non_consensus import analyze_cellwise_iso7243_non_consensus
from liljegren_analysis.cellwise.non_consensus_plots import plot_cellwise_iso7243_non_consensus


def test_joint_wbgt_only_non_consensus_and_map(tmp_path):
    run = tmp_path / "run"
    write_days(run, 2)
    out = analyze_cellwise_iso7243_non_consensus(
        run, start_date="2025-01-01", end_date="2025-01-02",
        thresholds_c=[25., 30.], output_root=tmp_path / "out", date_workers=1)
    values = pq.read_table(out.cellwise_non_consensus)
    assert values["n_joint"].to_pylist() == [36, 48, 0]
    assert values["n_non_consensus"].to_pylist() == [6, 0, 0]
    np.testing.assert_allclose(values["non_consensus_percent"].to_pylist(),
                               [100 / 6, 0, np.nan], equal_nan=True)
    figure = plot_cellwise_iso7243_non_consensus(out.metadata.parent)
    assert figure.png.stat().st_size > 0
