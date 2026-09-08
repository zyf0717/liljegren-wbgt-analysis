from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iso7243_fixtures import NAMES, reference_counts, reference_pairwise_counts, write_day, write_days
from liljegren_analysis.cellwise import iso7243
from liljegren_analysis.cellwise.non_consensus import analyze_cellwise_iso7243_non_consensus


def analyze(run, output, **kwargs):
    options = dict(start_date="2025-01-01", end_date="2025-01-01",
                   thresholds_c=[25.0, 30.0], output_root=output, date_workers=1)
    options.update(kwargs)
    return iso7243.analyze_cellwise_iso7243_classification(run, **options)


@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_raw_reference_direction_and_wbgt_only_support(tmp_path, batch_size):
    run = tmp_path / "run"
    write_day(run)
    output = analyze(run, tmp_path / "out", batch_size=batch_size)
    expected = reference_counts(run, ["2025-01-01"])
    non = pq.read_table(output.cellwise_non_consensus)
    np.testing.assert_array_equal(non["n_joint"], expected[0])
    np.testing.assert_array_equal(non["n_non_consensus"], expected[1])
    assert non["n_joint"].to_pylist() == [18, 24, 0]
    np.testing.assert_allclose(non["non_consensus_percent"].to_pylist(),
                               [100 / 6, 0, np.nan], equal_nan=True)
    pairs = pq.read_table(output.cell_pair_differences)
    assert pairs.num_rows == 45
    assert pairs.schema.field("n_compared").type == pa.uint32()
    assert pairs.schema.field("n_higher").type == pa.uint32()
    for pair in range(15):
        rows = pairs.slice(pair * 3, 3)
        np.testing.assert_array_equal(rows["n_compared"], expected[0])
        np.testing.assert_array_equal(rows["n_higher"], expected[2 + 2 * pair])
        np.testing.assert_array_equal(rows["n_lower"], expected[3 + 2 * pair])
    assert pairs["n_higher"][0].as_py() == 2
    assert pairs["n_lower"][0].as_py() == 1
    metadata = iso7243.load_hourly_metadata(output.metadata.parent)
    assert metadata["joint_observation_count"] == 42
    assert metadata["support_scope"] == "all-implementations-wbgt-only"
    assert '0,"< 25 C","wbgt_c < 25"' in output.classes.read_text()


@pytest.mark.parametrize("sample_scope", ["all-implementations", "pairwise"])
def test_parallel_resume_and_completed_reuse(tmp_path, sample_scope):
    run = tmp_path / "run"
    write_days(run, 2)
    serial = analyze(run, tmp_path / "serial", end_date="2025-01-02", batch_size=8,
                     sample_scope=sample_scope)
    parallel = analyze(run, tmp_path / "parallel", end_date="2025-01-02",
                       date_workers=2, batch_size=13, sample_scope=sample_scope)
    for field in ("cell_pair_differences", "cellwise_non_consensus", "cell_lookup"):
        left, right = (pq.read_table(getattr(output, field)) for output in (serial, parallel))
        assert left.schema == right.schema
        for column in left.column_names:
            if column == "non_consensus_percent":
                np.testing.assert_array_equal(left[column], right[column])
            else:
                assert left[column].equals(right[column])
    with patch.object(iso7243, "_write_outputs", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            analyze(run, tmp_path / "resume", end_date="2025-01-02", sample_scope=sample_scope)
    with patch.object(iso7243, "_scan_day", side_effect=AssertionError("must reuse")):
        resumed = analyze(run, tmp_path / "resume", end_date="2025-01-02", sample_scope=sample_scope)
        same = analyze_cellwise_iso7243_non_consensus(
            run, start_date="2025-01-01", end_date="2025-01-02",
            thresholds_c=[25., 30.], output_root=tmp_path / "resume", date_workers=1,
            sample_scope=sample_scope)
    assert pq.read_table(resumed.cell_pair_differences).equals(pq.read_table(serial.cell_pair_differences))
    assert same.cellwise_non_consensus == resumed.cellwise_non_consensus
    assert json.loads(resumed.metadata.read_text())["dates_reused"] == 2


@pytest.mark.parametrize("batch_size", [1, 11, 100])
def test_pairwise_support_matches_independent_reference(tmp_path, batch_size):
    run = tmp_path / "run"
    write_day(run)
    path = run / "results/2025-01-01/0001" / NAMES[0] / "results.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    # More A/B changes on hours excluded only because another target is invalid.
    for row in rows:
        if 3 <= row["row_ordinal"] <= 8:
            row["wbgt_c"] = 35.
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    out = analyze(run, tmp_path / "pairwise", sample_scope="pairwise", batch_size=batch_size)
    expected = reference_pairwise_counts(run, ["2025-01-01"])
    pairs = pq.read_table(out.cell_pair_differences)
    for pi in range(15):
        for ci, column in enumerate(("n_compared", "n_higher", "n_lower")):
            np.testing.assert_array_equal(pairs.slice(pi * 3, 3)[column], expected[pi, ci])
    assert pairs["n_compared"][0].as_py() == 24
    assert pairs["n_higher"][0].as_py() == 8
    assert pairs["n_lower"][0].as_py() == 1
    non = pq.read_table(out.cellwise_non_consensus)
    common = reference_counts(run, ["2025-01-01"])
    np.testing.assert_array_equal(non["n_joint"], common[0])
    np.testing.assert_array_equal(non["n_non_consensus"], common[1])
    assert non["n_non_consensus"][0].as_py() == 3
    metadata = iso7243.load_hourly_metadata(out.metadata.parent)
    assert metadata["sample_scope"] == "pairwise"
    assert metadata["support_scope"] == "pairwise-wbgt-only"
    assert metadata["non_consensus_support_scope"] == "all-implementations-wbgt-only"
    assert metadata["pair_observation_count"] == int(expected[:, 0].sum())


def test_scope_isolation_and_legacy_default_reuse(tmp_path):
    run = tmp_path / "run"
    write_day(run)
    root = tmp_path / "out"
    out = analyze(run, root)
    # Earlier all-implementation metadata did not carry these additive fields.
    metadata = json.loads(out.metadata.read_text())
    metadata.pop("sample_scope")
    metadata.pop("non_consensus_support_scope")
    out.metadata.write_text(json.dumps(metadata))
    with patch.object(iso7243, "_scan_day", side_effect=AssertionError("must reuse")):
        analyze(run, root)
    with pytest.raises(ValueError, match="Incompatible"):
        analyze(run, root, sample_scope="pairwise")
    analyze(run, root, sample_scope="pairwise", force=True)
    assert json.loads(out.metadata.read_text())["dates_scanned"] == 1
    assert len(list((root / ".work").glob("*/2025-01-01.npz"))) == 2
    # A forced return to the default recovers its original checkpoint generation.
    with patch.object(iso7243, "_scan_day", side_effect=AssertionError("must reuse")):
        analyze(run, root, force=True)
    assert json.loads(out.metadata.read_text())["dates_reused"] == 1
    with pytest.raises(ValueError, match="sample scope"):
        analyze(run, root, sample_scope="invalid")


def test_uint32_counts_above_255(tmp_path):
    run = tmp_path / "run"
    write_days(run, 12, complex_rows=False)
    out = analyze(run, tmp_path / "out", end_date="2025-01-12")
    pairs = pq.read_table(out.cell_pair_differences)
    assert pairs["n_compared"][0].as_py() == 288
    assert pairs["n_higher"][0].as_py() == 288
    assert pairs["n_lower"][0].as_py() == 0


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "Duplicate"),
    ("ordinal", "Conflicting"),
    ("hour", "identity"),
    ("date", "identity"),
    ("separator", "identity"),
])
def test_identity_rejection(tmp_path, mutation, match):
    run = tmp_path / "run"
    write_day(run)
    path = run / "results/2025-01-01/0001" / NAMES[1] / "results.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "ordinal":
        rows[0]["row_ordinal"] += 1
    elif mutation == "hour":
        rows[0]["row_id"] = rows[0]["row_id"][:-2] + "99"
    elif mutation == "date":
        rows[0]["row_id"] = rows[0]["row_id"].replace("20250101", "20250102")
    else:
        rows[0]["row_id"] = rows[0]["row_id"][:28] + "_" + rows[0]["row_id"][29:]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    for batch_size in (1, 1000):
        with pytest.raises(ValueError, match=match):
            analyze(run, tmp_path / f"out-{batch_size}", batch_size=batch_size)


def test_missing_shard_and_unselected_implementation(tmp_path):
    run = tmp_path / "run"
    write_day(run, names=(*NAMES, "unselected"))
    analyze(run, tmp_path / "out")
    missing = run / "results/2025-01-01/0001" / NAMES[0] / "results.parquet"
    missing.rename(missing.with_suffix(".saved"))
    with pytest.raises(ValueError, match="inventory"):
        analyze(run, tmp_path / "missing")


def test_configuration_lock_and_completion_validation(tmp_path):
    run = tmp_path / "run"
    write_day(run)
    root = tmp_path / "out"
    out = analyze(run, root)
    with pytest.raises(ValueError, match="Incompatible"):
        analyze(run, root, thresholds_c=[26.])
    with (root / ".writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer"):
            analyze(run, root)
    out.classes.write_text("modified")
    with pytest.raises(ValueError, match="modified"):
        analyze(run, root)
    analyze(run, root, force=True)
    iso7243.load_hourly_metadata(root)
    out.metadata.rename(root / "metadata.saved.json")
    with pytest.raises(ValueError, match="completion"):
        iso7243.load_hourly_metadata(root)
    # Existing compatible daily summaries remain sufficient to recover publication.
    with patch.object(iso7243, "_scan_day", side_effect=AssertionError("must reuse")):
        analyze(run, root)


def test_class_dtype_promotion(tmp_path):
    run = tmp_path / "run"
    write_day(run, complex_rows=False)
    thresholds = list(np.arange(-240., 31.))
    out = analyze(run, tmp_path / "out", thresholds_c=thresholds)
    assert pq.read_table(out.cell_pair_differences)["n_higher"][0].as_py() == 24


def test_corrupt_checkpoint_and_changed_input(tmp_path):
    run = tmp_path / "run"
    write_day(run)
    root = tmp_path / "out"
    out = analyze(run, root)
    checkpoint = next((root / ".work").glob("*/2025-01-01.npz"))
    checkpoint.write_bytes(b"interrupted checkpoint")
    analyze(run, root, force=True)
    assert json.loads(out.metadata.read_text())["dates_scanned"] == 1
    write_day(run, complex_rows=False)
    with pytest.raises(ValueError, match="Incompatible"):
        analyze(run, root)
    analyze(run, root, force=True)
    assert pq.read_table(out.cell_pair_differences)["n_higher"][0].as_py() == 24


def test_unknown_cell_on_later_date(tmp_path):
    run = tmp_path / "run"
    write_days(run, 2)
    path = run / "results/2025-01-02/0001" / NAMES[0] / "results.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["row_id"] = rows[0]["row_id"].replace("n000000", "n005000")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="unknown"):
        analyze(run, tmp_path / "out", end_date="2025-01-02")


@pytest.mark.parametrize("scope", ["all-implementations", "pairwise"])
def test_cli_sample_scope(tmp_path, scope):
    run = tmp_path / "run"
    write_day(run)
    root = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_cellwise_iso7243_classification.py"
    command = [sys.executable, str(script), "--hsb-run", str(run),
               "--start-date", "2025-01-01", "--end-date", "2025-01-01",
               "--thresholds-c", "25", "30", "--date-workers", "1",
               "--output-root", str(root)]
    if scope == "pairwise":
        command.extend(["--sample-scope", scope])
    subprocess.run(command, check=True, capture_output=True, text=True)
    metadata = iso7243.load_hourly_metadata(root)
    assert metadata["sample_scope"] == scope
    pairs = pq.read_table(root / "cell_pair_differences.parquet")
    assert pairs["n_compared"][0].as_py() == (24 if scope == "pairwise" else 18)
