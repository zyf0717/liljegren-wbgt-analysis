from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis.cellwise import (
    DEFAULT_DATE_WORKERS,
    DEFAULT_JOINT_IMPLEMENTATIONS,
    DEFAULT_PROBABILITIES,
    analyze_cellwise_quantile_spread,
    compute_joint_cellwise_temperature_quantiles,
)
from liljegren_analysis.cellwise.joint_pipeline import _open_manifest
from liljegren_analysis.cellwise.iso7243 import (
    DEFAULT_DATE_WORKERS as DEFAULT_ISO7243_DATE_WORKERS,
)
from liljegren_analysis.comparison.distributions import WBGTHistogramSpec

_CELLS = (
    "era5-global-n000000-e0000000",
    "era5-global-n002500-e0000000",
    "era5-global-n005000-e0000000",
)
_AVAILABLE_IMPLEMENTATIONS = (
    *DEFAULT_JOINT_IMPLEMENTATIONS,
    "mdljts-wbgt",
    "heatstress-jl",
    "heatstress-r",
)


def _write_joint_day(hsb_run: Path, result_date: str, day: int) -> None:
    for implementation_index, implementation in enumerate(_AVAILABLE_IMPLEMENTATIONS):
        path = (
            hsb_run
            / "results"
            / result_date
            / "0001"
            / implementation
            / "results.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        wbgt_status = ["ok", "ok", "ok"]
        if implementation == "meteor":
            wbgt_status[2] = "invalid"
        offset = implementation_index / 10
        tnwb_values: list[float | None] = [
            10.0 + day + offset,
            20.0 + day + offset,
            30.0 + day + offset,
        ]
        if implementation == "thermofeel" and day == 2:
            tnwb_values[1] = None
        pq.write_table(
            pa.table(
                {
                    "cycle_ordinal": pa.array([1, 1, 1], type=pa.uint32()),
                    "row_id": [
                        f"{cell}-{result_date.replace('-', '')}00" for cell in _CELLS
                    ],
                    "row_ordinal": pa.array([0, 1, 2], type=pa.uint32()),
                    "wbgt_status": wbgt_status,
                    "wbgt_c": [
                        20.0 + day + offset,
                        30.0 + day + offset,
                        40.0 + day + offset,
                    ],
                    "tg_status": ["ok", "ok", "ok"],
                    "tg_c": [
                        30.0 + day + offset,
                        40.0 + day + offset,
                        50.0 + day + offset,
                    ],
                    "tnwb_status": ["ok", "ok", "ok"],
                    "tnwb_c": pa.array(tnwb_values, type=pa.float64()),
                }
            ),
            path,
        )


class JointCellwiseTemperaturePipelineTests(unittest.TestCase):
    def test_default_date_worker_count_is_shared(self) -> None:
        self.assertEqual(DEFAULT_DATE_WORKERS, 8)
        self.assertEqual(DEFAULT_ISO7243_DATE_WORKERS, DEFAULT_DATE_WORKERS)

    def test_old_replay_manifest_is_not_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {"schema_version": 2, "replay_strategy": "current"}
            (root / "manifest.json").write_text(
                json.dumps({"schema_version": 2, "unit_provenance": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                _open_manifest(root / "manifest.json", expected, root / "metadata.json")

    def test_default_targets_match_report_scope(self) -> None:
        self.assertEqual(
            DEFAULT_JOINT_IMPLEMENTATIONS,
            (
                "argonne-liljegren-c",
                "heatstress",
                "meteor",
                "thermofeel",
                "kwodzicki-pywbgt",
                "kong-huber-pywbgt",
            ),
        )

    def test_all_implementation_all_quantity_support_is_shared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hsb_run = root / "hsb-run"
            output = root / "output"
            _write_joint_day(hsb_run, "2025-01-01", 1)
            _write_joint_day(hsb_run, "2025-01-02", 2)
            output.mkdir()
            stale = output / "stale"
            stale.write_text("old product")

            result_root = compute_joint_cellwise_temperature_quantiles(
                hsb_run,
                start_date="2025-01-01",
                end_date="2025-01-02",
                output_root=output,
                histogram_spec=WBGTHistogramSpec(
                    minimum_c=-70.0, maximum_c=70.0, width_c=1.0, version=99
                ),
                exact_validation_cells=3,
                date_workers=2,
                replay_workers=2,
                overwrite=True,
            )
            self.assertFalse(stale.exists())

            self.assertEqual(result_root, output.resolve())
            for implementation in DEFAULT_JOINT_IMPLEMENTATIONS:
                table = pq.read_table(
                    output / "implementations" / f"{implementation}.parquet"
                )
                self.assertEqual(table["n_joint"].to_pylist(), [2, 1, 0])
                self.assertEqual(table["n_expected"].to_pylist(), [48, 48, 48])
                for quantity in ("wbgt", "tg", "tnwb"):
                    self.assertEqual(table[f"{quantity}_p50_c"].null_count, 1)
                    self.assertEqual(table[f"{quantity}_p95_c"].null_count, 1)
                    native = table[f"{quantity}_native_valid_n"].to_pylist()
                    self.assertTrue(all(a >= b for a, b in zip(native, [2, 1, 0])))

            heatstress = pq.read_table(
                output / "implementations" / "heatstress.parquet"
            )
            meteor = pq.read_table(output / "implementations" / "meteor.parquet")
            self.assertEqual(heatstress["wbgt_native_valid_n"].to_pylist(), [2, 2, 2])
            self.assertEqual(meteor["wbgt_native_valid_n"].to_pylist(), [2, 2, 0])
            self.assertEqual(meteor["tg_native_valid_n"].to_pylist(), [2, 2, 2])
            thermofeel = pq.read_table(
                output / "implementations" / "thermofeel.parquet"
            )
            self.assertEqual(thermofeel["tnwb_native_valid_n"].to_pylist(), [2, 1, 2])
            self.assertEqual(heatstress["wbgt_p50_c"].to_pylist(), [21.5, 31.5, None])
            self.assertEqual(heatstress["tg_p95_c"].to_pylist(), [32.5, 41.5, None])

            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["probabilities"], list(DEFAULT_PROBABILITIES))
            self.assertEqual(
                metadata["support_scope"], "all-implementations-all-quantities"
            )
            self.assertEqual(metadata["quantities"], ["wbgt", "tg", "tnwb"])
            self.assertEqual(
                metadata["replay_strategy"],
                "implementation-all-quantities-in-memory-v3",
            )
            self.assertEqual(metadata["joint_observation_count"], 3)
            self.assertEqual(metadata["cells_with_joint_observations"], 2)
            self.assertEqual(metadata["date_workers_this_invocation"], 2)
            self.assertEqual(metadata["replay_workers_this_invocation"], 2)
            self.assertEqual(
                set(metadata["unit_provenance"]), set(DEFAULT_JOINT_IMPLEMENTATIONS)
            )
            first_provenance = metadata["unit_provenance"]["heatstress"]
            self.assertEqual(
                first_provenance["concurrent_histogram_memmap_bytes"],
                3 * first_provenance["histogram_memmap_bytes_each"],
            )
            self.assertFalse((output / ".work").exists())

            spread = analyze_cellwise_quantile_spread(
                output,
                output_root=root / "tg-analysis",
                quantity="tg",
            )
            spread_metadata = json.loads(spread.metadata.read_text())
            self.assertEqual(spread_metadata["quantity"], "tg")
            self.assertEqual(spread_metadata["complete_case_cell_count"], 2)
            self.assertLess(spread_metadata["complete_case_area_fraction"], 1.0)

            # A completed compatible run is idempotent and does not rebuild masks.
            self.assertEqual(
                compute_joint_cellwise_temperature_quantiles(
                    hsb_run,
                    start_date="2025-01-01",
                    end_date="2025-01-02",
                    output_root=output,
                    histogram_spec=WBGTHistogramSpec(
                        minimum_c=-70.0, maximum_c=70.0, width_c=1.0, version=99
                    ),
                ),
                output.resolve(),
            )

    def test_duplicate_target_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hsb_run = root / "hsb-run"
            _write_joint_day(hsb_run, "2025-01-01", 1)
            path = hsb_run / "results/2025-01-01/0001/heatstress/results.parquet"
            table = pq.read_table(path)
            pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), path)

            with self.assertRaisesRegex(ValueError, "Duplicate or inconsistent"):
                compute_joint_cellwise_temperature_quantiles(
                    hsb_run,
                    start_date="2025-01-01",
                    end_date="2025-01-01",
                    output_root=root / "output",
                    histogram_spec=WBGTHistogramSpec(
                        minimum_c=-70.0, maximum_c=70.0, width_c=1.0, version=99
                    ),
                    exact_validation_cells=0,
                )

    def test_reported_open_edge_quantile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hsb_run = root / "hsb-run"
            _write_joint_day(hsb_run, "2025-01-01", 1)

            with self.assertRaisesRegex(ValueError, "open histogram edge bin"):
                compute_joint_cellwise_temperature_quantiles(
                    hsb_run,
                    start_date="2025-01-01",
                    end_date="2025-01-01",
                    output_root=root / "output",
                    histogram_spec=WBGTHistogramSpec(
                        minimum_c=-10.0, maximum_c=10.0, width_c=1.0, version=99
                    ),
                    exact_validation_cells=0,
                )


if __name__ == "__main__":
    unittest.main()
