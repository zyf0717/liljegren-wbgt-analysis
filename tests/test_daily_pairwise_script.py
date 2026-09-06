from __future__ import annotations

import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "scripts" / "daily_pairwise.py"
SPEC = spec_from_file_location("daily_pairwise_script", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DailyPairwiseScriptTests(unittest.TestCase):
    def test_defaults_use_complete_case_product_and_one_worker(self) -> None:
        with patch.object(sys, "argv", ["daily_pairwise.py"]):
            args = MODULE.parse_args()

        self.assertEqual(args.sample_scope, "all-implementations")
        self.assertEqual(args.workers, 1)

    def test_default_output_is_parquet_root(self) -> None:
        for scope in ("pairwise", "all-implementations"):
            with (
                self.subTest(scope=scope),
                patch.object(
                    sys,
                    "argv",
                    ["daily_pairwise.py", "--sample-scope", scope],
                ),
            ):
                args = MODULE.parse_args()
                self.assertEqual(
                    args.output_root,
                    MODULE.REPOSITORY / "data" / "pairwise" / "6-targets",
                )
                self.assertEqual(
                    MODULE.DEFAULT_RUNS,
                    (
                        MODULE.REPOSITORY.parent
                        / "HeatStressBench"
                        / ".hsb"
                        / "runs"
                        / "wbgt-comparison-20260902-150355",
                    ),
                )

    def test_main_delegates_to_workflow_api(self) -> None:
        with (
            patch.object(sys, "argv", ["daily_pairwise.py"]),
            patch.object(
                MODULE, "run_daily_pairwise", return_value=Path("output")
            ) as run,
        ):
            MODULE.main()

        run.assert_called_once_with(
            MODULE.DEFAULT_RUNS,
            start_date=unittest.mock.ANY,
            end_date=unittest.mock.ANY,
            output_root=MODULE.REPOSITORY / "data/pairwise/6-targets",
            sample_scope="all-implementations",
            workers=1,
            overwrite=False,
        )


if __name__ == "__main__":
    unittest.main()
