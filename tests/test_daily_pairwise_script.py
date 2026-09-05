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
                    MODULE.REPOSITORY / "data" / "pairwise",
                )
                self.assertEqual(
                    args.log_file,
                    MODULE.REPOSITORY / "data" / f"daily_pairwise_{scope}.log",
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


if __name__ == "__main__":
    unittest.main()
