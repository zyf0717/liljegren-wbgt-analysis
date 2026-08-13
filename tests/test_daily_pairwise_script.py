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
    def test_scope_specific_default_paths(self) -> None:
        for scope, stem in {
            "pairwise": "daily_pairwise_v4",
            "all-implementations": "daily_pairwise_all_implementations_v4",
        }.items():
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
                    args.database,
                    MODULE.REPOSITORY / "data" / f"{stem}.duckdb",
                )
                self.assertEqual(
                    args.log_file,
                    MODULE.REPOSITORY / "data" / f"{stem}.log",
                )


if __name__ == "__main__":
    unittest.main()
