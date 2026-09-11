"""Default locations for the current cellwise artifact family."""

from pathlib import Path


_DEFAULT_TARGET_SCOPE = Path("data/cellwise/6-targets")
_DEFAULT_JOINT_SCOPE = Path("2025-joint-v1")
DEFAULT_JOINT_OUTPUT_ROOT = (
    _DEFAULT_TARGET_SCOPE / "temperature-quantiles" / _DEFAULT_JOINT_SCOPE
)
DEFAULT_QUANTILE_SPREAD_INPUT_ROOT = DEFAULT_JOINT_OUTPUT_ROOT
DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT = (
    _DEFAULT_TARGET_SCOPE / "quantile-spread" / _DEFAULT_JOINT_SCOPE
)
DEFAULT_ISO7243_OUTPUT_ROOT = (
    _DEFAULT_TARGET_SCOPE / "iso7243-classification" / "2025-hourly-wbgt-v2"
)
