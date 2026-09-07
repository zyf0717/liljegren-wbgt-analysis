"""Default locations for the current cellwise artifact family."""

from pathlib import Path


_DEFAULT_ARTIFACT_SCOPE = Path("6-targets/2025-joint-v1")
DEFAULT_JOINT_OUTPUT_ROOT = (
    Path("data/cellwise/temperature-quantiles") / _DEFAULT_ARTIFACT_SCOPE
)
DEFAULT_QUANTILE_SPREAD_INPUT_ROOT = DEFAULT_JOINT_OUTPUT_ROOT
DEFAULT_QUANTILE_SPREAD_OUTPUT_ROOT = (
    Path("data/cellwise/quantile-spread") / _DEFAULT_ARTIFACT_SCOPE
)
DEFAULT_ISO7243_OUTPUT_ROOT = (
    Path("data/cellwise/iso7243-classification/6-targets/2025-hourly-wbgt-v2")
)
