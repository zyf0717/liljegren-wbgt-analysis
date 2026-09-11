"""Stable pairwise comparison configuration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal


SampleScope = Literal["pairwise", "all-implementations"]
SAMPLE_SCOPES: tuple[SampleScope, ...] = ("pairwise", "all-implementations")
DEFAULT_ISO7243_IMPLEMENTATIONS = (
    "argonne-liljegren-c",
    "heatstress",
    "meteor",
    "thermofeel",
    "kwodzicki-pywbgt",
    "kong-huber-pywbgt",
)
DEFAULT_PAIRWISE_IMPLEMENTATIONS = (
    "argonne-liljegren-c",
    "heatstress",
    "kong-huber-pywbgt",
    "kwodzicki-pywbgt",
    "meteor",
    "thermofeel",
)
PAIRWISE_IMPLEMENTATION_SET_VERSION = 1
_PAIRWISE_IMPLEMENTATION_SET_VERSIONS = {
    DEFAULT_PAIRWISE_IMPLEMENTATIONS: PAIRWISE_IMPLEMENTATION_SET_VERSION,
}


def pairwise_implementations(
    implementations: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve an explicit implementation selection or the default set."""

    if isinstance(implementations, str):
        raise TypeError("implementations must be a sequence, not a string")
    selected = (
        DEFAULT_PAIRWISE_IMPLEMENTATIONS
        if implementations is None
        else tuple(implementations)
    )
    if len(selected) < 2:
        raise ValueError("implementations must contain at least two targets")
    if any(not implementation for implementation in selected):
        raise ValueError("implementation names must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("implementations must be unique")
    return selected


def pairwise_implementation_set_version(
    implementations: Sequence[str],
) -> int | None:
    """Return the stable version for a known implementation set."""

    return _PAIRWISE_IMPLEMENTATION_SET_VERSIONS.get(tuple(implementations))


REFERENCE_IMPLEMENTATION = "argonne-liljegren-c"
