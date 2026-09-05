"""Stable pairwise comparison configuration."""

from __future__ import annotations

from typing import Literal


SampleScope = Literal["pairwise", "all-implementations"]
SAMPLE_SCOPES: tuple[SampleScope, ...] = ("pairwise", "all-implementations")
IMPLEMENTATION_SET_VERSION = 2
EXPECTED_IMPLEMENTATIONS = (
    "argonne-liljegren-c",
    "heatstress",
    "heatstress-jl",
    "heatstress-r",
    "kong-huber-pywbgt",
    "kwodzicki-pywbgt",
    "mdljts-wbgt",
    "meteor",
    "thermofeel",
)
REFERENCE_IMPLEMENTATION = "argonne-liljegren-c"
EXPECTED_IMPLEMENTATION_PAIRS = (
    len(EXPECTED_IMPLEMENTATIONS) * (len(EXPECTED_IMPLEMENTATIONS) - 1) // 2
)
