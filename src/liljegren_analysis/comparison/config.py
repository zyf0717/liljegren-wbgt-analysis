"""Stable pairwise comparison configuration."""

from __future__ import annotations

from typing import Literal


SampleScope = Literal["pairwise", "all-implementations"]
SAMPLE_SCOPES: tuple[SampleScope, ...] = ("pairwise", "all-implementations")
EXPECTED_IMPLEMENTATIONS = (
    "heatstress",
    "heatstress-jl",
    "heatstress-r",
    "liljegren-c",
    "meteor",
    "pywbgt",
    "thermofeel",
)
