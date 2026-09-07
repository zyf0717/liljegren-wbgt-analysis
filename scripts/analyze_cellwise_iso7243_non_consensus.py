#!/usr/bin/env python3
"""Compute/reuse hourly WBGT class summaries and report non-consensus output."""

from __future__ import annotations

from liljegren_analysis.cellwise.iso7243_cli import run


if __name__ == "__main__":
    run(non_consensus=True)
