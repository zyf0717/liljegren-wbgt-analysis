"""Display precision shared by generated analysis tables."""

from math import floor, log10


def format_percentage(value: float) -> str:
    """Format a percentage (0–100) with two significant figures."""
    return f"{format_two_significant_figures(value)}%"


def format_two_significant_figures(value: float) -> str:
    if value == 0.0:
        return "0"
    decimals = 1 - floor(log10(abs(value)))
    rounded = round(value, decimals)
    decimals = max(0, 1 - floor(log10(abs(rounded))))
    return f"{rounded:.{decimals}f}"
