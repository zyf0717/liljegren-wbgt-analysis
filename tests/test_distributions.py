"""Check the binning expressions used by real pairwise computations."""

import numpy as np
import polars as pl
import pytest

from analysis.comparison.distributions import absolute_difference_bins


@pytest.mark.parametrize("sign", [1, -1])
def test_absolute_difference_boundaries(sign):
    values = np.array([0.0, 1.995, 2.0, 4.975, 5.0, 19.875, 20.0, 99.5, 100.0]) * sign
    index, lower, upper = absolute_difference_bins(pl.col("difference"))
    table = (
        pl.DataFrame({"difference": values})
        .with_columns(index.alias("abs_bin_id"))
        .with_columns(lower.alias("lower"), upper.alias("upper"))
    )
    assert table["abs_bin_id"].to_list() == [0, 199, 200, 259, 260, 319, 320, 399, 400]
    np.testing.assert_allclose(
        table["lower"], [0.0, 1.99, 2.0, 4.95, 5.0, 19.75, 20.0, 99.0, 100.0]
    )
    np.testing.assert_allclose(
        table["upper"][:-1], [0.01, 2.0, 2.05, 5.0, 5.25, 20.0, 21.0, 100.0]
    )
    assert table["upper"][-1] is None
