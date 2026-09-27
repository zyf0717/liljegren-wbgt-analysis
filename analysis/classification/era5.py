"""Monthly and annual country summaries from shared daily ERA5 class arrays."""

from collections import Counter
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..config import implementation_names
from ..grid import grid_from_source_cell_ids
from ..results import date_range, open_country_cache
from ._summary_tables import change_summary, non_consensus_summary, transition_summary
from .core import class_definitions, validate_thresholds
from .daily import load_daily_classes


def analyze_era5_country_cache(
    cache_path: str | Path,
    start_date: date,
    end_date: date,
    thresholds_c,
    *,
    implementations,
) -> dict[str, pd.DataFrame]:
    """Read bounded batches for each available date and keep monthly counts only."""
    requested = set(date_range(start_date, end_date))
    names = implementation_names(implementations)
    thresholds = validate_thresholds(thresholds_c)
    source = open_country_cache(Path(cache_path))
    dates = sorted(
        {day for day, name in source.scopes if day in requested and name in names}
    )
    if not dates:
        raise ValueError("country cache has no rows in the requested date range")
    # Read just cell identities once; the universe also includes cells missing on day one.
    source_ids = set()
    for batch in pq.ParquetFile(source.path).iter_batches(columns=["cell_id"]):
        source_ids.update(pc.unique(batch["cell_id"]).to_pylist())
    grid = grid_from_source_cell_ids(sorted(source_ids))
    ids = grid.table["source_cell_id"].combine_chunks()
    pairs = tuple(combinations(sorted(names), 2))
    index = {name: i for i, name in enumerate(names)}
    paired, transitions, consensus, coverage = {}, Counter(), {}, Counter()
    cells_seen = np.zeros(grid.cell_count, dtype=bool)
    valid_counts = np.zeros(len(names), dtype=np.int64)
    result_rows = 0
    class_count = len(thresholds) + 1
    for day in dates:
        daily = load_daily_classes(
            source, day, names, thresholds, ids, include_coverage=True
        )
        matrix = daily.classes
        valid_counts += (matrix >= 0).sum(axis=1)
        cells_seen |= daily.cells_seen
        result_rows += daily.rows
        coverage.update(daily.coverage)
        month = day.strftime("%Y-%m")
        for name_a, name_b in pairs:
            a, b = matrix[index[name_a]], matrix[index[name_b]]
            supported = (a >= 0) & (b >= 0)
            a, b = a[supported].astype(np.int64), b[supported].astype(np.int64)
            if not len(a):
                continue
            higher, lower = int((a > b).sum()), int((a < b).sum())
            key = (month, name_a, name_b)
            paired.setdefault(key, np.zeros(4, dtype=np.uint64))[:] += np.array(
                (len(a), higher + lower, higher, lower), dtype=np.uint64
            )
            counts = np.bincount(a * class_count + b, minlength=class_count**2)
            for code in np.flatnonzero(counts):
                transitions[
                    (*key, int(code // class_count), int(code % class_count))
                ] += int(counts[code])
        joint = np.all(matrix >= 0, axis=0)
        if joint.any():
            values = matrix[:, joint]
            counts = consensus.setdefault(month, np.zeros(2, dtype=np.uint64))
            counts += np.array(
                (
                    int(joint.sum()),
                    int((values.max(axis=0) != values.min(axis=0)).sum()),
                ),
                dtype=np.uint64,
            )
    if np.any(valid_counts == 0):
        raise ValueError(
            "classification implementations are missing valid observations"
        )
    if not paired:
        raise ValueError("fewer than two implementations have matched observations")
    classes = class_definitions(thresholds)[["class_id", "class_label"]]
    outputs = {"classes.csv": classes}
    for monthly, prefix in ((False, ""), (True, "monthly_")):
        outputs[prefix + "pair_change_summary.csv"] = change_summary(
            paired, monthly=monthly
        )
        outputs[prefix + "pair_transition_counts.csv"] = transition_summary(
            transitions, classes, monthly=monthly
        )
        outputs[prefix + "non_consensus_summary.csv"] = non_consensus_summary(
            consensus, monthly=monthly
        )
    outputs["observation_summary.csv"] = pd.DataFrame(
        [
            dict(
                result_rows=result_rows,
                valid_wbgt_observations=int(valid_counts.sum()),
                implementations=len(names),
                cells=int(cells_seen.sum()),
                first_date=dates[0],
                last_date=dates[-1],
            )
        ]
    )
    outputs["quantity_coverage.csv"] = pd.DataFrame(
        [(*key, count) for key, count in sorted(coverage.items())],
        columns=[
            "quantity",
            "implementation",
            "quantity_status",
            "quantity_value_present",
            "n",
        ],
    )
    return outputs
