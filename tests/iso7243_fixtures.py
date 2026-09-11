"""Small raw hourly datasets with stable identities and independent expectations."""

from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from liljegren_analysis.cellwise.joint_pipeline import DEFAULT_JOINT_IMPLEMENTATIONS

NAMES = DEFAULT_JOINT_IMPLEMENTATIONS
PAIR_COUNT = len(NAMES) * (len(NAMES) - 1) // 2
CELLS = (
    "era5-global-n000000-e0000000",
    "era5-global-n002500-e0000000",
    "era5-global-s002500-e0000000",
)


def write_day(root, day="2025-01-01", *, complex_rows=True, names=NAMES):
    for index, name in enumerate(names):
        rows = []
        for ci, cell in enumerate(CELLS):
            for hour in range(24):
                value, status = 24.0, "ok"
                if ci == 0:
                    if complex_rows:
                        if index == 0 and hour in (0, 2):
                            value = 25.0 if hour == 0 else 31.0
                        if index == 1 and hour == 1:
                            value = 25.0
                        if (index, hour) == (2, 3):
                            status = "invalid"
                        if (index, hour) == (3, 4):
                            value = np.nan
                        if (index, hour) == (4, 5):
                            continue
                        if (index, hour) == (5, 6):
                            value = None
                        if (index, hour) == (3, 7):
                            value = np.inf
                        if (index, hour) == (2, 8):
                            status = None
                    elif index == 0:
                        value = 30.0
                if ci == 2 and index == 0:
                    status = "invalid"
                rows.append(dict(
                    cycle_ordinal=1, row_id=f"{cell}-{day.replace('-', '')}{hour:02}",
                    row_ordinal=ci * 24 + hour, wbgt_c=value, wbgt_status=status,
                    tg_status="invalid", tg_c=np.nan, tnwb_status="invalid", tnwb_c=None,
                ))
        if index % 2:
            rows.reverse()
        table = pa.Table.from_pylist(rows)
        for column in ("cycle_ordinal", "row_ordinal"):
            table = table.set_column(table.schema.get_field_index(column), column,
                                     table[column].cast(pa.uint32()))
        table = table.set_column(table.schema.get_field_index("tnwb_c"), "tnwb_c",
                                 table["tnwb_c"].cast(pa.float32()))
        path = root / "results" / day / "0001" / name / "results.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=7 + index)


def write_days(root, count, **kwargs):
    for offset in range(count):
        write_day(root, str(date(2025, 1, 1) + timedelta(days=offset)), **kwargs)


def reference_counts(root, days, thresholds=(25.0, 30.0)):
    """Deliberately simple keyed Python reference, independent of dense arrays."""
    result = np.zeros((2 + 2 * PAIR_COUNT, len(CELLS)), dtype=np.uint64)
    for day in days:
        by_name = {}
        for name in NAMES:
            table = pq.read_table(root / "results" / day / "0001" / name / "results.parquet")
            by_name[name] = {row["row_id"]: row for row in table.to_pylist()}
        common = set.intersection(*(set(rows) for rows in by_name.values()))
        for identity in common:
            rows = [by_name[name][identity] for name in NAMES]
            if not all(row["wbgt_status"] == "ok" and row["wbgt_c"] is not None
                       and np.isfinite(row["wbgt_c"]) for row in rows):
                continue
            ci = CELLS.index(identity[:28])
            classes = [sum(row["wbgt_c"] >= cut for cut in thresholds) for row in rows]
            result[0, ci] += 1
            result[1, ci] += len(set(classes)) > 1
            pair = 0
            for a in range(len(NAMES)):
                for b in range(a + 1, len(NAMES)):
                    result[2 + 2 * pair, ci] += classes[a] > classes[b]
                    result[3 + 2 * pair, ci] += classes[a] < classes[b]
                    pair += 1
    return result


def reference_pairwise_counts(root, days, thresholds=(25.0, 30.0)):
    """Independent pair intersections; each result contains compared/higher/lower."""
    result = np.zeros((PAIR_COUNT, 3, len(CELLS)), dtype=np.uint64)
    for day in days:
        records = []
        for name in NAMES:
            rows = pq.read_table(
                root / "results" / day / "0001" / name / "results.parquet"
            ).to_pylist()
            records.append({row["row_id"]: row for row in rows
                            if row["wbgt_status"] == "ok" and row["wbgt_c"] is not None
                            and np.isfinite(row["wbgt_c"])})
        pair = 0
        for a in range(len(NAMES)):
            for b in range(a + 1, len(NAMES)):
                for identity in records[a].keys() & records[b].keys():
                    ci = CELLS.index(identity[:28])
                    ca = sum(records[a][identity]["wbgt_c"] >= cut for cut in thresholds)
                    cb = sum(records[b][identity]["wbgt_c"] >= cut for cut in thresholds)
                    result[pair, 0, ci] += 1
                    result[pair, 1, ci] += ca > cb
                    result[pair, 2, ci] += ca < cb
                pair += 1
    return result
