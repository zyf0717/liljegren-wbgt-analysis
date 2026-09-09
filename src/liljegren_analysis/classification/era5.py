"""Bounded-memory ERA5 country-cache classification analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .core import (
    ClassificationAnalysis,
    class_definitions,
    classification_implementations,
    validate_thresholds,
)


_CACHE_COLUMNS = (
    "result_date",
    "run_id",
    "implementation",
    "cycle_ordinal",
    "row_id",
    "cell_id",
    "wbgt_status",
    "wbgt_c",
    "tg_status",
    "tg_c",
    "tnwb_status",
    "tnwb_c",
)
_BATCH_SIZE = 250_000


@dataclass(frozen=True)
class Era5ClassificationAnalysis:
    classification: ClassificationAnalysis
    observation_summary: pd.DataFrame
    quantity_coverage: pd.DataFrame

    @property
    def outputs(self) -> dict[str, pd.DataFrame]:
        return {
            "observation_summary.csv": self.observation_summary,
            "quantity_coverage.csv": self.quantity_coverage,
            **self.classification.outputs,
        }


def analyze_era5_country_cache(
    cache_path: str | Path,
    start_date: date,
    end_date: date,
    thresholds_c: Sequence[float],
    *,
    implementations: Sequence[str] | None = None,
) -> Era5ClassificationAnalysis:
    """Stream one preselected country cache and retain aggregate state only."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"country cache does not exist: {path}")
    thresholds = validate_thresholds(thresholds_c)
    selected = classification_implementations(implementations)
    reader = pq.ParquetFile(path)
    missing_columns = sorted(set(_CACHE_COLUMNS) - set(reader.schema_arrow.names))
    if missing_columns:
        raise ValueError(f"country cache is missing columns: {missing_columns}")

    accumulator = _CountryAccumulator(selected, thresholds)
    selected_values = pa.array(selected, type=pa.string())
    lower = pa.scalar(start_date, type=pa.date32())
    upper = pa.scalar(end_date, type=pa.date32())
    for batch in reader.iter_batches(
        columns=list(_CACHE_COLUMNS),
        batch_size=_BATCH_SIZE,
        use_threads=False,
    ):
        table = pa.Table.from_batches([batch])
        dates = table["result_date"]
        in_range = pc.fill_null(
            pc.and_(pc.greater_equal(dates, lower), pc.less_equal(dates, upper)),
            False,
        )
        selected_rows = pc.fill_null(
            pc.is_in(table["implementation"], value_set=selected_values), False
        )
        table = table.filter(pc.and_(in_range, selected_rows))
        if table.num_rows:
            accumulator.consume(table, selected_values)

    return accumulator.finish()


class _CountryAccumulator:
    def __init__(self, implementations: tuple[str, ...], thresholds: tuple[float, ...]):
        self.implementations = implementations
        self.thresholds = thresholds
        self.pairs = tuple(combinations(sorted(implementations), 2))
        self.implementation_index = {
            implementation: index for index, implementation in enumerate(implementations)
        }
        self.current_date: date | None = None
        self.identity_ids: dict[tuple[str, int, str, str], int] = {}
        self.duplicate_keys = [set() for _ in implementations]
        self.daily_classes = [dict() for _ in implementations]
        self.pair_counts: dict[tuple[str, str, str], np.ndarray] = {}
        self.transition_counts: dict[tuple[str, str, str, int, int], int] = {}
        self.non_consensus_counts: dict[str, np.ndarray] = {}
        self.coverage_counts: dict[tuple[str, str, str, bool], int] = {}
        self.implementations_seen: set[str] = set()
        self.valid_implementations: set[str] = set()
        self.cells: set[str] = set()
        self.result_rows = 0
        self.valid_wbgt_observations = 0
        self.first_date: date | None = None
        self.last_date: date | None = None

    def consume(self, table: pa.Table, selected_values: pa.Array) -> None:
        date_ids = pc.cast(table["result_date"], pa.int32()).to_numpy(
            zero_copy_only=False
        )
        boundaries = np.flatnonzero(date_ids[1:] != date_ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [table.num_rows]))
        for start, stop in zip(starts, stops, strict=True):
            part = table.slice(int(start), int(stop - start))
            current_date = part["result_date"][0].as_py()
            if self.current_date is not None and current_date != self.current_date:
                if current_date < self.current_date:
                    raise ValueError("country cache result_date rows are not ordered")
                self._flush_day()
            if self.current_date is None:
                self.current_date = current_date
            self._consume_day_rows(part, selected_values)

    def _consume_day_rows(self, table: pa.Table, selected_values: pa.Array) -> None:
        implementation_ids = pc.index_in(
            table["implementation"], value_set=selected_values
        ).to_numpy(zero_copy_only=False)
        implementations = table["implementation"].to_pylist()
        run_ids = table["run_id"].to_pylist()
        cycles = table["cycle_ordinal"].to_numpy(zero_copy_only=False)
        row_ids = table["row_id"].to_pylist()
        cell_ids = table["cell_id"].to_pylist()
        valid_wbgt = _valid(table, "wbgt")
        values = table["wbgt_c"].to_numpy(zero_copy_only=False)
        classes = np.full(table.num_rows, -1, dtype=np.int16)
        classes[valid_wbgt] = np.searchsorted(
            self.thresholds, values[valid_wbgt], side="right"
        ).astype(np.int16)

        for (
            implementation_id,
            implementation,
            run_id,
            cycle,
            row_id,
            cell_id,
            class_id,
        ) in zip(
            implementation_ids,
            implementations,
            run_ids,
            cycles,
            row_ids,
            cell_ids,
            classes,
            strict=True,
        ):
            index = int(implementation_id)
            duplicate_key = (run_id, int(cycle), row_id)
            if duplicate_key in self.duplicate_keys[index]:
                raise ValueError("country cache contains duplicate result rows")
            self.duplicate_keys[index].add(duplicate_key)
            identity_key = (*duplicate_key, cell_id)
            identity_id = self.identity_ids.setdefault(identity_key, len(self.identity_ids))
            self.daily_classes[index][identity_id] = int(class_id)

        self.result_rows += table.num_rows
        self.valid_wbgt_observations += int(valid_wbgt.sum())
        self.implementations_seen.update(implementations)
        self.valid_implementations.update(
            implementation
            for implementation, valid in zip(implementations, valid_wbgt, strict=True)
            if valid
        )
        self.cells.update(pc.unique(table["cell_id"]).to_pylist())
        current_date = self.current_date
        assert current_date is not None
        self.first_date = current_date if self.first_date is None else self.first_date
        self.last_date = current_date
        for quantity in ("wbgt", "tg", "tnwb"):
            self._count_coverage(table, quantity, implementation_ids)

    def _count_coverage(
        self, table: pa.Table, quantity: str, implementation_ids: np.ndarray
    ) -> None:
        statuses = pc.dictionary_encode(table[f"{quantity}_status"].combine_chunks())
        codes = statuses.indices.to_numpy(zero_copy_only=False)
        present = _valid(table, quantity)
        usable = ~statuses.is_null().to_numpy(zero_copy_only=False)
        if not usable.any():
            return
        width = len(statuses.dictionary) * 2
        packed = (
            implementation_ids[usable] * width
            + codes[usable] * 2
            + present[usable].astype(np.int8)
        )
        keys, counts = np.unique(packed, return_counts=True)
        dictionary = statuses.dictionary.to_pylist()
        for key, count in zip(keys, counts, strict=True):
            implementation_id, remainder = divmod(int(key), width)
            status_id, value_present = divmod(remainder, 2)
            group = (
                quantity,
                self.implementations[implementation_id],
                dictionary[status_id],
                bool(value_present),
            )
            self.coverage_counts[group] = self.coverage_counts.get(group, 0) + int(count)

    def _flush_day(self) -> None:
        if self.current_date is None:
            return
        size = len(self.identity_ids)
        matrix = np.full((len(self.implementations), size), -1, dtype=np.int16)
        for implementation_id, values in enumerate(self.daily_classes):
            if values:
                indices = np.fromiter(values, dtype=np.int64, count=len(values))
                matrix[implementation_id, indices] = np.fromiter(
                    values.values(), dtype=np.int16, count=len(values)
                )
        month = self.current_date.strftime("%Y-%m")
        class_count = len(self.thresholds) + 1
        for implementation_a, implementation_b in self.pairs:
            a = matrix[self.implementation_index[implementation_a]]
            b = matrix[self.implementation_index[implementation_b]]
            supported = (a >= 0) & (b >= 0)
            if not supported.any():
                continue
            a_values = a[supported]
            b_values = b[supported]
            higher = int(np.count_nonzero(a_values > b_values))
            lower = int(np.count_nonzero(a_values < b_values))
            key = (month, implementation_a, implementation_b)
            counts = self.pair_counts.setdefault(key, np.zeros(4, dtype=np.uint64))
            counts += np.array(
                (len(a_values), higher + lower, higher, lower), dtype=np.uint64
            )
            transitions = np.bincount(
                a_values * class_count + b_values,
                minlength=class_count * class_count,
            )
            for transition, count in enumerate(transitions):
                if count:
                    class_a, class_b = divmod(transition, class_count)
                    transition_key = (
                        month,
                        implementation_a,
                        implementation_b,
                        class_a,
                        class_b,
                    )
                    self.transition_counts[transition_key] = (
                        self.transition_counts.get(transition_key, 0) + int(count)
                    )
        joint = np.all(matrix >= 0, axis=0)
        if joint.any():
            joint_classes = matrix[:, joint]
            non_consensus = int(
                np.count_nonzero(joint_classes.max(axis=0) != joint_classes.min(axis=0))
            )
            counts = self.non_consensus_counts.setdefault(
                month, np.zeros(2, dtype=np.uint64)
            )
            counts += np.array((int(joint.sum()), non_consensus), dtype=np.uint64)
        self.current_date = None
        self.identity_ids.clear()
        for values in self.daily_classes:
            values.clear()
        for keys in self.duplicate_keys:
            keys.clear()

    def finish(self) -> Era5ClassificationAnalysis:
        self._flush_day()
        if not self.result_rows:
            raise ValueError("country cache has no rows in the requested date range")
        missing = sorted(set(self.implementations) - self.implementations_seen)
        missing_valid = sorted(set(self.implementations) - self.valid_implementations)
        if missing or missing_valid:
            raise ValueError(
                f"classification implementations are missing: {missing or missing_valid}"
            )
        if not self.pair_counts:
            raise ValueError("fewer than two implementations have matched observations")

        classes = class_definitions(self.thresholds)
        outputs = {
            "classes.csv": classes,
            "pair_change_summary.csv": _change_summary(self.pair_counts, monthly=False),
            "pair_transition_counts.csv": _transition_summary(
                self.transition_counts, classes, monthly=False
            ),
            "monthly_pair_change_summary.csv": _change_summary(
                self.pair_counts, monthly=True
            ),
            "monthly_pair_transition_counts.csv": _transition_summary(
                self.transition_counts, classes, monthly=True
            ),
            "non_consensus_summary.csv": _non_consensus_summary(
                self.non_consensus_counts, monthly=False
            ),
            "monthly_non_consensus_summary.csv": _non_consensus_summary(
                self.non_consensus_counts, monthly=True
            ),
        }
        classification = ClassificationAnalysis(classes, None, None, outputs)
        observation_summary = pd.DataFrame(
            {
                "result_rows": [self.result_rows],
                "valid_wbgt_observations": [self.valid_wbgt_observations],
                "implementations": [len(self.implementations_seen)],
                "cells": [len(self.cells)],
                "first_date": [self.first_date],
                "last_date": [self.last_date],
            }
        )
        coverage = pd.DataFrame(
            [(*key, count) for key, count in sorted(self.coverage_counts.items())],
            columns=[
                "quantity",
                "implementation",
                "quantity_status",
                "quantity_value_present",
                "n",
            ],
        )
        return Era5ClassificationAnalysis(classification, observation_summary, coverage)


def _valid(table: pa.Table, quantity: str) -> np.ndarray:
    return pc.fill_null(
        pc.and_(
            pc.equal(table[f"{quantity}_status"], "ok"),
            pc.is_finite(table[f"{quantity}_c"]),
        ),
        False,
    ).to_numpy(zero_copy_only=False)


def _change_summary(
    monthly_counts: dict[tuple[str, str, str], np.ndarray], *, monthly: bool
) -> pd.DataFrame:
    counts = monthly_counts if monthly else _sum_without_month(monthly_counts)
    rows = []
    for key, values in sorted(counts.items()):
        n_paired, n_changed, a_higher, b_higher = map(int, values)
        rows.append(
            (
                *key,
                n_paired,
                n_changed,
                round(100.0 * n_changed / n_paired, 4),
                a_higher,
                b_higher,
            )
        )
    groups = ["result_month"] if monthly else []
    return pd.DataFrame(
        rows,
        columns=[
            *groups,
            "implementation_a",
            "implementation_b",
            "n_paired",
            "n_changed",
            "percent_changed",
            "a_higher_class",
            "b_higher_class",
        ],
    )


def _transition_summary(
    monthly_counts: dict[tuple[str, str, str, int, int], int],
    classes: pd.DataFrame,
    *,
    monthly: bool,
) -> pd.DataFrame:
    counts = monthly_counts if monthly else _sum_without_month(monthly_counts)
    labels = classes.set_index("class_id")["class_label"]
    rows = []
    for key, count in sorted(counts.items()):
        *groups, class_a, class_b = key
        rows.append(
            (
                *groups,
                class_a,
                class_b,
                count,
                labels[class_a],
                labels[class_b],
                class_a != class_b,
            )
        )
    groups = ["result_month"] if monthly else []
    return pd.DataFrame(
        rows,
        columns=[
            *groups,
            "implementation_a",
            "implementation_b",
            "class_id_a",
            "class_id_b",
            "n",
            "class_a_label",
            "class_b_label",
            "classification_changed",
        ],
    )


def _non_consensus_summary(
    monthly_counts: dict[str, np.ndarray], *, monthly: bool
) -> pd.DataFrame:
    if monthly:
        items = sorted(monthly_counts.items())
    else:
        total = sum(monthly_counts.values(), start=np.zeros(2, dtype=np.uint64))
        items = [(None, total)]
    rows = []
    for month, values in items:
        n_joint, n_non_consensus = map(int, values)
        percent = round(100.0 * n_non_consensus / n_joint, 4) if n_joint else np.nan
        row = (n_joint, n_non_consensus, percent)
        rows.append((month, *row) if monthly else row)
    groups = ["result_month"] if monthly else []
    return pd.DataFrame(
        rows,
        columns=[*groups, "n_joint", "n_non_consensus", "non_consensus_percent"],
    )


def _sum_without_month(
    counts: dict[tuple, int | np.ndarray],
) -> dict[tuple, int | np.ndarray]:
    totals: dict[tuple, int | np.ndarray] = {}
    for key, value in counts.items():
        group = key[1:]
        if group not in totals:
            totals[group] = value.copy() if isinstance(value, np.ndarray) else value
        else:
            totals[group] += value
    return totals
