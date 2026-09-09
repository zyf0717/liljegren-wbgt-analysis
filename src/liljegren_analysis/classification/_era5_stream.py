"""Daily dense-array accumulator for ERA5 country classification caches."""

from __future__ import annotations

from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from ._summary_tables import (
    change_summary,
    non_consensus_summary,
    transition_summary,
)
from .core import ClassificationAnalysis, class_definitions


class CountryAccumulator:
    """Reduce date-ordered cache batches without retaining observation rows."""

    def __init__(self, implementations: tuple[str, ...], thresholds: tuple[float, ...]):
        self.implementations = implementations
        self.thresholds = thresholds
        self.pairs = tuple(combinations(sorted(implementations), 2))
        self.implementation_index = {
            implementation: index
            for index, implementation in enumerate(implementations)
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
        """Consume selected rows, flushing aggregate state at date boundaries."""

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
        valid_wbgt = valid_quantity(table, "wbgt")
        values = table["wbgt_c"].to_numpy(zero_copy_only=False)
        classes = np.full(table.num_rows, -1, dtype=np.int16)
        classes[valid_wbgt] = np.searchsorted(
            self.thresholds, values[valid_wbgt], side="right"
        ).astype(np.int16)

        rows = zip(
            implementation_ids,
            implementations,
            run_ids,
            cycles,
            row_ids,
            cell_ids,
            classes,
            strict=True,
        )
        for (
            implementation_id,
            implementation,
            run_id,
            cycle,
            row_id,
            cell_id,
            class_id,
        ) in rows:
            index = int(implementation_id)
            duplicate_key = (run_id, int(cycle), row_id)
            if duplicate_key in self.duplicate_keys[index]:
                raise ValueError("country cache contains duplicate result rows")
            self.duplicate_keys[index].add(duplicate_key)
            identity_key = (*duplicate_key, cell_id)
            identity_id = self.identity_ids.setdefault(
                identity_key, len(self.identity_ids)
            )
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
        present = valid_quantity(table, quantity)
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
            self.coverage_counts[group] = self.coverage_counts.get(group, 0) + int(
                count
            )

    def _flush_day(self) -> None:
        if self.current_date is None:
            return
        matrix = self._daily_class_matrix()
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
                    self.transition_counts[transition_key] = self.transition_counts.get(
                        transition_key, 0
                    ) + int(count)
        self._count_non_consensus(matrix, month)
        self._reset_day()

    def _daily_class_matrix(self) -> np.ndarray:
        matrix = np.full(
            (len(self.implementations), len(self.identity_ids)), -1, dtype=np.int16
        )
        for implementation_id, values in enumerate(self.daily_classes):
            if values:
                indices = np.fromiter(values, dtype=np.int64, count=len(values))
                matrix[implementation_id, indices] = np.fromiter(
                    values.values(), dtype=np.int16, count=len(values)
                )
        return matrix

    def _count_non_consensus(self, matrix: np.ndarray, month: str) -> None:
        joint = np.all(matrix >= 0, axis=0)
        if not joint.any():
            return
        joint_classes = matrix[:, joint]
        non_consensus = int(
            np.count_nonzero(joint_classes.max(axis=0) != joint_classes.min(axis=0))
        )
        counts = self.non_consensus_counts.setdefault(
            month, np.zeros(2, dtype=np.uint64)
        )
        counts += np.array((int(joint.sum()), non_consensus), dtype=np.uint64)

    def _reset_day(self) -> None:
        self.current_date = None
        self.identity_ids.clear()
        for values in self.daily_classes:
            values.clear()
        for keys in self.duplicate_keys:
            keys.clear()

    def finish(
        self,
    ) -> tuple[ClassificationAnalysis, pd.DataFrame, pd.DataFrame]:
        """Flush and render all aggregate tables."""

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
            "pair_change_summary.csv": change_summary(self.pair_counts, monthly=False),
            "pair_transition_counts.csv": transition_summary(
                self.transition_counts, classes, monthly=False
            ),
            "monthly_pair_change_summary.csv": change_summary(
                self.pair_counts, monthly=True
            ),
            "monthly_pair_transition_counts.csv": transition_summary(
                self.transition_counts, classes, monthly=True
            ),
            "non_consensus_summary.csv": non_consensus_summary(
                self.non_consensus_counts, monthly=False
            ),
            "monthly_non_consensus_summary.csv": non_consensus_summary(
                self.non_consensus_counts, monthly=True
            ),
        }
        classification = ClassificationAnalysis(classes, None, None, outputs)
        return classification, self._observation_summary(), self._coverage_summary()

    def _observation_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "result_rows": [self.result_rows],
                "valid_wbgt_observations": [self.valid_wbgt_observations],
                "implementations": [len(self.implementations_seen)],
                "cells": [len(self.cells)],
                "first_date": [self.first_date],
                "last_date": [self.last_date],
            }
        )

    def _coverage_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(*key, count) for key, count in sorted(self.coverage_counts.items())],
            columns=[
                "quantity",
                "implementation",
                "quantity_status",
                "quantity_value_present",
                "n",
            ],
        )


def valid_quantity(table: pa.Table, quantity: str) -> np.ndarray:
    """Return finite successful values for one cached result quantity."""

    return pc.fill_null(
        pc.and_(
            pc.equal(table[f"{quantity}_status"], "ok"),
            pc.is_finite(table[f"{quantity}_c"]),
        ),
        False,
    ).to_numpy(zero_copy_only=False)
