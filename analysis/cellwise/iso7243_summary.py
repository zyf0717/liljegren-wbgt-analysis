"""Reduce hourly classification products to weighted map and table summaries."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from ..grid import area_weights
from .iso7243 import load_hourly_metadata
from .population import load_population_weights
from .projection import country_boundary_from_metadata

POPULATION_AFFECTED_THRESHOLDS_PERCENT = (5, 10, 20, 30)


@dataclass(frozen=True, kw_only=True)
class NonConsensusSummary:
    latitude: np.ndarray
    longitude: np.ndarray
    non_consensus_fraction: np.ndarray
    population: np.ndarray | None
    total_population: int | None
    region_label: str | None
    country_boundary: BaseGeometry | None
    country_iso_a3: str | None = None


@dataclass(frozen=True, kw_only=True)
class ISO7243PlotSummary(NonConsensusSummary):
    changed_fraction: np.ndarray
    histogram_shares: np.ndarray
    implementation_pairs: tuple[tuple[str, str], ...]
    pair_changed_fraction: np.ndarray
    population_affected: np.ndarray


def _load_cells(root: Path):
    metadata = load_hourly_metadata(root)
    lookup = pq.read_table(root / "cell_lookup.parquet")
    counts = pq.read_table(root / "cellwise_non_consensus.parquet")
    if not lookup["cell_id"].equals(counts["cell_id"]):
        raise ValueError("Non-consensus lookup differs")
    return metadata, lookup, counts


def _map_summary(metadata, lookup, counts, population_weights_path):
    joint = np.asarray(counts["n_joint"], dtype=np.int64)
    changed = np.asarray(counts["n_non_consensus"], dtype=np.int64)
    selection = metadata.get("country_selection", {})
    region = selection.get("country_name")
    population, total = (
        _population_weights(
            lookup["source_cell_id"].to_pylist(),
            population_weights_path,
            regional=region is not None,
        )
        if population_weights_path is not None
        else (None, None)
    )
    return NonConsensusSummary(
        latitude=np.asarray(lookup["latitude"]),
        longitude=np.asarray(lookup["longitude"]),
        non_consensus_fraction=np.divide(
            changed, joint, out=np.full(len(joint), np.nan), where=joint > 0
        ),
        population=population,
        total_population=total,
        region_label=region,
        country_iso_a3=selection.get("country_iso_a3"),
        country_boundary=country_boundary_from_metadata(metadata),
    )


def summarize_non_consensus(
    input_root: Path, *, population_weights_path: Path | None = None
):
    """Read only cell counts and optional population, without scanning pair tables."""
    return _map_summary(*_load_cells(Path(input_root)), population_weights_path)


def summarize_cellwise_iso7243_classification(
    input_root: str | Path,
    *,
    population_weights_path: str | Path,
) -> ISO7243PlotSummary:
    """Stream hourly counts into rates and weighted cell-pair distributions."""
    root = Path(input_root).expanduser().resolve()
    metadata, lookup, non_consensus = _load_cells(root)
    maps = _map_summary(metadata, lookup, non_consensus, population_weights_path)
    implementation_pairs = _implementation_pairs(metadata)
    cell_ids = np.asarray(lookup["cell_id"], dtype=np.int64)
    joint = np.asarray(non_consensus["n_joint"], dtype=np.int64)
    non_unanimous = np.asarray(non_consensus["n_non_consensus"], dtype=np.int64)
    source_ids = lookup["source_cell_id"].to_pylist()
    population = maps.population
    weights = np.vstack(
        (
            np.ones(len(cell_ids)),
            area_weights(source_ids),
            population.astype(np.float64),
        )
    )
    population_affected = np.zeros(
        (len(implementation_pairs) + 1, len(POPULATION_AFFECTED_THRESHOLDS_PERCENT)),
        dtype=np.int64,
    )
    _accumulate_population_affected(
        population_affected[-1:],
        np.zeros(len(cell_ids), dtype=np.int64),
        non_unanimous,
        joint,
        population,
    )
    pair_changed_fraction = np.full(
        (len(implementation_pairs), len(cell_ids)),
        np.nan,
        dtype=np.float64,
    )
    pair_index_by_code = _pair_index_by_code(metadata, implementation_pairs)
    implementation_count = len(metadata["implementation_ids"])
    histogram = np.zeros((3, 20), dtype=np.float64)
    totals = np.zeros(3, dtype=np.float64)
    product = pq.ParquetFile(root / "cell_pair_differences.parquet")
    for batch in product.iter_batches(
        columns=[
            "cell_id",
            "n_compared",
            "n_higher",
            "n_lower",
            "implementation_a_id",
            "implementation_b_id",
        ],
        batch_size=250_000,
        use_threads=False,
    ):
        ids, compared, higher, lower, a, b = (
            np.asarray(batch.column(i), dtype=np.int64) for i in range(6)
        )
        pair_indices = pair_index_by_code[a * implementation_count + b]
        changed = higher + lower
        supported = compared > 0
        fractions = np.divide(
            changed, compared, out=np.full(len(ids), np.nan), where=supported
        )
        pair_changed_fraction[pair_indices, ids] = fractions
        _accumulate_population_affected(
            population_affected[:-1],
            pair_indices,
            changed,
            compared,
            population[ids],
        )
        for index in range(3):
            mass = weights[index, ids[supported]]
            histogram[index] += np.histogram(
                100.0 * fractions[supported],
                bins=np.arange(0.0, 105.0, 5.0),
                weights=mass,
            )[0]
            totals[index] += mass.sum()
    supported_pairs = np.isfinite(pair_changed_fraction).sum(axis=0)
    mean = np.divide(
        np.nansum(pair_changed_fraction, axis=0),
        supported_pairs,
        out=np.full(len(cell_ids), np.nan),
        where=supported_pairs > 0,
    )
    return ISO7243PlotSummary(
        **vars(maps),
        changed_fraction=mean,
        histogram_shares=np.divide(
            histogram,
            totals[:, None],
            out=np.zeros_like(histogram),
            where=totals[:, None] > 0,
        ),
        implementation_pairs=implementation_pairs,
        pair_changed_fraction=pair_changed_fraction,
        population_affected=population_affected,
    )


def _implementation_pairs(metadata: dict) -> tuple[tuple[str, str], ...]:
    return tuple(combinations(metadata["implementations"], 2))


def _pair_index_by_code(
    metadata: dict[str, object], implementation_pairs: tuple[tuple[str, str], ...]
) -> np.ndarray:
    ids = metadata["implementation_ids"]
    implementation_count = len(ids)
    result = np.full(implementation_count**2, -1, dtype=np.int32)
    for pair_index, (implementation_a, implementation_b) in enumerate(
        implementation_pairs
    ):
        result[
            int(ids[implementation_a]) * implementation_count
            + int(ids[implementation_b])
        ] = pair_index
    return result


def _population_weights(
    source_ids: list[str], path: str | Path, *, regional: bool = False
) -> tuple[np.ndarray, int]:
    weights, global_population = load_population_weights(source_ids, path)
    if global_population <= 0:
        raise ValueError("Population weights have no positive total population")
    total_population = (
        int(weights.sum(dtype=np.int64)) if regional else global_population
    )
    if total_population <= 0:
        raise ValueError("Analysis grid has no positive population")
    return weights, total_population


def _accumulate_population_affected(
    totals: np.ndarray,
    group_indices: np.ndarray,
    changed: np.ndarray,
    compared: np.ndarray,
    population: np.ndarray,
) -> None:
    """Add each qualifying cell population once per group and threshold."""
    supported = compared > 0
    for column, threshold in enumerate(POPULATION_AFFECTED_THRESHOLDS_PERCENT):
        affected = supported & (100 * changed >= threshold * compared)
        np.add.at(totals[:, column], group_indices[affected], population[affected])
