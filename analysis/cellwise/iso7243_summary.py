"""Reduce hourly classification products to weighted map and table summaries."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from ..config import DEFAULT_ISO7243_OUTPUT_ROOT
from ..grid import area_weights
from .iso7243 import load_hourly_metadata
from .population import DEFAULT_POPULATION_WEIGHTS_PATH, load_population_weights
from .projection import country_boundary_from_metadata

POPULATION_AFFECTED_THRESHOLDS_PERCENT = (5, 10, 20, 30)


@dataclass(frozen=True)
class ISO7243PlotSummary:
    """Compact aggregates used by the overview figure."""

    latitude: np.ndarray
    longitude: np.ndarray
    changed_fraction: np.ndarray
    non_consensus_fraction: np.ndarray
    population: np.ndarray
    histogram_shares: np.ndarray
    implementation_pairs: tuple[tuple[str, str], ...]
    pair_changed_fraction: np.ndarray
    population_affected: np.ndarray
    total_population: int
    region_label: str | None
    country_boundary: BaseGeometry | None
    country_iso_a3: str | None = None


def summarize_cellwise_iso7243_classification(
    input_root: str | Path = DEFAULT_ISO7243_OUTPUT_ROOT,
    *,
    population_weights_path: str | Path = DEFAULT_POPULATION_WEIGHTS_PATH,
) -> ISO7243PlotSummary:
    """Stream hourly counts into rates and weighted cell-pair distributions."""
    root = Path(input_root).expanduser().resolve()
    metadata = load_hourly_metadata(root)
    implementation_pairs = _implementation_pairs(metadata)
    lookup = pq.read_table(root / "cell_lookup.parquet")
    cell_ids = np.asarray(lookup["cell_id"], dtype=np.int64)
    non_consensus = pq.read_table(root / "cellwise_non_consensus.parquet")
    if not np.array_equal(
        np.asarray(non_consensus["cell_id"], dtype=np.int64), cell_ids
    ):
        raise ValueError("Non-consensus lookup differs")
    joint = np.asarray(non_consensus["n_joint"], dtype=np.int64)
    non_unanimous = np.asarray(non_consensus["n_non_consensus"], dtype=np.int64)
    source_ids = lookup["source_cell_id"].combine_chunks().to_pylist()
    country_selection = metadata.get("country_selection", {})
    region_label = country_selection.get("country_name")
    raw_country_iso_a3 = country_selection.get("country_iso_a3")
    population, total_population = _population_weights(
        source_ids, population_weights_path, regional=region_label is not None
    )
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
        latitude=np.asarray(lookup["latitude"], dtype=np.float64),
        longitude=np.asarray(lookup["longitude"], dtype=np.float64),
        changed_fraction=mean,
        non_consensus_fraction=np.divide(
            non_unanimous, joint, out=np.full(len(cell_ids), np.nan), where=joint > 0
        ),
        population=population,
        histogram_shares=np.divide(
            histogram,
            totals[:, None],
            out=np.zeros_like(histogram),
            where=totals[:, None] > 0,
        ),
        implementation_pairs=implementation_pairs,
        pair_changed_fraction=pair_changed_fraction,
        population_affected=population_affected,
        total_population=total_population,
        region_label=region_label,
        country_boundary=country_boundary_from_metadata(metadata),
        country_iso_a3=raw_country_iso_a3,
    )


def _implementation_pairs(metadata: dict[str, object]) -> tuple[tuple[str, str], ...]:
    ids = metadata["implementation_ids"]
    implementations = tuple(
        name for name, _ in sorted(ids.items(), key=lambda item: item[1])
    )
    pairs = tuple(
        (implementations[implementation_a], implementations[implementation_b])
        for implementation_a in range(len(implementations))
        for implementation_b in range(implementation_a + 1, len(implementations))
    )
    return pairs


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
