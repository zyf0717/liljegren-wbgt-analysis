"""Bounded-memory Arrow extraction of country result observations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..io.artifacts import atomic_json, atomic_path
from ..results import ResultFile, date_range, discover_results
from .cells import (
    Country,
    GeographicBounds,
    cached_natural_earth_boundaries,
    country_cells,
    load_countries,
)


@dataclass(frozen=True)
class CountryExtraction:
    schema_version: int
    generated_at_utc: str
    country_iso_a2: str
    country_iso_a3: str
    country_name: str
    selection_bounds: GeographicBounds | None
    start_date: str
    end_date: str
    selected_cell_count: int
    result_row_count: int
    valid_observation_counts: dict[str, int]
    runs: tuple[str, ...]
    parquet_file: str
    parquet_size_bytes: int
    boundary_path: str
    scopes: list[dict]


_OUTPUT_COLUMNS = (
    "result_date",
    "run_id",
    "implementation",
    "cycle_ordinal",
    "row_id",
    "cell_id",
    "row_ordinal",
    "wbgt_status",
    "wbgt_c",
    "tg_status",
    "tg_c",
    "tnwb_status",
    "tnwb_c",
)
_SOURCE_COLUMNS = tuple(
    column
    for column in _OUTPUT_COLUMNS
    if column not in {"result_date", "run_id", "implementation", "cell_id"}
)
_OUTPUT_ROW_GROUP_SIZE = 250_000
_ROUTING_COLUMNS = ["result_date", "implementation", "run_id"]


@dataclass(frozen=True)
class _Selection:
    country: Country
    cells: tuple[str, ...]
    output: Path


class _BufferedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.pending: list[pa.Table] = []
        self.pending_rows = 0
        self.closed = False
        self.row_group = 0
        self.scopes = {}

    def write(self, table: pa.Table) -> None:
        self.pending.append(table)
        self.pending_rows += table.num_rows
        if self.pending_rows >= _OUTPUT_ROW_GROUP_SIZE:
            self._flush_complete_groups()

    def finish(self) -> None:
        if self.closed:
            return
        if self.pending:
            self._write_group(pa.concat_tables(self.pending))
            self.pending.clear()
            self.pending_rows = 0
        self.writer.close()
        self.closed = True

    def abort(self) -> None:
        if not self.closed:
            self.writer.close()
            self.closed = True

    def _flush_complete_groups(self) -> None:
        table = pa.concat_tables(self.pending)
        offset = 0
        while table.num_rows - offset >= _OUTPUT_ROW_GROUP_SIZE:
            self._write_group(table.slice(offset, _OUTPUT_ROW_GROUP_SIZE))
            offset += _OUTPUT_ROW_GROUP_SIZE
        remainder = table.slice(offset)
        self.pending = [remainder] if remainder.num_rows else []
        self.pending_rows = remainder.num_rows

    def _write_group(self, table: pa.Table) -> None:
        self.writer.write_table(table, row_group_size=_OUTPUT_ROW_GROUP_SIZE)
        for row in (
            table.select(_ROUTING_COLUMNS)
            .group_by(_ROUTING_COLUMNS)
            .aggregate([])
            .to_pylist()
        ):
            key = tuple(str(row[name]) for name in _ROUTING_COLUMNS)
            self.scopes.setdefault(key, []).append(self.row_group)
        self.row_group += 1

    def routing(self) -> list[dict]:
        return [
            dict(zip(_ROUTING_COLUMNS, key), row_groups=groups)
            for key, groups in sorted(self.scopes.items())
        ]


def extract_countries_observations(
    country_codes: Sequence[str],
    runs: Sequence[str | Path],
    start_date: date,
    end_date: date,
    output_directory: Path,
    *,
    implementations: Sequence[str] | None = None,
    boundaries: str | Path | None = None,
    boundary_cache: str | Path | None = None,
    overwrite: bool = False,
    bounds: GeographicBounds | None = None,
) -> tuple[CountryExtraction, ...]:
    """Scan selected targets once; omit implementations to extract every target."""
    dates = date_range(start_date, end_date)
    resolved_runs = tuple(Path(run).expanduser().resolve() for run in runs)
    output_directory = Path(output_directory).expanduser().resolve()
    boundary_path = (
        Path(boundaries).expanduser().resolve()
        if boundaries is not None
        else cached_natural_earth_boundaries(Path(boundary_cache))
    )
    countries = load_countries(boundary_path, country_codes)
    if bounds is not None and len(countries) != 1:
        raise ValueError("bounds can only be used with one country")
    selections = tuple(
        _Selection(
            country,
            tuple(cell.cell_id for cell in country_cells(country, bounds=bounds)),
            output_directory / country.iso_a3.lower() / "observations.parquet",
        )
        for country in countries
    )
    if any(not item.cells for item in selections):
        raise ValueError("No ERA5 cells intersect a requested country")
    _validate_outputs(selections, overwrite)
    files = _select_result_files(
        discover_results(resolved_runs, dates, implementations=implementations),
        _merge_ranges(
            (min(item.cells) + "-", max(item.cells) + ".") for item in selections
        ),
    )
    if not files:
        raise ValueError("No source result files overlap the requested countries")
    writers: dict[str, _BufferedParquetWriter] = {}
    counts = {item.country.iso_a3: 0 for item in selections}
    valid = {
        item.country.iso_a3: {quantity: 0 for quantity in ("wbgt", "tg", "tnwb")}
        for item in selections
    }
    cell_sets = {
        item.country.iso_a3: pa.array(item.cells, type=pa.string())
        for item in selections
    }
    selected = 0
    with ExitStack() as pending:
        temporary = {
            item.country.iso_a3: pending.enter_context(atomic_path(item.output))
            for item in selections
        }
        try:
            for index, (item, row_groups) in enumerate(files, 1):
                reader = pq.ParquetFile(item.path)
                result_date, run_id, implementation = (
                    item.result_date,
                    item.run_id,
                    item.implementation,
                )
                for batch in reader.iter_batches(
                    columns=list(_SOURCE_COLUMNS),
                    row_groups=row_groups,
                    batch_size=65_536,
                ):
                    source = pa.Table.from_batches([batch])
                    cell_ids = pc.utf8_slice_codeunits(
                        source["row_id"], start=0, stop=28
                    )
                    for iso_a3, value_set in cell_sets.items():
                        mask = pc.is_in(cell_ids, value_set=value_set)
                        rows = int(pc.sum(mask).as_py() or 0)
                        if not rows:
                            continue
                        taken = source.filter(mask)
                        table = pa.table(
                            {
                                "result_date": pa.array(
                                    [result_date] * rows,
                                    type=pa.date32(),
                                ),
                                "run_id": pa.array([run_id] * rows),
                                "implementation": pa.array([implementation] * rows),
                                "cycle_ordinal": taken["cycle_ordinal"],
                                "row_id": taken["row_id"],
                                "cell_id": cell_ids.filter(mask),
                                "row_ordinal": taken["row_ordinal"],
                                "wbgt_status": taken["wbgt_status"],
                                "wbgt_c": taken["wbgt_c"],
                                "tg_status": taken["tg_status"],
                                "tg_c": taken["tg_c"],
                                "tnwb_status": taken["tnwb_status"],
                                "tnwb_c": taken["tnwb_c"],
                            }
                        ).select(_OUTPUT_COLUMNS)
                        if iso_a3 not in writers:
                            writers[iso_a3] = _BufferedParquetWriter(
                                temporary[iso_a3], table.schema
                            )
                        writers[iso_a3].write(table)
                        counts[iso_a3] += table.num_rows
                        selected += table.num_rows
                        for quantity in valid[iso_a3]:
                            valid[iso_a3][quantity] += int(
                                pc.sum(
                                    pc.and_(
                                        pc.equal(table[f"{quantity}_status"], "ok"),
                                        pc.is_valid(table[f"{quantity}_c"]),
                                    )
                                ).as_py()
                                or 0
                            )
                print(
                    f"Country extraction: {index}/{len(files)} files, {selected:,} rows selected",
                    flush=True,
                )
            missing = [
                item.country.name
                for item in selections
                if not counts[item.country.iso_a3]
            ]
            if missing:
                raise ValueError(
                    f"No result rows found for {', '.join(missing)} from {start_date} to {end_date}"
                )
            for writer in writers.values():
                writer.finish()
            result = []
            for item in selections:
                iso_a3 = item.country.iso_a3
                item.output.parent.mkdir(parents=True, exist_ok=True)
                extraction = CountryExtraction(
                    4,
                    datetime.now(UTC).isoformat(),
                    item.country.iso_a2,
                    iso_a3,
                    item.country.name,
                    bounds,
                    dates[0].isoformat(),
                    dates[-1].isoformat(),
                    len(item.cells),
                    counts[iso_a3],
                    valid[iso_a3],
                    tuple(map(str, resolved_runs)),
                    item.output.name,
                    temporary[iso_a3].stat().st_size,
                    str(boundary_path),
                    writers[iso_a3].routing(),
                )
                result.append(extraction)
            for item in selections:
                item.output.with_name("manifest.json").unlink(missing_ok=True)
        finally:
            for writer in writers.values():
                writer.abort()
    for item, extraction in zip(selections, result, strict=True):
        atomic_json(asdict(extraction), item.output.with_name("manifest.json"))
    return tuple(result)


def _validate_outputs(selections: Sequence[_Selection], overwrite: bool) -> None:
    if not overwrite:
        for item in selections:
            if item.output.exists() or item.output.with_name("manifest.json").exists():
                raise FileExistsError(
                    f"Country cache already exists: {item.output}; pass overwrite=True to replace it"
                )


def _merge_ranges(ranges: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    merged: list[tuple[str, str]] = []
    for lower, upper in sorted(ranges):
        if merged and lower <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], upper)
        else:
            merged.append((lower, upper))
    return tuple(merged)


def _overlapping_row_groups(
    path: Path, ranges: Sequence[tuple[str, str]]
) -> tuple[int, ...]:
    metadata = pq.read_metadata(path)
    if "row_id" not in metadata.schema.names:
        return tuple(range(metadata.num_row_groups))
    row = metadata.schema.names.index("row_id")
    selected: list[int] = []
    for index in range(metadata.num_row_groups):
        stats = metadata.row_group(index).column(row).statistics
        if stats is None or not stats.has_min_max:
            selected.append(index)
            continue
        minimum, maximum = _statistic_text(stats.min), _statistic_text(stats.max)
        if minimum is None or maximum is None:
            selected.append(index)
            continue
        if any(maximum >= lower and minimum < upper for lower, upper in ranges):
            selected.append(index)
    return tuple(selected)


def _select_result_files(
    files: Sequence[ResultFile], ranges: Sequence[tuple[str, str]]
) -> tuple[tuple[ResultFile, tuple[int, ...]], ...]:
    selected = ((item, _overlapping_row_groups(item.path, ranges)) for item in files)
    return tuple((path, row_groups) for path, row_groups in selected if row_groups)


def _statistic_text(value: object) -> str | None:
    return (
        value
        if isinstance(value, str)
        else value.decode()
        if isinstance(value, bytes)
        else None
    )
