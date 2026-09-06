"""Discover and validate HeatStressBench result Parquet files."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

import pyarrow as pa
import pyarrow.parquet as pq


ResultQuantity = Literal["wbgt", "tg", "tnwb"]
RESULT_QUANTITIES: tuple[ResultQuantity, ...] = ("wbgt", "tg", "tnwb")


@dataclass(frozen=True)
class ResultQuantitySpec:
    """Physical columns and presentation metadata for one result quantity."""

    status_column: str
    value_column: str
    label: str
    unit: str = "°C"


RESULT_QUANTITY_SPECS: Mapping[ResultQuantity, ResultQuantitySpec] = MappingProxyType(
    {
        "wbgt": ResultQuantitySpec("wbgt_status", "wbgt_c", "WBGT"),
        "tg": ResultQuantitySpec("tg_status", "tg_c", "Globe temperature"),
        "tnwb": ResultQuantitySpec(
            "tnwb_status", "tnwb_c", "Natural wet-bulb temperature"
        ),
    }
)
RESULT_SCHEMA_ID = "hsb-wbgt-comparison-results-parquet-float32-v1"
_IDENTITY_COLUMNS = ("cycle_ordinal", "row_id", "row_ordinal")


@dataclass(frozen=True)
class ResultFile:
    """One result shard with provenance derived from its filesystem location."""

    path: Path
    result_date: date
    run_id: str
    implementation: str


class HSBRunResults:
    """Resolve ``results/<date>/<run>/<implementation>/results.parquet`` files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"results directory does not exist: {self.root}")

    @classmethod
    def from_hsb_run(cls, hsb_run: str | Path) -> Self:
        run = Path(hsb_run).expanduser().resolve()
        manifest_path = run / "run-manifest.json"
        if manifest_path.is_file():
            _validate_run_manifest(manifest_path)
        return cls(run / "results")

    def files(
        self,
        *,
        result_date: str | date | None = None,
        dates: Sequence[str | date] | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        run_id: str | int | None = None,
        run_ids: Sequence[str | int] | None = None,
        implementation: str | None = None,
        implementations: Sequence[str] | None = None,
    ) -> tuple[ResultFile, ...]:
        """Return deterministically ordered, schema-validated result files."""

        date_parts = _date_parts(result_date, dates, start_date, end_date)
        run_parts = _parts(run_id, run_ids, "run_id", _run_part)
        implementation_parts = _parts(
            implementation, implementations, "implementation", str
        )
        paths = sorted(
            {
                path.resolve()
                for date_part in date_parts
                for run_part in run_parts
                for implementation_part in implementation_parts
                for path in self.root.glob(
                    f"{date_part}/{run_part}/{implementation_part}/results.parquet"
                )
                if path.is_file()
            }
        )
        result = tuple(_result_file(path, self.root) for path in paths)
        for item in result:
            validate_result_schema(pq.read_schema(item.path))
        return result


def _result_file(path: Path, root: Path) -> ResultFile:
    relative = path.relative_to(root)
    if len(relative.parts) != 4 or relative.name != "results.parquet":
        raise ValueError(f"Invalid result file path: {path}")
    result_date, run_id, implementation, _ = relative.parts
    return ResultFile(path, date.fromisoformat(result_date), run_id, implementation)


def validate_result_schema(schema: pa.Schema) -> None:
    """Validate the HSB result-file schema without reading observations."""

    fields = {field.name: field for field in schema}
    required = {
        *_IDENTITY_COLUMNS,
        *(
            column
            for spec in RESULT_QUANTITY_SPECS.values()
            for column in (spec.status_column, spec.value_column)
        ),
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(
            f"Unsupported result schema {RESULT_SCHEMA_ID}: "
            f"missing columns {sorted(missing)}"
        )
    invalid_status = {
        spec.status_column: str(fields[spec.status_column].type)
        for spec in RESULT_QUANTITY_SPECS.values()
        if not pa.types.is_string(fields[spec.status_column].type)
    }
    invalid_values = {
        spec.value_column: str(fields[spec.value_column].type)
        for spec in RESULT_QUANTITY_SPECS.values()
        if not pa.types.is_floating(fields[spec.value_column].type)
        and not pa.types.is_integer(fields[spec.value_column].type)
        and not pa.types.is_decimal(fields[spec.value_column].type)
    }
    if invalid_status or invalid_values:
        raise ValueError(
            f"Unsupported result schema {RESULT_SCHEMA_ID}: "
            f"invalid status types={invalid_status}, value types={invalid_values}"
        )


@cache
def _validate_run_manifest(path: Path) -> None:
    try:
        manifest = json.loads(path.read_text())
        execution = manifest["execution"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid HSB run manifest: {path}") from error
    if (
        not isinstance(execution, dict)
        or execution.get("result_format") != "parquet"
        or execution.get("result_schema_id") != RESULT_SCHEMA_ID
    ):
        raise ValueError(
            f"Unsupported HSB run result contract in {path}: "
            f"format={execution.get('result_format') if isinstance(execution, dict) else None}, "
            f"schema_id={execution.get('result_schema_id') if isinstance(execution, dict) else None}"
        )


def _date_parts(result_date: str | date | None, dates: Sequence[str | date] | None, start_date: str | date | None, end_date: str | date | None) -> list[str]:
    modes = sum((result_date is not None, dates is not None, start_date is not None or end_date is not None))
    if modes > 1:
        raise ValueError("use only one of result_date, dates, or start_date/end_date")
    if result_date is not None:
        return [_as_date(result_date).isoformat()]
    if dates is not None:
        return _many(dates, "dates", lambda value: _as_date(value).isoformat())
    if (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be provided together")
    if start_date is None:
        return ["*"]
    start, end = _as_date(start_date), _as_date(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    return [(start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)]


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _run_part(value: str | int) -> str:
    return f"{value:04d}" if isinstance(value, int) else value


def _parts[T](value: T | None, values: Sequence[T] | None, name: str, render: Callable[[T], str]) -> list[str]:
    if value is not None and values is not None:
        raise ValueError(f"{name} and {name}s cannot be combined")
    if values is not None:
        return _many(values, f"{name}s", render)
    return [render(value)] if value is not None else ["*"]


def _many[T](values: Sequence[T], name: str, render: Callable[[T], str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not a string")
    rendered = list(dict.fromkeys(render(value) for value in values))
    if not rendered:
        raise ValueError(f"{name} must not be empty")
    return rendered
