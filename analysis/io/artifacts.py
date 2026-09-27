"""Atomic writers shared by analysis artifact producers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq


@contextmanager
def atomic_path(path: Path):
    """Publish a sibling temporary file only after its writer closes successfully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_parquet(table: pa.Table, path: Path) -> None:
    with atomic_path(path) as temporary:
        pq.write_table(table, temporary, compression="zstd")


def atomic_csv(table: pa.Table, path: Path) -> None:
    with atomic_path(path) as temporary:
        pcsv.write_csv(table, temporary)


def atomic_json(value: dict, path: Path) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
        )


def save_settings(
    settings: dict, output_root: Path, *, filename: str = "settings.json"
) -> None:
    """Record the effective arguments used by a script or notebook."""
    atomic_json(settings, Path(output_root) / filename)


def write_tables(tables: dict, output_root: Path) -> None:
    """Write the named pandas summary tables produced by classification."""
    for filename, frame in tables.items():
        atomic_csv(
            pa.Table.from_pandas(frame, preserve_index=False),
            Path(output_root) / filename,
        )


def clear_outputs(root: Path, names: tuple[str, ...]) -> None:
    """Remove only the listed workflow artifacts when overwrite is requested."""
    for name in names:
        path = root / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def input_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.as_posix().encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()
