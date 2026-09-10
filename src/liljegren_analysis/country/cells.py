"""Select ERA5 grid cells intersecting country boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep


ERA5_SPACING_UNITS = 2_500
_COORDINATE_SCALE = 10_000
_HALF_SPACING_DEGREES = ERA5_SPACING_UNITS / _COORDINATE_SCALE / 2
GeographicBounds = tuple[float, float, float, float]
NATURAL_EARTH_VERSION = "5.1.2"
NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    f"v{NATURAL_EARTH_VERSION}/geojson/ne_10m_admin_0_countries.geojson"
)
NATURAL_EARTH_SHA256 = (
    "239eec57ac17f100a11e2536cffc56752c318b50ae765b0918ff7aab4ce8f255"
)


@dataclass(frozen=True)
class Country:
    """One ADM0 country geometry and its stable identifiers."""

    iso_a2: str
    iso_a3: str
    name: str
    geometry: BaseGeometry


@dataclass(frozen=True)
class CountryCell:
    """An ERA5 0.25-degree cell with positive country overlap."""

    country_iso_a2: str
    country_iso_a3: str
    country_name: str
    cell_id: str
    latitude: float
    longitude: float
    center_in_country: bool

    @property
    def row_id_prefix(self) -> str:
        return f"{self.cell_id}-"


def cached_natural_earth_boundaries(cache_directory: Path) -> Path:
    """Return a checksum-verified, pinned Natural Earth GeoJSON file."""

    target = cache_directory / (
        f"natural-earth-{NATURAL_EARTH_VERSION}-admin-0-countries-10m.geojson"
    )
    if target.is_file() and _sha256(target) == NATURAL_EARTH_SHA256:
        return target

    cache_directory.mkdir(parents=True, exist_ok=True)
    with urlopen(NATURAL_EARTH_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != NATURAL_EARTH_SHA256:
        raise ValueError(
            "Natural Earth download checksum mismatch: "
            f"expected {NATURAL_EARTH_SHA256}, found {digest}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_directory,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return target


def load_countries(boundaries: Path, country_codes: Iterable[str]) -> list[Country]:
    """Load requested ISO alpha-2 or alpha-3 countries from GeoJSON."""

    requested = list(dict.fromkeys(_normalize_country_code(code) for code in country_codes))
    if not requested:
        raise ValueError("at least one country code is required")

    with boundaries.open(encoding="utf-8") as source:
        document = json.load(source)
    features = document.get("features")
    if not isinstance(features, list):
        raise ValueError(f"GeoJSON has no feature collection: {boundaries}")

    matched: dict[str, Country] = {}
    for feature in features:
        properties = feature.get("properties", {})
        aliases = _country_aliases(properties)
        for code in requested:
            if code not in aliases:
                continue
            country = _as_country(properties, feature.get("geometry"))
            previous = matched.get(code)
            if previous is not None and previous.iso_a3 != country.iso_a3:
                raise ValueError(f"Country code {code!r} is ambiguous in {boundaries}")
            matched[code] = country

    missing = [code for code in requested if code not in matched]
    if missing:
        raise ValueError(f"Country code(s) not found: {', '.join(missing)}")

    countries: list[Country] = []
    seen: set[str] = set()
    for code in requested:
        country = matched[code]
        if country.iso_a3 not in seen:
            countries.append(country)
            seen.add(country.iso_a3)
    return countries


def country_cells(
    country: Country, *, bounds: GeographicBounds | None = None
) -> list[CountryCell]:
    """Return grid cells overlapping the country, optionally clipped to bounds."""

    geometry = country_geometry(country, bounds=bounds)
    if geometry.is_empty:
        return []
    prepared = prep(geometry)
    min_longitude, min_latitude, max_longitude, max_latitude = geometry.bounds
    latitude_units = _candidate_units(min_latitude, max_latitude, -900_000, 900_000)
    longitude_units = _candidate_units(
        min_longitude,
        max_longitude,
        -1_800_000,
        1_797_500,
    )

    selected: list[CountryCell] = []
    for latitude_units_value in latitude_units:
        latitude = latitude_units_value / _COORDINATE_SCALE
        lower_latitude = max(-90.0, latitude - _HALF_SPACING_DEGREES)
        upper_latitude = min(90.0, latitude + _HALF_SPACING_DEGREES)
        for longitude_units_value in longitude_units:
            longitude = longitude_units_value / _COORDINATE_SCALE
            footprint = box(
                longitude - _HALF_SPACING_DEGREES,
                lower_latitude,
                longitude + _HALF_SPACING_DEGREES,
                upper_latitude,
            )
            if not prepared.intersects(footprint):
                continue
            if geometry.intersection(footprint).area <= 0.0:
                continue
            selected.append(
                CountryCell(
                    country_iso_a2=country.iso_a2,
                    country_iso_a3=country.iso_a3,
                    country_name=country.name,
                    cell_id=_cell_id(latitude_units_value, longitude_units_value),
                    latitude=latitude,
                    longitude=longitude,
                    center_in_country=prepared.covers(Point(longitude, latitude)),
                )
            )
    return selected


def country_geometry(
    country: Country, *, bounds: GeographicBounds | None = None
) -> BaseGeometry:
    """Return a country geometry, optionally clipped to validated bounds."""

    return (
        country.geometry
        if bounds is None
        else country.geometry.intersection(_bounds_geometry(bounds))
    )


def _bounds_geometry(bounds: GeographicBounds) -> BaseGeometry:
    west, south, east, north = bounds
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("bounds must be finite")
    if not (-180.0 <= west < east <= 180.0):
        raise ValueError("bounds longitudes must satisfy -180 <= west < east <= 180")
    if not (-90.0 <= south < north <= 90.0):
        raise ValueError("bounds latitudes must satisfy -90 <= south < north <= 90")
    return box(west, south, east, north)


def _candidate_units(
    minimum_degrees: float,
    maximum_degrees: float,
    lower_limit: int,
    upper_limit: int,
) -> range:
    half_spacing_units = ERA5_SPACING_UNITS // 2
    first = math.ceil(
        (minimum_degrees * _COORDINATE_SCALE - half_spacing_units)
        / ERA5_SPACING_UNITS
    ) * ERA5_SPACING_UNITS
    last = math.floor(
        (maximum_degrees * _COORDINATE_SCALE + half_spacing_units)
        / ERA5_SPACING_UNITS
    ) * ERA5_SPACING_UNITS
    first = max(first, lower_limit)
    last = min(last, upper_limit)
    return range(first, last + ERA5_SPACING_UNITS, ERA5_SPACING_UNITS)


def _cell_id(latitude_units: int, longitude_units: int) -> str:
    latitude_hemisphere = "n" if latitude_units >= 0 else "s"
    longitude_hemisphere = "e" if longitude_units >= 0 else "w"
    return (
        f"era5-global-{latitude_hemisphere}{abs(latitude_units):06d}-"
        f"{longitude_hemisphere}{abs(longitude_units):07d}"
    )


def _normalize_country_code(code: str) -> str:
    normalized = code.strip().upper()
    if len(normalized) not in (2, 3) or not normalized.isalpha():
        raise ValueError(f"Invalid ISO alpha-2/alpha-3 country code: {code!r}")
    return normalized


def _country_aliases(properties: Mapping[str, Any]) -> set[str]:
    return {
        str(properties.get(field, "")).upper()
        for field in ("ISO_A2_EH", "ISO_A2", "ISO_A3_EH", "ISO_A3", "ADM0_A3")
        if properties.get(field) not in (None, "", "-99")
    }


def _as_country(properties: Mapping[str, Any], geometry_document: Any) -> Country:
    iso_a2 = str(properties.get("ISO_A2_EH") or properties.get("ISO_A2") or "")
    iso_a3 = str(
        properties.get("ISO_A3_EH")
        or properties.get("ISO_A3")
        or properties.get("ADM0_A3")
        or ""
    )
    if not iso_a2 or not iso_a3 or geometry_document is None:
        raise ValueError("Country feature lacks ISO codes or geometry")
    return Country(
        iso_a2=iso_a2,
        iso_a3=iso_a3,
        name=str(properties.get("ADMIN") or properties.get("NAME") or iso_a3),
        geometry=shape(geometry_document),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
