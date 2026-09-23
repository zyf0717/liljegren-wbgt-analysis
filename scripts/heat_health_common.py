"""Shared contracts for the 2025 Korean and Taiwanese heat-health workflows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from shapely.geometry import shape

from liljegren_analysis.country.cells import Country, country_cells


@dataclass(frozen=True)
class CountrySpec:
    iso3: str
    source: str
    timezone: str
    first_day: date
    last_day: date
    names: dict[str, str]

    @property
    def root(self) -> Path:
        return Path("data") / self.source / "2025"


KOREA_NAMES = dict(zip(
    ("서울", "부산", "대구", "인천", "광주", "대전", "울산", "경기", "강원", "충북",
     "충남", "전북", "전남", "경북", "경남", "제주", "세종"),
    ("KR-11", "KR-26", "KR-27", "KR-28", "KR-29", "KR-30", "KR-31", "KR-41",
     "KR-42", "KR-43", "KR-44", "KR-45", "KR-46", "KR-47", "KR-48", "KR-49", "KR-50"),
    strict=True,
))
TAIWAN_NAMES = dict(zip(
    ("南投縣", "台中市", "台北市", "台南市", "台東縣", "嘉義市", "嘉義縣", "基隆市",
     "宜蘭縣", "屏東縣", "彰化縣", "新北市", "新竹市", "新竹縣", "桃園市", "澎湖縣",
     "花蓮縣", "苗栗縣", "連江縣", "金門縣", "雲林縣", "高雄市"),
    ("TW-NAN", "TW-TXG", "TW-TPE", "TW-TNN", "TW-TTT", "TW-CYI", "TW-CYQ",
     "TW-KEE", "TW-ILA", "TW-PIF", "TW-CHA", "TW-NWT", "TW-HSZ", "TW-HSQ",
     "TW-TAO", "TW-PEN", "TW-HUA", "TW-MIA", "TW-LIE", "TW-KIN", "TW-YUN", "TW-KHH"),
    strict=True,
))
SPECS = {
    "KOR": CountrySpec("KOR", "kdca", "Asia/Seoul", date(2025, 5, 15), date(2025, 9, 25), KOREA_NAMES),
    "TWN": CountrySpec("TWN", "hpa", "Asia/Taipei", date(2025, 1, 1), date(2025, 12, 31), TAIWAN_NAMES),
}
BOUNDARY_REVISION = "9469f09"
BOUNDARY_METADATA = {
    "KOR": {"represented_year": 2021, "source": "Natural Earth", "license": "Public Domain"},
    "TWN": {"represented_year": 2017, "source": "OpenStreetMap/Wambacher", "license": "ODbL 1.0"},
}


def boundary_url(iso3: str) -> str:
    return (f"https://github.com/wmgeolab/geoBoundaries/raw/{BOUNDARY_REVISION}/"
            f"releaseData/gbOpen/{iso3}/ADM1/geoBoundaries-{iso3}-ADM1.geojson")


def download(url: str, path: Path, *, refresh: bool = False, ca_bundle: Path | None = None) -> str:
    """Atomically cache a bounded HTTPS response with certificate verification."""
    if urlparse(url).scheme != "https":
        raise ValueError(f"Source URL must use HTTPS: {url}")
    if not path.exists() or refresh:
        import ssl

        path.parent.mkdir(parents=True, exist_ok=True)
        context = ssl.create_default_context(cafile=str(ca_bundle) if ca_bundle else None)
        if urlparse(url).hostname == "www.hpa.gov.tw" and ca_bundle is None:
            # HPA sends only its leaf certificate. Pin the missing
            # intermediate before adding it to the TLS trust store.
            issuer_url = "http://sslserver.twca.com.tw/cacert/Secure_sha2_2023G3.crt"
            with urlopen(issuer_url, timeout=20) as response:
                intermediate = response.read(10_000)
            if hashlib.sha256(intermediate).hexdigest() != "1a2c75fd096e0499e9ff6ac74e526f61eaae3edfc8c2ea4436fee0c24d8b7d0e":
                raise ValueError("HPA intermediate certificate changed; supply --ca-bundle")
            context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(intermediate))
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(descriptor, "wb") as target, urlopen(
                Request(url, headers={"User-Agent": "Mozilla/5.0 (liljegren-wbgt-analysis)", "Referer": "https://data.gov.tw/"}),
                timeout=60, context=context,
            ) as response:
                if urlparse(response.url).scheme != "https":
                    raise ValueError(f"Source redirected away from HTTPS: {url}")
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 50_000_000:
                        raise ValueError(f"Download exceeds 50 MB: {url}")
                    target.write(chunk)
            os.replace(temporary_name, path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_input(source: Path, target: Path) -> str:
    """Retain a local source file with the same atomic publication as downloads."""
    if source.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        try:
            with source.open("rb") as original, os.fdopen(descriptor, "wb") as copy:
                while chunk := original.read(1024 * 1024):
                    copy.write(chunk)
            os.replace(temporary_name, target)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def boundaries(spec: CountrySpec, path: Path, *, refresh: bool = False):
    digest = download(boundary_url(spec.iso3), path, refresh=refresh)
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    expected = set(spec.names.values())
    found = {feature["properties"]["shapeISO"] for feature in features}
    if found != expected:
        raise ValueError(f"{spec.iso3} ADM1 codes changed: missing={sorted(expected - found)}, extra={sorted(found - expected)}")
    result = {}
    for feature in features:
        code = feature["properties"]["shapeISO"]
        geometry = shape(feature["geometry"])
        if not geometry.is_valid or geometry.is_empty:
            raise ValueError(f"Invalid {spec.iso3} ADM1 geometry: {code}")
        result[code] = geometry
    return result, digest


def region_cells(spec: CountrySpec, geometries):
    return {
        code: country_cells(Country("", spec.iso3, code, geometry))
        for code, geometry in geometries.items()
    }
