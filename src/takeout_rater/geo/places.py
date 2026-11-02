"""Offline reverse geocoding: GPS coordinates → country / region / city.

Uses the GeoNames ``cities1000`` gazetteer (every populated place with at
least 1,000 inhabitants, ~155k entries) that ships inside the
``reverse_geocode`` package.  No network access is needed at runtime.

Lookup works in two steps:

1. Find the nearest gazetteer entry with a k-d tree over 3-D unit vectors
   (so distances are correct at every latitude, unlike a lat/lon tree).
2. "Snap" that entry to a larger settlement nearby.  GeoNames lists many
   city districts as their own places (Berlin-Mitte, Lisbon-Intendente…),
   which would split one city into dozens of browse filter values.  A place
   is replaced by the most populous place in the same country and region
   whose *catchment radius* covers it.  The radius grows with the square
   root of the population (≈ 4 km for 100k inhabitants, ≈ 19 km for 3.4M),
   so neighbouring independent towns stay separate while districts collapse
   into their city.

Public API (kept deliberately small, other features build on it):

- :class:`Place` – the result of a lookup, with a human-readable ``label``.
- :func:`reverse_geocode` – one coordinate pair → ``Place | None``.
- :func:`reverse_geocode_many` – many coordinate pairs, vectorised.
- :func:`asset_coords` – choose the usable coordinates of an asset row.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources

import numpy as np

#: Bump when the lookup algorithm or data changes, so stored rows can be
#: recognised as stale and recomputed.
GEOCODER_VERSION: int = 1

_EARTH_RADIUS_KM = 6371.0088

#: Beyond this distance from the nearest populated place we do not claim a
#: city (e.g. photos taken at sea, in deserts or high mountains).  Country and
#: region are still reported from the nearest place.
CITY_MAX_DISTANCE_KM = 50.0

#: Beyond this distance we report no place at all (open sea, polar regions).
PLACE_MAX_DISTANCE_KM = 200.0

#: Upper bound on any snap radius, which also bounds the neighbour search.
_MAX_SNAP_KM = 30.0


@dataclass(frozen=True)
class Place:
    """A reverse-geocoded place name.

    Attributes:
        country_code: ISO 3166-1 alpha-2 code, e.g. ``"PT"``.
        country: English country name, e.g. ``"Portugal"``.
        admin1: First-level subdivision (state / region / province), or ``None``.
        admin2: Second-level subdivision (county / district), or ``None``.
        city: Populated place name, or ``None`` when the coordinates are far
            from any known place.
        distance_km: Distance from the coordinates to the nearest gazetteer
            entry (before snapping), or ``None`` when unknown.
    """

    country_code: str
    country: str
    admin1: str | None = None
    admin2: str | None = None
    city: str | None = None
    distance_km: float | None = None

    @property
    def label(self) -> str:
        """Short display label: ``"City, Country"`` (or region / country only)."""
        first = self.city or self.admin1
        if first and self.country:
            return f"{first}, {self.country}"
        return first or self.country or self.country_code


# ---------------------------------------------------------------------------
# Coordinate selection
# ---------------------------------------------------------------------------


def _valid(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    if lat == 0.0 and lon == 0.0:  # Takeout writes 0/0 for "unknown"
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def asset_coords(
    geo_lat: float | None,
    geo_lon: float | None,
    geo_exif_lat: float | None = None,
    geo_exif_lon: float | None = None,
) -> tuple[float, float] | None:
    """Return the usable ``(lat, lon)`` of an asset, or ``None``.

    Prefers the sidecar ``geoData`` coordinates (which include manual edits
    made in Google Photos) and falls back to ``geoDataExif``.  ``0.0, 0.0``
    means "unknown" in Takeout sidecars and is treated as missing.
    """
    if _valid(geo_lat, geo_lon):
        return float(geo_lat), float(geo_lon)  # type: ignore[arg-type]
    if _valid(geo_exif_lat, geo_exif_lon):
        return float(geo_exif_lat), float(geo_exif_lon)  # type: ignore[arg-type]
    return None


# ---------------------------------------------------------------------------
# Gazetteer
# ---------------------------------------------------------------------------


def _to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def _chord_to_km(chord: np.ndarray | float) -> np.ndarray | float:
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.clip(np.asarray(chord) / 2.0, 0.0, 1.0))


def _km_to_chord(km: float) -> float:
    return 2.0 * math.sin(km / (2.0 * _EARTH_RADIUS_KM))


def snap_radius_km(population: int) -> float:
    """Catchment radius of a settlement, used to merge districts into cities."""
    return min(_MAX_SNAP_KM, 1.0 + 3.0 * math.sqrt(max(population, 0) / 100_000.0))


class _Gazetteer:
    def __init__(self, locations: list[dict], countries: dict[str, str]) -> None:
        from scipy.spatial import cKDTree

        n = len(locations)
        lat = np.fromiter((loc["latitude"] for loc in locations), dtype=np.float64, count=n)
        lon = np.fromiter((loc["longitude"] for loc in locations), dtype=np.float64, count=n)
        self.population = np.fromiter(
            (int(loc.get("population") or 0) for loc in locations), dtype=np.int64, count=n
        )
        self.city: list[str] = [loc.get("city") or "" for loc in locations]
        self.cc: list[str] = [loc.get("country_code") or "" for loc in locations]
        self.admin1: list[str | None] = [loc.get("state") or None for loc in locations]
        self.admin2: list[str | None] = [loc.get("county") or None for loc in locations]
        self.countries = countries
        self.xyz = _to_xyz(lat, lon)
        self.tree = cKDTree(self.xyz)
        self._snap_cache: dict[int, int] = {}
        self._snap_chord = _km_to_chord(_MAX_SNAP_KM)

    def snap(self, idx: int) -> int:
        cached = self._snap_cache.get(idx)
        if cached is not None:
            return cached
        cands = np.asarray(self.tree.query_ball_point(self.xyz[idx], self._snap_chord), dtype=int)
        best = idx
        if cands.size > 1:
            pops = self.population[cands]
            mask = pops > self.population[idx]
            cands = cands[mask]
            if cands.size:
                dists = _chord_to_km(np.linalg.norm(self.xyz[cands] - self.xyz[idx], axis=1))
                cc, a1 = self.cc[idx], self.admin1[idx]
                best_pop = -1
                for j, d in zip(cands.tolist(), np.atleast_1d(dists).tolist(), strict=True):
                    p = int(self.population[j])
                    if p <= best_pop or self.cc[j] != cc or self.admin1[j] != a1:
                        continue
                    if d <= snap_radius_km(p):
                        best, best_pop = j, p
        self._snap_cache[idx] = best
        return best

    def lookup(self, coords: Sequence[tuple[float, float]]) -> list[Place | None]:
        if not coords:
            return []
        arr = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
        chords, indices = self.tree.query(_to_xyz(arr[:, 0], arr[:, 1]), k=1)
        dists_km = np.atleast_1d(_chord_to_km(chords))
        out: list[Place | None] = []
        for idx, dist in zip(np.atleast_1d(indices).tolist(), dists_km.tolist(), strict=True):
            if dist > PLACE_MAX_DISTANCE_KM:
                out.append(None)
                continue
            cc = self.cc[idx]
            country = self.countries.get(cc, cc)
            if dist > CITY_MAX_DISTANCE_KM:
                out.append(Place(cc, country, self.admin1[idx], None, None, round(dist, 2)))
                continue
            s = self.snap(idx)
            out.append(
                Place(
                    country_code=cc,
                    country=country,
                    admin1=self.admin1[s],
                    admin2=self.admin2[s],
                    city=self.city[s] or None,
                    distance_km=round(dist, 2),
                )
            )
        return out


_GAZETTEER: _Gazetteer | None = None
_GAZETTEER_LOCK = threading.Lock()


def _load_gazetteer() -> _Gazetteer:
    """Load the bundled GeoNames data (once per process, ~1–2 s)."""
    global _GAZETTEER
    if _GAZETTEER is not None:
        return _GAZETTEER
    with _GAZETTEER_LOCK:
        if _GAZETTEER is not None:
            return _GAZETTEER
        try:
            pkg = resources.files("reverse_geocode")
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "Place-name lookup needs the 'reverse_geocode' package "
                "(pip install reverse_geocode)."
            ) from exc
        data_file = pkg / "geocode.gz"
        if not data_file.is_file():  # pragma: no cover - broken install
            raise RuntimeError(
                "The reverse_geocode package is installed without its bundled "
                "geocode.gz data file; reinstall it."
            )
        with data_file.open("rb") as fh, gzip.open(fh) as gz:
            locations = json.loads(gz.read())
        countries: dict[str, str] = {}
        countries_file = pkg / "countries.csv"
        if countries_file.is_file():
            with countries_file.open("r", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    if len(row) >= 2:
                        countries[row[0]] = row[1]
        _GAZETTEER = _Gazetteer(locations, countries)
        return _GAZETTEER


def reverse_geocode_many(coords: Sequence[tuple[float, float]]) -> list[Place | None]:
    """Reverse-geocode many ``(lat, lon)`` pairs at once.

    Invalid pairs (``None``, ``0/0``, out of range) and points more than
    :data:`PLACE_MAX_DISTANCE_KM` from any populated place yield ``None``.
    """
    valid_idx = [i for i, c in enumerate(coords) if c is not None and _valid(c[0], c[1])]
    result: list[Place | None] = [None] * len(coords)
    if not valid_idx:
        return result
    places = _load_gazetteer().lookup([coords[i] for i in valid_idx])
    for i, place in zip(valid_idx, places, strict=True):
        result[i] = place
    return result


def reverse_geocode(lat: float | None, lon: float | None) -> Place | None:
    """Reverse-geocode one coordinate pair; ``None`` when coordinates are unusable."""
    if not _valid(lat, lon):
        return None
    return reverse_geocode_many([(float(lat), float(lon))])[0]  # type: ignore[arg-type]
