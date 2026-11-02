"""Fill the ``asset_places`` table for a library."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from takeout_rater.db.queries import (
    delete_stale_asset_places,
    list_assets_for_geocoding,
    upsert_asset_places,
)
from takeout_rater.geo.places import GEOCODER_VERSION, asset_coords, reverse_geocode_many

_BATCH_SIZE = 5000


@dataclass
class GeocodeResult:
    """Outcome of :func:`geocode_library`."""

    candidates: int = 0  # assets with usable coordinates
    processed: int = 0  # assets looked up in this run
    named: int = 0  # of those, assets that got a place name
    removed: int = 0  # stale rows deleted (coordinates no longer usable)
    cancelled: bool = False


def geocode_library(
    conn: sqlite3.Connection,
    *,
    missing_only: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> GeocodeResult:
    """Reverse-geocode assets and store the results in ``asset_places``.

    Args:
        conn: Open library connection.
        missing_only: When ``True`` (default), only assets without a stored
            place, with changed coordinates, or geocoded by an older geocoder
            version are processed.  ``False`` recomputes every asset.
        on_progress: Called as ``on_progress(processed, total)`` after each batch.
        cancel_check: Returns ``True`` to stop after the current batch.
    """
    todo: list[tuple[int, float, float]] = []
    candidates = 0
    usable_ids: set[int] = set()
    for row in list_assets_for_geocoding(conn):
        coords = asset_coords(
            row["geo_lat"], row["geo_lon"], row["geo_exif_lat"], row["geo_exif_lon"]
        )
        if coords is None:
            continue
        candidates += 1
        usable_ids.add(row["id"])
        if (
            missing_only
            and row["p_lat"] is not None
            and (row["p_ver"] or 0) >= GEOCODER_VERSION
            and abs(row["p_lat"] - coords[0]) < 1e-9
            and abs(row["p_lon"] - coords[1]) < 1e-9
        ):
            continue
        todo.append((row["id"], coords[0], coords[1]))

    # Drop places of assets whose coordinates were cleared (or deleted assets).
    removed = delete_stale_asset_places(conn, usable_ids)

    result = GeocodeResult(candidates=candidates, removed=removed)
    total = len(todo)
    if on_progress is not None:
        on_progress(0, total)
    for start in range(0, total, _BATCH_SIZE):
        if cancel_check is not None and cancel_check():
            result.cancelled = True
            break
        batch = todo[start : start + _BATCH_SIZE]
        places = reverse_geocode_many([(lat, lon) for _, lat, lon in batch])
        rows = []
        for (asset_id, lat, lon), place in zip(batch, places, strict=True):
            row: dict = {
                "asset_id": asset_id,
                "lat": lat,
                "lon": lon,
                "geocoder_version": GEOCODER_VERSION,
            }
            if place is not None:
                result.named += 1
                row.update(
                    country_code=place.country_code,
                    country=place.country,
                    admin1=place.admin1,
                    admin2=place.admin2,
                    city=place.city,
                    distance_km=place.distance_km,
                )
            rows.append(row)
        upsert_asset_places(conn, rows)
        result.processed += len(batch)
        if on_progress is not None:
            on_progress(result.processed, total)
    return result
