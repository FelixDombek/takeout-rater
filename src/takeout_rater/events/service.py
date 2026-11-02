"""Run trip/event detection against a library database and store the result."""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from takeout_rater.db import event_queries as eq
from takeout_rater.events.detector import (
    DEFAULT_PARAMS,
    DetectedGroup,
    DetectionParams,
    PhotoPoint,
    auto_name,
    detect,
    place_label,
)

_log = logging.getLogger(__name__)


def _overlap_ratio(a0: int, a1: int, b0: int, b1: int) -> float:
    """Intersection over union of two closed time ranges."""
    inter = min(a1, b1) - max(a0, b0)
    if inter < 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return 1.0 if union <= 0 else inter / union


def _reverse_geocode_center(lat: float, lon: float) -> str | None:
    """Label a coordinate via the optional offline geocoder, if installed."""
    try:
        from takeout_rater.geo.places import reverse_geocode  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - feature not available in this build
        return None
    try:
        place = reverse_geocode(lat, lon)
    except Exception:  # noqa: BLE001
        _log.debug("reverse_geocode failed for %s,%s", lat, lon, exc_info=True)
        return None
    return getattr(place, "label", None) if place is not None else None


def run_detection(
    conn: sqlite3.Connection, params: DetectionParams = DEFAULT_PARAMS
) -> dict[str, Any]:
    """Detect trips and events for the whole library and replace the stored ones.

    Names come from per-asset place names (``asset_places``) when present, else
    from reverse-geocoding the group's centre, else a date-based fallback.  A
    custom name given by the user is carried over to the re-detected group
    that overlaps it most (at least 50 % IoU of the time range).

    Returns:
        A summary dict (counts, timings, home location).
    """
    t0 = time.monotonic()
    raw = eq.load_detection_points(conn)
    points = [PhotoPoint(r["id"], r["taken_at"], r["lat"], r["lon"]) for r in raw]
    favorited = {r["id"] for r in raw if r["favorited"]}
    scores = eq.load_aesthetic_scores(conn)
    places = eq.load_asset_places(conn)
    custom = eq.load_custom_names(conn)

    result = detect(points, params)
    now = int(time.time())

    def _cover(group: DetectedGroup) -> int | None:
        ids = group.asset_ids
        if not ids:
            return None
        scored = sorted(((scores[a], a) for a in ids if a in scores), reverse=True)
        if scored:
            # Best-scored photo, but prefer a favourite from the top quarter.
            top = scored[: max(1, len(scored) // 4)]
            fav_top = [a for _, a in top if a in favorited]
            return fav_top[0] if fav_top else scored[0][1]
        favs = [a for a in ids if a in favorited]
        if favs:
            return favs[len(favs) // 2]
        return ids[len(ids) // 2]

    def _label(group: DetectedGroup) -> str | None:
        if places:
            label = place_label([places[a] for a in group.asset_ids if a in places])
            if label:
                return label
        if group.center_lat is not None and group.center_lon is not None:
            return _reverse_geocode_center(group.center_lat, group.center_lon)
        return None

    def _custom_for(group: DetectedGroup) -> str | None:
        best, best_ratio = None, 0.5
        for c in custom:
            if c["kind"] != group.kind:
                continue
            r = _overlap_ratio(group.start_ts, group.end_ts, c["start_ts"], c["end_ts"])
            if r >= best_ratio:
                best, best_ratio = c["custom_name"], r
        return best

    def _row(group: DetectedGroup) -> dict[str, Any]:
        label = _label(group)
        return {
            "kind": group.kind,
            "auto_name": auto_name(group, label),
            "custom_name": _custom_for(group),
            "place_label": label,
            "start_ts": group.start_ts,
            "end_ts": group.end_ts,
            "n_days": group.n_days,
            "center_lat": group.center_lat,
            "center_lon": group.center_lon,
            "max_distance_km": group.max_distance_km,
            "cover_asset_id": _cover(group),
            "detected_at": now,
            "asset_ids": group.asset_ids,
            "children": [_row(c) for c in group.children],
        }

    rows = [_row(g) for g in result.trips] + [_row(g) for g in result.events]
    eq.replace_events(conn, rows)

    elapsed = time.monotonic() - t0
    summary = {
        "n_photos": len(points),
        "n_geotagged": sum(1 for p in points if p.has_geo),
        "n_trips": len(result.trips),
        "n_events": len(result.events),
        "n_nested_events": sum(len(t.children) for t in result.trips),
        "home": result.default_home,
        "used_place_names": places is not None and bool(places),
        "elapsed_s": round(elapsed, 2),
    }
    _log.info("Trip/event detection: %s", summary)
    return summary
