"""Run trip/event detection against a library database and store the result."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
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


class DetectionCancelled(Exception):
    """Raised when a detection run is cancelled before anything was written."""


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _match_custom_names(
    groups: list[DetectedGroup], custom: list[dict[str, Any]]
) -> dict[int, str]:
    """Carry user names over to re-detected groups, one-to-one.

    Old and new groups are matched by the photos they share (Jaccard >= 0.5),
    best pairs first, and each old name is used at most once, so a renamed trip
    that splits in two keeps its name on the larger half only.

    Returns:
        ``{id(group): custom_name}``.
    """
    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(groups):
        ids = set(g.asset_ids)
        for ci, c in enumerate(custom):
            if c["kind"] != g.kind:
                continue
            score = _jaccard(ids, c["asset_ids"])
            if score >= 0.5:
                pairs.append((score, gi, ci))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_g: set[int] = set()
    used_c: set[int] = set()
    out: dict[int, str] = {}
    for _, gi, ci in pairs:
        if gi in used_g or ci in used_c:
            continue
        used_g.add(gi)
        used_c.add(ci)
        out[id(groups[gi])] = custom[ci]["custom_name"]
    return out


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
    conn: sqlite3.Connection,
    params: DetectionParams = DEFAULT_PARAMS,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Detect trips and events for the whole library and replace the stored ones.

    Names come from per-asset place names (``asset_places``) when present, else
    from reverse-geocoding the group's centre, else a date-based fallback.  A
    custom name given by the user is carried over to the re-detected group
    sharing most of its photos (see :func:`_match_custom_names`).

    Args:
        on_progress: Called with ``(message, step, n_steps)`` between stages.
        should_cancel: Polled between stages; when it returns ``True`` the run
            stops with :class:`DetectionCancelled` before writing anything.

    Returns:
        A summary dict (counts, timings, home location).
    """
    n_steps = 4

    def _step(message: str, step: int) -> None:
        if should_cancel is not None and should_cancel():
            raise DetectionCancelled()
        if on_progress is not None:
            on_progress(message, step, n_steps)

    t0 = time.monotonic()
    _step("Loading photos…", 0)
    raw = eq.load_detection_points(conn)
    points = [PhotoPoint(r["id"], r["taken_at"], r["lat"], r["lon"]) for r in raw]
    favorited = {r["id"] for r in raw if r["favorited"]}
    scores = eq.load_aesthetic_scores(conn)
    places = eq.load_asset_places(conn)
    custom = eq.load_custom_names(conn)

    _step(f"Detecting trips and events in {len(points)} photos…", 1)
    result = detect(points, params)
    now = int(time.time())
    all_groups = [g for top in result.trips + result.events for g in (top, *top.children)]
    custom_names = _match_custom_names(all_groups, custom)

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

    def _row(group: DetectedGroup) -> dict[str, Any]:
        label = _label(group)
        return {
            "kind": group.kind,
            "auto_name": auto_name(group, label),
            "custom_name": custom_names.get(id(group)),
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

    _step("Naming and choosing cover photos…", 2)
    rows = [_row(g) for g in result.trips] + [_row(g) for g in result.events]
    _step("Saving…", 3)
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
