"""Pure trip/event detection on ``(timestamp, lat, lon)`` points.

Two kinds of groups are detected:

* **Trips**: runs of local calendar days on which most geotagged photos were
  taken far away (``trip_min_distance_km``) from *home*.  Home is estimated
  week by week as the coarse grid cell photographed on the most distinct days
  within a centred rolling window, so a move to another city switches home at
  about the moving date instead of turning into a months-long "trip".
  Days without any geotagged photo do not interrupt a trip (up to
  ``trip_max_gap_days`` of them); a day with geotagged photos at home ends it.
  A trip spans its away days (widened to the travel days' photos without GPS,
  but never past a photo taken at home), and every photo in that time range
  belongs to it, with or without GPS.

* **Events**: bursts of photos separated from their neighbours by more than
  ``event_gap_hours`` (the classic "time gap" rule used by photo apps).  Events
  that fall inside a trip become children of that trip; the others are
  stand-alone events (birthdays, concerts, a day at the lake...).

Everything here is deterministic, dependency-free and O(n log n), so it runs in
a few seconds on a 200k-photo library.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

_EARTH_RADIUS_KM = 6371.0088
_DAY = 86400


@dataclass(frozen=True)
class PhotoPoint:
    """One photo: asset id, Unix timestamp (UTC) and optional coordinates."""

    asset_id: int
    ts: int
    lat: float | None = None
    lon: float | None = None

    @property
    def has_geo(self) -> bool:
        return self.lat is not None and self.lon is not None


@dataclass(frozen=True)
class DetectionParams:
    """Tunable thresholds for :func:`detect`."""

    home_cell_deg: float = 0.05  # ~5 km grid cells for the home estimate
    home_window_days: int = 60  # home = most-photographed cell within ± this many days
    min_home_days: int = 10  # geotagged days needed in that window, else all-time home
    trip_min_distance_km: float = 50.0
    trip_max_gap_days: int = 2  # days without geotagged photos tolerated inside a trip
    trip_min_photos: int = 10
    trip_max_days: int = 60  # longer "trips" are relocations, not trips
    event_gap_hours: float = 4.0
    event_min_photos: int = 15


DEFAULT_PARAMS = DetectionParams()


@dataclass
class DetectedGroup:
    """A detected trip or event."""

    kind: str  # "trip" | "event"
    start_ts: int
    end_ts: int
    asset_ids: list[int]
    center_lat: float | None = None
    center_lon: float | None = None
    max_distance_km: float | None = None  # furthest geotagged photo from home
    n_days: int = 1
    children: list[DetectedGroup] = field(default_factory=list)


@dataclass
class DetectionResult:
    trips: list[DetectedGroup]
    events: list[DetectedGroup]  # stand-alone events (not inside any trip)
    homes: dict[int, tuple[float, float]]  # local week number -> (lat, lon)
    default_home: tuple[float, float] | None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _utc_offset_s(lon: float | None) -> int:
    """Rough local-time offset from longitude (solar time, whole hours)."""
    if lon is None:
        return 0
    return int(round(lon / 15.0)) * 3600


def _local_day(p: PhotoPoint) -> int:
    """Local calendar day number of a photo (UTC shifted by its solar offset)."""
    return (p.ts + _utc_offset_s(p.lon)) // _DAY


def estimate_homes(
    points: list[PhotoPoint], params: DetectionParams = DEFAULT_PARAMS
) -> tuple[dict[int, tuple[float, float]], tuple[float, float] | None]:
    """Estimate where home was, week by week, with a centred rolling window.

    Home for a given week is the grid cell on which geotagged photos were taken
    on the most distinct days within ``± home_window_days`` (weighting by days,
    not photos, keeps one big photo session from outweighing everyday life).
    Because the window is centred, a move switches home at about the moving
    date, while a trip shorter than the half-window never becomes "home".
    Weeks whose window holds fewer than ``min_home_days`` geotagged days use
    the all-time home instead.

    Returns:
        ``(homes_by_week, default_home)``: keys are local week numbers
        (``local_day // 7``); ``default_home`` is ``None`` when no photo has
        coordinates.
    """
    cell = params.home_cell_deg
    day_cells: dict[int, set[tuple[int, int]]] = defaultdict(set)
    all_days: dict[tuple[int, int], set[int]] = defaultdict(set)
    coords_per_cell: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)

    for p in points:
        if not p.has_geo:
            continue
        key = (int(math.floor(p.lat / cell)), int(math.floor(p.lon / cell)))  # type: ignore[operator]
        day = _local_day(p)
        day_cells[day].add(key)
        all_days[key].add(day)
        coords_per_cell[key].append((p.lat, p.lon))  # type: ignore[arg-type]

    if not all_days:
        return {}, None

    centers: dict[tuple[int, int], tuple[float, float]] = {}

    def _cell_center(key: tuple[int, int]) -> tuple[float, float]:
        if key not in centers:
            pts = coords_per_cell[key]
            centers[key] = (
                statistics.median(lat for lat, _ in pts),
                statistics.median(lon for _, lon in pts),
            )
        return centers[key]

    default_home = _cell_center(max(all_days, key=lambda k: (len(all_days[k]), k)))

    # Slide a window of ± home_window_days over the sorted geotagged days and
    # keep, per cell, the number of distinct days it was photographed on.
    days = sorted(day_cells)
    half = params.home_window_days
    counts: Counter[tuple[int, int]] = Counter()
    lo = hi = 0
    homes: dict[int, tuple[float, float]] = {}
    for week in sorted({d // 7 for d in days}):
        mid = week * 7 + 3
        while hi < len(days) and days[hi] <= mid + half:
            counts.update(day_cells[days[hi]])
            hi += 1
        while lo < hi and days[lo] < mid - half:
            counts.subtract(day_cells[days[lo]])
            for c in day_cells[days[lo]]:
                if counts[c] <= 0:
                    del counts[c]
            lo += 1
        if hi - lo >= params.min_home_days and counts:
            best = max(counts, key=lambda k: (counts[k], k))
            homes[week] = _cell_center(best)
        else:
            homes[week] = default_home
    return homes, default_home


def _home_for(
    p: PhotoPoint, homes: dict[int, tuple[float, float]], default: tuple[float, float] | None
) -> tuple[float, float] | None:
    return homes.get(_local_day(p) // 7, default)


def _center(points: list[PhotoPoint]) -> tuple[float | None, float | None]:
    geo = [p for p in points if p.has_geo]
    if not geo:
        return None, None
    return (
        statistics.median(p.lat for p in geo),  # type: ignore[misc]
        statistics.median(p.lon for p in geo),  # type: ignore[misc]
    )


def detect_trips(
    points: list[PhotoPoint],
    homes: dict[int, tuple[float, float]],
    default_home: tuple[float, float] | None,
    params: DetectionParams = DEFAULT_PARAMS,
) -> list[DetectedGroup]:
    """Detect multi-day (or day-long) trips away from home.

    *points* must be sorted by timestamp.
    """
    if default_home is None:
        return []

    # Classify each local day: "away" when >= half its geotagged photos are far
    # from home, "home" otherwise.  Days with no geotagged photos stay unknown.
    by_day: dict[int, list[tuple[PhotoPoint, float]]] = defaultdict(list)
    for p in points:
        if not p.has_geo:
            continue
        home = _home_for(p, homes, default_home)
        assert home is not None
        dist = haversine_km(home[0], home[1], p.lat, p.lon)  # type: ignore[arg-type]
        by_day[_local_day(p)].append((p, dist))

    runs: list[list[int]] = []
    current: list[int] = []
    for day in sorted(by_day):
        entries = by_day[day]
        n_away = sum(1 for _, d in entries if d >= params.trip_min_distance_km)
        away = n_away * 2 >= len(entries)
        if away:
            if current and day - current[-1] - 1 <= params.trip_max_gap_days:
                current.append(day)
            else:
                if current:
                    runs.append(current)
                current = [day]
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    timestamps = [p.ts for p in points]
    home_ts = sorted(
        p.ts for entries in by_day.values() for p, d in entries if d < params.trip_min_distance_km
    )
    trips: list[DetectedGroup] = []
    for run in runs:
        n_days = run[-1] - run[0] + 1
        if n_days > params.trip_max_days:
            continue  # months "away" means home moved, not a trip
        away_pts = [
            (p, d) for day in run for p, d in by_day[day] if d >= params.trip_min_distance_km
        ]
        first = min(away_pts, key=lambda e: e[0].ts)[0]
        last = max(away_pts, key=lambda e: e[0].ts)[0]
        # Widen to the travel days' photos without GPS: back to the start of the
        # first local day (or just after the last photo at home), and likewise
        # forward to the end of the last day.
        day_start = run[0] * _DAY - _utc_offset_s(first.lon)
        day_end = (run[-1] + 1) * _DAY - _utc_offset_s(last.lon) - 1
        i = bisect.bisect_left(home_ts, first.ts)
        start_ts = max(day_start, home_ts[i - 1] + 1) if i > 0 else day_start
        j = bisect.bisect_right(home_ts, last.ts)
        end_ts = min(day_end, home_ts[j] - 1) if j < len(home_ts) else day_end
        start_ts, end_ts = min(start_ts, first.ts), max(end_ts, last.ts)
        lo = bisect.bisect_left(timestamps, start_ts)
        hi = bisect.bisect_right(timestamps, end_ts)
        members = points[lo:hi]
        # Report the span of actual photos, not of the widened window.
        start_ts, end_ts = members[0].ts, members[-1].ts
        if len(members) < params.trip_min_photos:
            continue
        clat, clon = _center([p for p, _ in away_pts])
        trips.append(
            DetectedGroup(
                kind="trip",
                start_ts=start_ts,
                end_ts=end_ts,
                asset_ids=[p.asset_id for p in members],
                center_lat=clat,
                center_lon=clon,
                max_distance_km=round(max(d for _, d in away_pts), 1),
                n_days=n_days,
            )
        )
    return trips


def _trip_index(ts: int, trips: list[DetectedGroup], starts: list[int]) -> int | None:
    """Index of the trip whose time range contains *ts* (trips sorted, disjoint)."""
    i = bisect.bisect_right(starts, ts) - 1
    if i >= 0 and ts <= trips[i].end_ts:
        return i
    return None


def detect_events(
    points: list[PhotoPoint],
    homes: dict[int, tuple[float, float]],
    default_home: tuple[float, float] | None,
    params: DetectionParams = DEFAULT_PARAMS,
    trips: list[DetectedGroup] | None = None,
) -> list[DetectedGroup]:
    """Split the timeline at gaps longer than ``event_gap_hours``.

    *points* must be sorted by timestamp.  When *trips* (sorted, disjoint) are
    given, the timeline is also split at trip boundaries, so an event never
    straddles a departure or return; events inside a trip are appended to that
    trip's ``children`` and only the stand-alone events are returned.
    Segments with fewer than ``event_min_photos`` photos are dropped.
    """
    if not points:
        return []
    trips = trips or []
    starts = [t.start_ts for t in trips]
    gap = params.event_gap_hours * 3600

    owners = [_trip_index(p.ts, trips, starts) for p in points]
    segments: list[tuple[int | None, list[PhotoPoint]]] = [(owners[0], [points[0]])]
    for i in range(1, len(points)):
        prev, cur = points[i - 1], points[i]
        if cur.ts - prev.ts > gap or owners[i] != owners[i - 1]:
            segments.append((owners[i], [cur]))
        else:
            segments[-1][1].append(cur)

    standalone: list[DetectedGroup] = []
    for owner, seg in segments:
        if len(seg) < params.event_min_photos:
            continue
        clat, clon = _center(seg)
        max_dist = None
        if default_home is not None:
            dists = []
            for p in seg:
                if p.has_geo:
                    home = _home_for(p, homes, default_home)
                    assert home is not None
                    dists.append(haversine_km(home[0], home[1], p.lat, p.lon))  # type: ignore[arg-type]
            if dists:
                max_dist = round(max(dists), 1)
        off = _utc_offset_s(clon)
        event = DetectedGroup(
            kind="event",
            start_ts=seg[0].ts,
            end_ts=seg[-1].ts,
            asset_ids=[p.asset_id for p in seg],
            center_lat=clat,
            center_lon=clon,
            max_distance_km=max_dist,
            n_days=(seg[-1].ts + off) // _DAY - (seg[0].ts + off) // _DAY + 1,
        )
        if owner is None:
            standalone.append(event)
        else:
            trips[owner].children.append(event)
    return standalone


def detect(points: list[PhotoPoint], params: DetectionParams = DEFAULT_PARAMS) -> DetectionResult:
    """Detect trips and events; events inside a trip are nested under it."""
    pts = sorted(points, key=lambda p: (p.ts, p.asset_id))
    homes, default_home = estimate_homes(pts, params)
    trips = detect_trips(pts, homes, default_home, params)
    events = detect_events(pts, homes, default_home, params, trips)
    return DetectionResult(trips=trips, events=events, homes=homes, default_home=default_home)


# ── Naming ────────────────────────────────────────────────────────────────────

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def local_year(ts: int, lon: float | None = None) -> int:
    """Calendar year of *ts* in (approximate) local time at longitude *lon*."""
    return datetime.fromtimestamp(ts + _utc_offset_s(lon), tz=UTC).year


def format_date_range(start_ts: int, end_ts: int, lon: float | None = None) -> str:
    """Compact human date range in (approximate) local time.

    Examples: ``"12 Mar 2024"``, ``"3–9 Jun 2023"``, ``"28 Jun – 2 Jul 2023"``,
    ``"30 Dec 2023 – 2 Jan 2024"``.
    """
    off = _utc_offset_s(lon)
    a = datetime.fromtimestamp(start_ts + off, tz=UTC)
    b = datetime.fromtimestamp(end_ts + off, tz=UTC)
    if a.date() == b.date():
        return f"{a.day} {_MONTHS[a.month - 1]} {a.year}"
    if a.year != b.year:
        return f"{a.day} {_MONTHS[a.month - 1]} {a.year} – {b.day} {_MONTHS[b.month - 1]} {b.year}"
    if a.month != b.month:
        return f"{a.day} {_MONTHS[a.month - 1]} – {b.day} {_MONTHS[b.month - 1]} {b.year}"
    return f"{a.day}–{b.day} {_MONTHS[a.month - 1]} {a.year}"


def place_label(places: list[tuple[str | None, str | None, str | None]]) -> str | None:
    """Summarise a list of ``(city, admin1, country)`` tuples into a short label.

    * One dominant city: ``"Lisbon, Portugal"``
    * Two notable cities in one country: ``"Lisbon & Porto, Portugal"``
    * Several countries: ``"Portugal & Spain"`` (up to three)

    A place is "notable" when it accounts for at least 20 % of the photos.
    """
    places = [p for p in places if any(p)]
    if not places:
        return None
    n = len(places)
    countries = Counter(p[2] for p in places if p[2])
    notable_countries = [c for c, k in countries.most_common() if k / n >= 0.2]
    if len(notable_countries) > 1:
        return " & ".join(notable_countries[:3])
    country = notable_countries[0] if notable_countries else None

    cities = Counter(p[0] for p in places if p[0])
    notable_cities = [c for c, k in cities.most_common() if k / n >= 0.2][:2]
    if not notable_cities:
        regions = Counter(p[1] for p in places if p[1])
        notable_cities = [r for r, _ in regions.most_common(1)]
    parts = " & ".join(notable_cities)
    if parts and country:
        return f"{parts}, {country}"
    return parts or country


def auto_name(group: DetectedGroup, place: str | None) -> str:
    """Default display name for a group.

    Trips are named after where they went (``"Lisbon & Porto, Portugal"``);
    events carry their date too, since many happen in the same home town
    (``"Berlin · 24 Dec 2024"``).  Without place names, the date alone is used
    (``"Trip · Jun 2023"`` / ``"Event · 24 Dec 2024"``).
    """
    off = _utc_offset_s(group.center_lon)
    a = datetime.fromtimestamp(group.start_ts + off, tz=UTC)
    if group.kind == "trip":
        return place or f"Trip · {_MONTHS[a.month - 1]} {a.year}"
    day = f"{a.day} {_MONTHS[a.month - 1]} {a.year}"
    if place:
        return f"{place.split(',')[0]} · {day}"
    return f"Event · {day}"
