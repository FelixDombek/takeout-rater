"""Folder layouts for exported photos (GitHub issue #118).

An export copies one photo per cluster.  By default everything lands flat in
the export directory; the helpers here instead assign every exported photo a
relative sub-folder:

- ``category``: zero-shot CLIP classification of the stored ViT-L/14 image
  embedding against a small label set (e.g. ``Nature/Beach``, ``Food & Drink``).
- ``trip``: the detected trip or event the photo belongs to
  (``2024-05-12 Lisbon``), otherwise ``Not in a trip/<year>``.
- ``place``: the reverse-geocoded ``Country/City``, otherwise ``No location``.

Nothing is written to the database: every layout is derived from tables other
features already fill (``clip_embeddings``, ``events``, ``asset_places``).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

FOLDER_MODES: tuple[str, ...] = ("flat", "category", "trip", "place")

#: Folder for photos the classifier is not confident about.
OTHER_FOLDER = "Other"
#: Folder for photos that have no CLIP embedding yet.
UNCLASSIFIED_FOLDER = "Unclassified"
NO_TRIP_FOLDER = "Not in a trip"
NO_PLACE_FOLDER = "No location"

#: CLIP's logit scale for ViT-L/14 (OpenAI), used to turn cosine similarities
#: into a softmax over the categories.
_LOGIT_SCALE = 100.0

#: Default minimum softmax probability of the best category; below it the photo
#: goes to :data:`OTHER_FOLDER`.  With ~25 categories a uniform guess is 0.04;
#: 0.25 keeps clear matches (a coffee cup scored ~0.3, a cat ~0.95 in a quick check)
#: and sends muddled ones to Other.  Users can change it in the export card.
DEFAULT_MIN_CONFIDENCE = 0.25


@dataclass(frozen=True)
class Category:
    """One export folder and the CLIP prompts that describe it."""

    folder: str
    prompts: tuple[str, ...]


DEFAULT_CATEGORIES: tuple[Category, ...] = (
    Category(
        "People/Portraits",
        ("a portrait photo of a person", "a close-up photo of a person's face"),
    ),
    Category("People/Groups", ("a group photo of friends", "a photo of a group of people")),
    Category("People/Kids", ("a photo of a child playing", "a photo of a baby")),
    Category("People/Selfies", ("a selfie",)),
    Category("Pets/Dogs", ("a photo of a dog",)),
    Category("Pets/Cats", ("a photo of a cat",)),
    Category(
        "Animals",
        ("a photo of a wild animal", "a photo of a bird", "a photo of farm animals"),
    ),
    Category(
        "Food & Drink",
        ("a photo of food on a plate", "a photo of a meal", "a photo of drinks"),
    ),
    Category("Nature/Beach", ("a photo of a beach", "a photo of the sea and the coast")),
    Category("Nature/Mountains", ("a photo of mountains", "a hiking trail in the mountains")),
    Category(
        "Nature/Forest & Countryside",
        ("a photo of a forest", "a photo of fields and countryside"),
    ),
    Category("Nature/Lakes & Rivers", ("a photo of a lake", "a photo of a river")),
    Category("Nature/Snow", ("a snowy winter landscape", "people skiing in the snow")),
    Category("Nature/Sunsets", ("a photo of a sunset", "a photo of a sunrise")),
    Category("Nature/Flowers & Plants", ("a close-up photo of flowers", "a photo of a garden")),
    Category("City/Streets", ("a photo of a city street", "a city skyline")),
    Category(
        "City/Landmarks & Architecture",
        ("a photo of a famous landmark", "a photo of a historic building", "a church interior"),
    ),
    Category("City/Night", ("a photo of a city at night", "city lights at night")),
    Category("Home", ("a photo inside a home", "a photo of a living room")),
    Category(
        "Celebrations",
        ("a birthday party", "a wedding", "people celebrating at a party"),
    ),
    Category("Concerts & Shows", ("a concert", "a stage performance")),
    Category("Sports", ("people playing sports", "a person swimming", "a person riding a bike")),
    Category("Vehicles", ("a photo of a car", "a photo of a train", "a photo of an airplane")),
    Category("Art & Museums", ("a painting in a museum", "a sculpture")),
    Category(
        "Documents & Screenshots",
        ("a screenshot", "a photo of a document", "a photo of a receipt", "a photo of text"),
    ),
    # Catch-all: gives textures, graphics and other odd shots somewhere to go
    # instead of being forced into the nearest real category.
    Category(
        OTHER_FOLDER,
        ("an abstract pattern", "a logo or graphic"),
    ),
)


# ---------------------------------------------------------------------------
# Folder name handling
# ---------------------------------------------------------------------------

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_MAX_SEGMENT = 80


def sanitize_segment(name: str) -> str:
    """Make *name* safe as a single folder name on Windows, macOS and Linux."""
    cleaned = _INVALID_CHARS.sub("_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(". ")
    cleaned = cleaned[:_MAX_SEGMENT].rstrip(". ")
    if not cleaned or cleaned in {".", ".."}:
        return "_"
    if cleaned.split(".")[0].upper() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned


def sanitize_folder(path: str) -> str:
    """Sanitize every ``/``-separated segment of a relative folder path."""
    segments = [s for s in path.replace("\\", "/").split("/") if s.strip()]
    return "/".join(sanitize_segment(s) for s in segments) or "_"


# ---------------------------------------------------------------------------
# Category labels
# ---------------------------------------------------------------------------


def parse_categories(text: str) -> list[Category]:
    """Parse a user-edited label list, one category per line.

    Each line is ``Folder/Name: prompt one; prompt two``.  Without a colon the
    prompt defaults to ``a photo of <last folder segment>``.  Blank lines and
    lines starting with ``#`` are ignored.

    Raises:
        ValueError: When a line has an empty folder, or fewer than two
            categories are given (zero-shot needs something to compare with).
    """
    categories: list[Category] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        folder, sep, rest = line.partition(":")
        folder = folder.strip().strip("/")
        if not folder:
            raise ValueError(f"Line {lineno}: missing folder name before ':'")
        prompts = tuple(p.strip() for p in rest.split(";") if p.strip()) if sep else ()
        if not prompts:
            prompts = (f"a photo of {folder.split('/')[-1].lower()}",)
        categories.append(Category(sanitize_folder(folder), prompts))
    if len(categories) < 2:
        raise ValueError("Give at least two categories.")
    return categories


def format_categories(categories: Iterable[Category]) -> str:
    """Inverse of :func:`parse_categories` (used to prefill the UI)."""
    return "\n".join(f"{c.folder}: {'; '.join(c.prompts)}" for c in categories)


def encode_texts_clip(texts: Sequence[str]) -> np.ndarray:
    """Encode *texts* with CLIP ViT-L/14; return L2-normalised float32 rows."""
    import torch

    from takeout_rater.scoring.scorers.clip_backbone import get_clip_model

    model, _preprocess, tokenizer, device = get_clip_model()
    tokens = tokenizer(list(texts)).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()


def category_text_matrix(
    categories: Sequence[Category],
    encode_texts: Callable[[Sequence[str]], np.ndarray] = encode_texts_clip,
) -> np.ndarray:
    """One L2-normalised text vector per category (the mean of its prompts)."""
    prompts = [p for c in categories for p in c.prompts]
    vecs = np.asarray(encode_texts(prompts), dtype=np.float32)
    rows = []
    start = 0
    for c in categories:
        mean = vecs[start : start + len(c.prompts)].mean(axis=0)
        start += len(c.prompts)
        rows.append(mean / (np.linalg.norm(mean) or 1.0))
    return np.stack(rows)


def classify_embeddings(
    image_vecs: np.ndarray,
    text_matrix: np.ndarray,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-shot classify L2-normalised *image_vecs* (N, D) against categories.

    Returns ``(best_index, confidence)``; ``best_index`` is ``-1`` where the
    softmax probability of the best category is below *min_confidence*.
    """
    if len(image_vecs) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    logits = _LOGIT_SCALE * (image_vecs @ text_matrix.T)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    best = probs.argmax(axis=1)
    conf = probs[np.arange(len(best)), best]
    best = np.where(conf >= min_confidence, best, -1)
    return best.astype(np.int64), conf.astype(np.float32)


def category_folders(
    conn: sqlite3.Connection,
    asset_ids: Sequence[int],
    categories: Sequence[Category] = DEFAULT_CATEGORIES,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    encode_texts: Callable[[Sequence[str]], np.ndarray] = encode_texts_clip,
) -> dict[int, str]:
    """Map each asset id to a category folder using its stored CLIP embedding."""
    vectors: dict[int, np.ndarray] = {}
    for chunk in _chunks(list(asset_ids), 500):
        placeholders = ",".join("?" * len(chunk))
        for aid, blob in conn.execute(
            f"SELECT asset_id, embedding FROM clip_embeddings WHERE asset_id IN ({placeholders})",
            chunk,
        ):
            vec = np.frombuffer(blob, dtype="<f4")
            vectors[aid] = vec / (np.linalg.norm(vec) or 1.0)

    out = {aid: UNCLASSIFIED_FOLDER for aid in asset_ids}
    if not vectors:
        return out
    text_matrix = category_text_matrix(categories, encode_texts)
    ids = list(vectors)
    dim = text_matrix.shape[1]
    ids = [aid for aid in ids if vectors[aid].shape[0] == dim]
    if not ids:
        return out
    best, _conf = classify_embeddings(
        np.stack([vectors[aid] for aid in ids]), text_matrix, min_confidence
    )
    for aid, idx in zip(ids, best, strict=True):
        out[aid] = categories[idx].folder if idx >= 0 else OTHER_FOLDER
    return out


# ---------------------------------------------------------------------------
# Trips and places
# ---------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _date(ts: int | None, fmt: str) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).strftime(fmt)


def _taken_at(conn: sqlite3.Connection, asset_ids: Sequence[int]) -> dict[int, int | None]:
    out: dict[int, int | None] = {}
    for chunk in _chunks(list(asset_ids), 500):
        placeholders = ",".join("?" * len(chunk))
        for aid, ts in conn.execute(
            f"SELECT id, taken_at FROM assets WHERE id IN ({placeholders})", chunk
        ):
            out[aid] = ts
    return out


def trip_folders(conn: sqlite3.Connection, asset_ids: Sequence[int]) -> dict[int, str]:
    """Map each asset to ``<start date> <trip name>[/<event name>]``.

    Photos in a top-level trip or event get that group's folder; photos in an
    event nested inside a trip get a sub-folder for the event.  Everything else
    goes to ``Not in a trip/<year>``.
    """
    groups: dict[int, dict[str, Any]] = {}
    membership: dict[int, list[int]] = {}
    if _table_exists(conn, "events"):
        for row in conn.execute(
            "SELECT id, parent_id, COALESCE(custom_name, auto_name), start_ts FROM events"
        ):
            groups[row[0]] = {"parent": row[1], "name": row[2], "start": row[3]}
        for chunk in _chunks(list(asset_ids), 500):
            placeholders = ",".join("?" * len(chunk))
            for aid, eid in conn.execute(
                f"SELECT asset_id, event_id FROM event_assets WHERE asset_id IN ({placeholders})",
                chunk,
            ):
                membership.setdefault(aid, []).append(eid)

    def _label(eid: int) -> str:
        g = groups[eid]
        day = _date(g["start"], "%Y-%m-%d")
        return sanitize_segment(f"{day} {g['name']}" if day else g["name"])

    taken = _taken_at(conn, asset_ids)
    out: dict[int, str] = {}
    for aid in asset_ids:
        eids = [e for e in membership.get(aid, []) if e in groups]
        top = [e for e in eids if groups[e]["parent"] is None]
        nested = [e for e in eids if groups[e]["parent"] is not None]
        if nested:
            child = nested[0]
            out[aid] = f"{_label(groups[child]['parent'])}/{_label(child)}"
        elif top:
            out[aid] = _label(top[0])
        else:
            year = _date(taken.get(aid), "%Y") or "Unknown date"
            out[aid] = f"{NO_TRIP_FOLDER}/{year}"
    return out


def place_folders(conn: sqlite3.Connection, asset_ids: Sequence[int]) -> dict[int, str]:
    """Map each asset to ``<Country>/<City>`` from the reverse-geocoded places."""
    places: dict[int, tuple[str | None, str | None, str | None]] = {}
    if _table_exists(conn, "asset_places"):
        for chunk in _chunks(list(asset_ids), 500):
            placeholders = ",".join("?" * len(chunk))
            for aid, country, admin1, city in conn.execute(
                "SELECT asset_id, country, admin1, city FROM asset_places"
                f" WHERE country IS NOT NULL AND asset_id IN ({placeholders})",
                chunk,
            ):
                places[aid] = (country, admin1, city)
    out: dict[int, str] = {}
    for aid in asset_ids:
        place = places.get(aid)
        if place is None:
            out[aid] = NO_PLACE_FOLDER
            continue
        country, admin1, city = place
        local = city or admin1 or "Unknown area"
        out[aid] = f"{sanitize_segment(country or 'Unknown country')}/{sanitize_segment(local)}"
    return out


def assign_folders(
    conn: sqlite3.Connection,
    asset_ids: Sequence[int],
    mode: str,
    *,
    categories: Sequence[Category] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    encode_texts: Callable[[Sequence[str]], np.ndarray] = encode_texts_clip,
) -> dict[int, str]:
    """Return ``{asset_id: relative folder}`` for *mode* (``""`` for ``flat``)."""
    if mode == "flat":
        return dict.fromkeys(asset_ids, "")
    if mode == "category":
        return category_folders(
            conn,
            asset_ids,
            categories or DEFAULT_CATEGORIES,
            min_confidence=min_confidence,
            encode_texts=encode_texts,
        )
    if mode == "trip":
        return trip_folders(conn, asset_ids)
    if mode == "place":
        return place_folders(conn, asset_ids)
    raise ValueError(f"Unknown folder mode {mode!r}; expected one of {', '.join(FOLDER_MODES)}")


def _chunks(items: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
