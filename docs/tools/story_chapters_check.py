"""Calibration check for story chapters (issue #109) on your own library (read-only).

Run with the app's venv python:

    python docs/tools/story_chapters_check.py OUT_DIR [N_TRIPS]

Reads ~/.takeout_rater/config.json for the current library, takes the stored
trips (or detects them in memory when none are stored), splits the largest
N_TRIPS (default 12) into chapters at every sensitivity and writes:

* ``summary.csv``: per trip and sensitivity, photos, chapters, threshold, sizes
* ``scores.txt``: percentiles of break scores across all trips, split into
  gaps that became chapter starts and gaps that did not
* ``trip_<id>.jpg``: a contact sheet per trip at "normal" sensitivity, one row
  per chapter (label, time, up to 8 evenly spaced thumbnails)

Chapter labels need the CLIP text model; pass ``--no-labels`` to skip them.
Nothing is written to the library database.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from takeout_rater.events import chapters as ch
from takeout_rater.events.detector import PhotoPoint, detect
from takeout_rater.indexing.thumbnailer import thumb_path_for_id

args = [a for a in sys.argv[1:] if not a.startswith("--")]
want_labels = "--no-labels" not in sys.argv
out = Path(args[0])
n_trips = int(args[1]) if len(args) > 1 else 12
out.mkdir(parents=True, exist_ok=True)

cfg = json.loads((Path.home() / ".takeout_rater" / "config.json").read_text(encoding="utf-8"))
state = Path(cfg["db_root"]) / "takeout-rater"
db = state / "library.sqlite"
if not db.exists():
    db = next(state.glob("*.sqlite"), None) or next(state.glob("*.db"))
thumbs = state / "thumbs"
print("db", db)
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def _blob(aid_blobs: list[tuple[int, bytes]]) -> dict[int, np.ndarray]:
    return {a: np.frombuffer(b, dtype="<f4") for a, b in aid_blobs if len(b) == 3072}


# ── Trips: stored ones, else detect in memory ────────────────────────────────
trips: list[dict] = []
try:
    rows = conn.execute(
        "SELECT id, COALESCE(custom_name, auto_name), n_days, center_lon FROM events"
        " WHERE kind = 'trip'"
    ).fetchall()
    for eid, name, n_days, lon in rows:
        photos = conn.execute(
            "SELECT a.id, a.taken_at FROM event_assets ea JOIN assets a ON a.id = ea.asset_id"
            " WHERE ea.event_id = ? AND a.taken_at IS NOT NULL",
            (eid,),
        ).fetchall()
        trips.append({"id": eid, "name": name, "n_days": n_days, "lon": lon, "photos": photos})
except sqlite3.Error as exc:
    print("no stored trips:", exc)
if not trips:
    print("detecting trips in memory…")
    pts = []
    for r in conn.execute(
        "SELECT id, taken_at, geo_lat, geo_lon FROM assets"
        " WHERE taken_at IS NOT NULL AND COALESCE(trashed, 0) = 0"
    ):
        lat, lon = r[2], r[3]
        if lat is None or lon is None or (lat == 0 and lon == 0):
            lat = lon = None
        pts.append(PhotoPoint(r[0], r[1], lat, lon))
    ts_of = {p.asset_id: p.ts for p in pts}
    for i, g in enumerate(detect(pts).trips):
        trips.append(
            {
                "id": f"mem{i}",
                "name": f"trip {i}",
                "n_days": g.n_days,
                "lon": g.center_lon,
                "photos": [(a, ts_of[a]) for a in g.asset_ids],
            }
        )
print("trips:", len(trips))
trips.sort(key=lambda t: -len(t["photos"]))
trips = trips[:n_trips]

vocab = None
if want_labels:
    try:
        import torch

        from takeout_rater.scoring.scorers.clip_backbone import get_clip_model
        from takeout_rater.scoring.scorers.clip_vocab import CLIP_VOCAB_TERMS

        model, _p, tok, dev = get_clip_model()
        terms = list(CLIP_VOCAB_TERMS)
        with torch.no_grad():
            f = model.encode_text(tok(terms).to(dev))
            f = (f / f.norm(dim=-1, keepdim=True)).cpu().float().numpy()
        vocab = (f, terms)
    except Exception as exc:  # noqa: BLE001
        print("labels unavailable:", exc)

summary = []
cut_scores: list[float] = []
other_scores: list[float] = []
for trip in trips:
    ids = [p[0] for p in trip["photos"]]
    qmarks = ",".join("?" * len(ids))
    emb = _blob(
        conn.execute(
            f"SELECT asset_id, embedding FROM clip_embeddings WHERE asset_id IN ({qmarks})", ids
        ).fetchall()
    )
    for sens in ("fewer", "normal", "more"):
        t0 = time.perf_counter()
        story = ch.split_chapters(
            trip["photos"],
            emb,
            ch.params_for(sens),
            min_photos=ch.default_min_photos(len(ids), sens),
        )
        ms = (time.perf_counter() - t0) * 1000
        sizes = [len(c.asset_ids) for c in story.chapters]
        summary.append(
            [
                trip["id"],
                trip["name"],
                len(ids),
                len(emb),
                trip["n_days"],
                sens,
                len(sizes),
                story.threshold,
                round(ms, 1),
                " ".join(map(str, sizes)),
            ]
        )
        if sens != "normal":
            continue
        starts = {c.start - 1 for c in story.chapters if c.start > 0}
        for g, s in enumerate(story.scores):
            (cut_scores if g in starts else other_scores).append(s)
        labels = (
            ch.label_chapters(story.chapters, vocab[0], vocab[1], allowed=ch.label_terms())
            if vocab
            else [[] for _ in story.chapters]
        )
        tw, th, lw = 160, 120, 300
        sheet = Image.new("RGB", (lw + 8 * tw, th * len(story.chapters)), "white")
        d = ImageDraw.Draw(sheet)
        font = None
        for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
            try:  # the default bitmap font has no en dash or arrow
                font = ImageFont.truetype(name, 13)
                break
            except OSError:
                pass
        for row, c in enumerate(story.chapters):
            y = row * th
            title = ch.chapter_title(labels[row]) or f"Chapter {row + 1}"
            tl = ch.time_label(c.start_ts, c.end_ts, story.chapters[0].start_ts, trip["lon"])
            brk = f"break {c.break_score:.3f}" if c.break_score is not None else "start"
            d.text((6, y + 8), f"{row + 1}. {title}", fill="black", font=font)
            d.text((6, y + 28), f"{tl}", fill="gray", font=font)
            d.text((6, y + 48), f"{len(c.asset_ids)} photos, {brk}", fill="gray", font=font)
            pick = np.linspace(0, len(c.asset_ids) - 1, min(8, len(c.asset_ids))).astype(int)
            for k, idx in enumerate(pick):
                p = thumb_path_for_id(thumbs, c.asset_ids[idx])
                if p.exists():
                    im = Image.open(p).convert("RGB")
                    im.thumbnail((tw - 4, th - 4))
                    sheet.paste(im, (lw + k * tw, y + 2))
            d.line([(0, y + th - 1), (sheet.width, y + th - 1)], fill="#ccc")
        sheet.save(out / f"trip_{trip['id']}.jpg", quality=82)

with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(
        [
            "trip",
            "name",
            "photos",
            "with_emb",
            "days",
            "sensitivity",
            "chapters",
            "threshold",
            "ms",
            "sizes",
        ]
    )
    w.writerows(summary)
with open(out / "scores.txt", "w", encoding="utf-8") as fh:
    for name, arr in (("chapter starts", cut_scores), ("other gaps", other_scores)):
        if arr:
            q = np.percentile(arr, [5, 25, 50, 75, 95, 99])
            fh.write(f"{name} (n={len(arr)}): p5/25/50/75/95/99 = {np.round(q, 3).tolist()}\n")
print((out / "scores.txt").read_text(encoding="utf-8"))
for row in summary:
    print(row)
