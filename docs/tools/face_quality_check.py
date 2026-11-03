"""Calibration check for the face_quality scorer on your own library (read-only).

Run with the app's venv python:

    python docs/tools/face_quality_check.py OUT_DIR [N_SAMPLE]

Reads ~/.takeout_rater/config.json for the current library, samples thumbnails
that have face detections (falls back to random assets), rates every face and
writes per-face CSV, timing, and labelled face-crop contact sheets to OUT_DIR.
Nothing is written to the library database.
"""

from __future__ import annotations

import csv
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from takeout_rater.faces.quality import aggregate_faces, select_subject_faces
from takeout_rater.indexing.thumbnailer import thumb_path_for_id
from takeout_rater.scoring.scorers.face_quality import FaceQualityScorer, _decode

out = Path(sys.argv[1])
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 400
out.mkdir(parents=True, exist_ok=True)

cfg = json.loads((Path.home() / ".takeout_rater" / "config.json").read_text(encoding="utf-8"))
db_root = Path(cfg["db_root"])
state = db_root / "takeout-rater"
db = state / "library.sqlite"
if not db.exists():
    db = next(state.glob("*.sqlite"), None) or next(state.glob("*.db"))
thumbs = state / "thumbs"
print("db", db, "thumbs", thumbs)
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

random.seed(42)
ids: list[int] = []
try:
    rows = conn.execute(
        "SELECT asset_id, COUNT(*) FROM face_embeddings GROUP BY asset_id"
    ).fetchall()
    print("assets with faces:", len(rows))
    by_n: dict[int, list[int]] = {}
    for aid, n in rows:
        by_n.setdefault(min(n, 5), []).append(aid)
    per = max(1, n_sample // max(1, len(by_n)))
    for _n, lst in sorted(by_n.items()):
        ids += random.sample(lst, min(per, len(lst)))
except sqlite3.Error as exc:
    print("no face_embeddings:", exc)
if len(ids) < n_sample:
    all_ids = [r[0] for r in conn.execute("SELECT id FROM assets").fetchall()]
    ids += random.sample(all_ids, min(n_sample - len(ids), len(all_ids)))

# Bursts: runs of >= 3 assets with faces taken within 3 s of each other.
bursts: list[list[int]] = []
try:
    face_assets = {r[0] for r in conn.execute("SELECT DISTINCT asset_id FROM face_embeddings")}
    seq = conn.execute(
        "SELECT id, taken_at FROM assets WHERE taken_at IS NOT NULL ORDER BY taken_at, id"
    ).fetchall()
    run: list[int] = []
    last = None
    for aid, ts in seq:
        if last is not None and ts - last <= 3 and aid in face_assets:
            run.append(aid)
        else:
            if len(run) >= 3:
                bursts.append(run)
            run = [aid] if aid in face_assets else []
        last = ts
    if len(run) >= 3:
        bursts.append(run)
except sqlite3.Error:
    pass
random.shuffle(bursts)
bursts = [b[:8] for b in bursts[:12]]
print("bursts:", len(bursts))
burst_ids = [a for b in bursts for a in b]

scorer = FaceQualityScorer.create()
t0 = time.perf_counter()
scorer._ensure_loaded()
print("load s", round(time.perf_counter() - t0, 2))
providers = scorer._fer.get_providers()
det_prov = scorer._app.det_model.session.get_providers()
print("providers fer", providers, "det", det_prov)

records = []
photo_rows = []
images = {}
times = []
for aid in dict.fromkeys(ids + burst_ids):
    p = thumb_path_for_id(thumbs, aid)
    img = _decode(p)
    if img is None:
        continue
    t = time.perf_counter()
    faces = scorer.analyze_image(img)
    times.append(time.perf_counter() - t)
    agg = aggregate_faces(faces)
    subj = set(id(f) for f in select_subject_faces(faces))
    photo_rows.append({"asset_id": aid, **agg})
    images[aid] = (img, faces, agg)
    for i, f in enumerate(faces):
        records.append(
            dict(
                asset_id=aid,
                face=i,
                subject=int(id(f) in subj),
                h=round(f.height, 1),
                eyes=round(f.eyes_open, 3),
                ear_l=round(f.ear_left, 3),
                ear_r=round(f.ear_right, 3),
                smile=round(f.smile, 3),
                facing=round(f.facing_camera, 3),
                yaw=round(f.yaw, 1),
                pitch=round(f.pitch, 1),
                sharp=round(f.sharpness, 3),
                sharp_raw=round(f.sharpness_raw, 3),
                visible=round(f.visible, 3),
                in_frame=round(f.in_frame, 3),
                det=round(f.det_score, 3),
                quality=round(f.quality, 3),
            )
        )
times_ms = np.array(times[3:] or times) * 1000
summary = dict(
    images=len(times),
    faces=len(records),
    ms_median=float(np.median(times_ms)),
    ms_p90=float(np.percentile(times_ms, 90)),
    providers_fer=providers,
    providers_det=det_prov,
)
print(summary)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
with open(out / "faces.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
keys = sorted({k for r in photo_rows for k in r})
with open(out / "photos.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(photo_rows)


def crop(aid: int, i: int, size: int = 128) -> np.ndarray:
    img, faces, _ = images[aid]
    f = faces[i]
    x1, y1, x2, y2 = f.bbox
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.75
    m = np.array(
        [
            [size / (2 * s), 0, size / 2 - cx * size / (2 * s)],
            [0, size / (2 * s), size / 2 - cy * size / (2 * s)],
        ],
        np.float32,
    )
    c = cv2.warpAffine(img, m, (size, size + 40), borderValue=(40, 40, 40))
    lines = [
        f"e{f.eyes_open:.2f} s{f.smile:.2f} f{f.facing_camera:.2f}",
        f"sh{f.sharpness:.2f} v{f.visible:.2f} q{f.quality:.2f}",
        f"{aid}/{i} h{f.height:.0f}",
    ]
    for j, t in enumerate(lines):
        cv2.putText(c, t, (2, size + 11 + 13 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
    return c


def sheet(name: str, rows: list[dict], cols: int = 8) -> None:
    tiles = [crop(r["asset_id"], r["face"]) for r in rows]
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out / f"{name}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


subj = [r for r in records if r["subject"]]
for key in ("eyes", "smile", "facing", "sharp", "visible", "quality"):
    s = sorted(subj, key=lambda r: r[key])
    sheet(f"{key}_lowest", s[:40])
    sheet(f"{key}_highest", s[-24:])
mid = [r for r in subj if 0.2 < r["eyes"] < 0.8]
sheet("eyes_middle", mid[:40])
random.shuffle(subj)
sheet("random_faces", subj[:48])

# Burst sheets: whole thumbnails in a row, annotated, best (face_quality) marked.
for bi, b in enumerate(bursts):
    tiles = []
    best = max((images[a][2].get("face_quality", -1), a) for a in b if a in images)[1]
    for a in b:
        if a not in images:
            continue
        img, faces, agg = images[a]
        t = img.copy()
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            col = (0, 255, 0) if f.eyes_open > 0.5 else (0, 0, 255)
            cv2.rectangle(t, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                t,
                f"e{f.eyes_open:.1f} q{f.quality:.2f}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                1,
            )
        t = cv2.resize(t, (int(t.shape[1] * 300 / t.shape[0]), 300))
        label = (
            f"{a} fq{agg.get('face_quality', -1):.2f} worst{agg.get('face_quality_worst', -1):.2f}"
        )
        cv2.putText(
            t,
            label + (" BEST" if a == best else ""),
            (4, 292),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        tiles.append(t)
    if tiles:
        cv2.imwrite(
            str(out / f"burst_{bi:02d}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
print("done", out)
