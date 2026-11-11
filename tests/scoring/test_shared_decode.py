"""Tests for shared-decode scoring passes and stored-embedding reuse."""

from __future__ import annotations

import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

import pytest

from takeout_rater.db.queries import get_asset_scores, upsert_asset
from takeout_rater.db.schema import migrate
from takeout_rater.indexing.thumbnailer import thumb_path_for_id
from takeout_rater.scoring import shared_decode
from takeout_rater.scoring.pipeline import (
    plan_scorer_groups,
    run_scorer,
    run_scorer_group,
)
from takeout_rater.scoring.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec
from takeout_rater.scoring.scorers.simple import SimpleScorer


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return conn


def _add_assets(conn: sqlite3.Connection, thumbs_dir: Path, n: int, *, thumbs: bool = True):
    from PIL import Image

    ids = []
    for i in range(n):
        aid = upsert_asset(
            conn,
            {
                "relpath": f"Photos/img{i}.jpg",
                "filename": f"img{i}.jpg",
                "ext": ".jpg",
                "size_bytes": 1,
                "mime": "image/jpeg",
                "indexed_at": int(time.time()),
            },
        )
        ids.append(aid)
        if thumbs:
            thumb = thumb_path_for_id(thumbs_dir, aid)
            thumb.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40 + i, 30), color=(10 * i % 255, 120, 200)).save(thumb, "JPEG")
    return ids


# ── shared_decode.open_rgb ───────────────────────────────────────────────────


def test_open_rgb_decodes_without_cache(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "a.jpg"
    Image.new("RGB", (8, 6), color=(1, 2, 3)).save(path, "PNG")
    img = shared_decode.open_rgb(path)
    assert img.mode == "RGB" and img.size == (8, 6)


def test_open_rgb_uses_shared_copy(tmp_path: Path) -> None:
    from PIL import Image

    cached = Image.new("RGB", (4, 4), color=(9, 9, 9))
    path = tmp_path / "not-on-disk.jpg"
    with shared_decode.shared_images({str(path): cached}):
        got = shared_decode.open_rgb(path)
        assert got is not cached
        assert got.tobytes() == cached.tobytes()
    with pytest.raises(OSError):
        shared_decode.open_rgb(path)  # cache uninstalled → reads the (missing) file


# ── plan_scorer_groups ───────────────────────────────────────────────────────


def test_plan_groups_packs_by_budget_and_keeps_order() -> None:
    costs = {"a": (0, None), "b": (3000, None), "c": (2000, None), "d": (900, None), "e": (0, None)}
    groups = plan_scorer_groups(list(costs), lambda k: costs[k], 4000)
    assert groups == [["a", "b", "d", "e"], ["c"]]


def test_plan_groups_counts_shared_model_once() -> None:
    costs = {"laion": (1750, "clip"), "iqa": (1750, "clip"), "v25": (1700, None)}
    groups = plan_scorer_groups(list(costs), lambda k: costs[k], 3500)
    assert groups == [["laion", "iqa", "v25"]]


def test_plan_groups_disabled_and_oversized() -> None:
    costs = {"a": (100, None), "b": (100, None)}
    assert plan_scorer_groups(list(costs), lambda k: costs[k], 0) == [["a"], ["b"]]
    big = {"x": (9000, None), "y": (10, None)}
    assert plan_scorer_groups(list(big), lambda k: big[k], 1000) == [["x"], ["y"]]


def test_registered_scorers_have_sane_estimates() -> None:
    from takeout_rater.scoring.scorers.registry import list_scorers

    for cls in list_scorers():
        for v in cls.spec().variants:
            assert cls.gpu_memory_estimate_mb(v.variant_id) >= 0
    assert SimpleScorer.gpu_memory_estimate_mb("blur") == 0


# ── run_scorer_group ─────────────────────────────────────────────────────────


def test_group_decodes_each_thumbnail_once_and_matches_single_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    thumbs = tmp_path / "thumbs"
    conn_single = _conn()
    conn_group = _conn()
    ids = _add_assets(conn_single, thumbs, 5)
    assert _add_assets(conn_group, thumbs, 5, thumbs=False) == ids

    for vid in ("blur", "luminosity", "noise"):
        run_scorer(conn_single, SimpleScorer.create(variant_id=vid), thumbs)

    decoded: list[str] = []
    real_decode = shared_decode.decode_rgb

    def _counting_decode(path: Path) -> Any:
        decoded.append(Path(path).name)
        return real_decode(path)

    monkeypatch.setattr(shared_decode, "decode_rgb", _counting_decode)
    scorers = [SimpleScorer.create(variant_id=v) for v in ("blur", "luminosity", "noise")]
    progress: list[tuple[int, int]] = []
    counts = run_scorer_group(
        conn_group, scorers, thumbs, batch_size=2, on_progress=lambda d, t: progress.append((d, t))
    )

    assert counts == {("simple", "blur"): 5, ("simple", "luminosity"): 5, ("simple", "noise"): 5}
    assert sorted(decoded) == sorted({n for n in decoded})  # each thumbnail once
    assert len(decoded) == 5
    assert progress[-1] == (5, 5)
    for aid in ids:
        single = {
            (r["variant_id"], r["metric_key"]): r["value"]
            for r in get_asset_scores(conn_single, aid)
        }
        group = {
            (r["variant_id"], r["metric_key"]): r["value"]
            for r in get_asset_scores(conn_group, aid)
        }
        assert group == single


def test_group_skips_already_scored_per_scorer(tmp_path: Path) -> None:
    thumbs = tmp_path / "thumbs"
    conn = _conn()
    _add_assets(conn, thumbs, 3)
    run_scorer(conn, SimpleScorer.create(variant_id="blur"), thumbs)
    counts = run_scorer_group(
        conn, [SimpleScorer.create(variant_id=v) for v in ("blur", "noise")], thumbs
    )
    assert counts == {("simple", "blur"): 0, ("simple", "noise"): 3}


def test_group_respects_cancel(tmp_path: Path) -> None:
    thumbs = tmp_path / "thumbs"
    conn = _conn()
    _add_assets(conn, thumbs, 4)
    counts = run_scorer_group(
        conn, [SimpleScorer.create(variant_id="blur")], thumbs, cancel_check=lambda: True
    )
    assert counts == {("simple", "blur"): 0}


# ── stored CLIP embeddings ───────────────────────────────────────────────────


class _EmbeddingScorer(BaseScorer):
    consumes_clip_embeddings = True

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        self.image_calls: list[int] = []
        self.embedding_rows = 0

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="emb_test",
            display_name="Embedding test",
            variants=(
                VariantSpec(
                    variant_id="default",
                    display_name="d",
                    metrics=(MetricSpec(key="m", display_name="m"),),
                ),
            ),
        )

    @classmethod
    def is_available(cls) -> bool:
        return True

    def score_batch(self, image_paths, *, variant_id=None):  # type: ignore[no-untyped-def]
        self.image_calls.append(len(image_paths))
        return [{"m": -1.0} for _ in image_paths]

    def score_clip_embeddings(self, embeddings):  # type: ignore[no-untyped-def]
        self.embedding_rows += len(embeddings)
        return [{"m": float(row[0])} for row in embeddings]


def _store_embedding(conn: sqlite3.Connection, aid: int, first: float) -> None:
    vec = [first] + [0.0] * 767
    conn.execute(
        "INSERT INTO clip_embeddings (asset_id, embedding, computed_at) VALUES (?, ?, 0)",
        (aid, struct.pack("768f", *vec)),
    )


@pytest.mark.parametrize("grouped", [False, True])
def test_embedding_scorer_reads_stored_vectors_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grouped: bool
) -> None:
    thumbs = tmp_path / "thumbs"
    conn = _conn()
    ids = _add_assets(conn, thumbs, 3)
    _store_embedding(conn, ids[0], 0.25)
    _store_embedding(conn, ids[2], 0.75)

    decoded: list[Path] = []
    real_decode = shared_decode.decode_rgb
    monkeypatch.setattr(shared_decode, "decode_rgb", lambda p: decoded.append(p) or real_decode(p))
    scorer = _EmbeddingScorer.create()
    if grouped:
        run_scorer_group(conn, [scorer], thumbs)
        assert len(decoded) == 1  # only the asset without an embedding
    else:
        run_scorer(conn, scorer, thumbs)

    assert scorer.embedding_rows == 2
    assert scorer.image_calls == [1]
    values = {aid: get_asset_scores(conn, aid)[0]["value"] for aid in ids}
    assert values == {ids[0]: 0.25, ids[1]: -1.0, ids[2]: 0.75}
