"""Tests for OCR text extraction and FTS5 text search.

Covers:
- schema migration to v17 (ocr_text + ocr_text_fts + triggers)
- OCR DB queries (upsert keeps FTS in sync, search ranking, query escaping)
- OcrEngine result filtering (with a fake RapidOCR backend)
- /api/jobs/ocr/start with a fake engine, /api/search?mode=text, /api/ocr/status
- the Text tab on the asset detail page
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from takeout_rater.db.connection import open_library_db
from takeout_rater.db.queries import (
    OCR_SNIPPET_END,
    OCR_SNIPPET_START,
    build_ocr_fts_query,
    clear_ocr_text,
    get_ocr_stats,
    get_ocr_text,
    list_asset_ids_without_ocr,
    parse_ocr_query,
    search_ocr_text,
    upsert_asset,
    upsert_ocr_texts,
)
from takeout_rater.ui.app import create_app


def _add_asset(conn, name: str) -> int:
    aid = upsert_asset(
        conn,
        {
            "relpath": f"Photos/{name}",
            "filename": name,
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": 1,
        },
    )
    conn.commit()
    return aid


def _ocr_row(aid: int, text: str) -> tuple[int, str, int, float | None, str, str]:
    n = len([line for line in text.splitlines() if line])
    return (aid, text, n, 0.9 if n else None, "thumbnail", "test")


@pytest.fixture()
def conn(tmp_path: Path):
    c = open_library_db(tmp_path)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_fresh_db_has_ocr_tables(self, conn) -> None:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
            )
        }
        assert {"ocr_text", "ocr_text_fts", "ocr_text_ai", "ocr_text_ad", "ocr_text_au"} <= names

    def test_incremental_ocr_migration(self, tmp_path: Path) -> None:
        import sqlite3

        from takeout_rater.db.schema import CURRENT_SCHEMA_VERSION, migrate

        db = sqlite3.connect(tmp_path / "lib.sqlite")
        migrate(db)
        # Roll the fresh DB back to the version just before the OCR migration.
        for trig in ("ocr_text_ai", "ocr_text_ad", "ocr_text_au"):
            db.execute(f"DROP TRIGGER {trig}")
        db.execute("DROP TABLE ocr_text_fts")
        db.execute("DROP TABLE ocr_text")
        db.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION - 1}")
        db.commit()

        migrate(db)
        assert db.execute("PRAGMA user_version").fetchone()[0] == CURRENT_SCHEMA_VERSION
        db.execute(
            "INSERT INTO assets (relpath, filename, ext, indexed_at) VALUES ('a', 'a', '.jpg', 0)"
        )
        db.execute(
            "INSERT INTO ocr_text (asset_id, text, text_folded, source, engine, processed_at)"
            " VALUES (1, 'hello world', 'hello world', 'thumbnail', 't', 0)"
        )
        hits = db.execute(
            "SELECT rowid FROM ocr_text_fts WHERE ocr_text_fts MATCH 'hello'"
        ).fetchall()
        assert hits == [(1,)]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueryParsing:
    def test_whitespace_and_phrases(self) -> None:
        assert parse_ocr_query('gleis 7 "happy  birthday" ') == ["gleis", "7", "happy birthday"]

    def test_fts_expression_quotes_terms_and_skips_short_ones(self) -> None:
        terms = parse_ocr_query("a OR NEAR(x) 9.50")
        assert build_ocr_fts_query(terms) == '"NEAR(x)" "9.50"'

    def test_empty(self) -> None:
        assert parse_ocr_query("   ") == []
        assert build_ocr_fts_query(["ab"]) == ""


class TestOcrQueries:
    def test_search_and_update_keep_fts_in_sync(self, conn) -> None:
        a = _add_asset(conn, "a.jpg")
        b = _add_asset(conn, "b.jpg")
        c = _add_asset(conn, "c.jpg")
        upsert_ocr_texts(
            conn,
            [
                _ocr_row(a, "Welcome to Berlin\nHauptbahnhof"),
                _ocr_row(b, ""),
                _ocr_row(c, "Café Größe"),
            ],
        )

        results, total = search_ocr_text(conn, "berl")
        assert total == 1
        assert results[0]["asset_id"] == a
        assert f"{OCR_SNIPPET_START}Berl{OCR_SNIPPET_END}in" in results[0]["snippet"]

        # Substring + case-insensitive: "bahn" finds "Hauptbahnhof".
        assert [r["asset_id"] for r in search_ocr_text(conn, "BAHN")[0]] == [a]
        assert [r["asset_id"] for r in search_ocr_text(conn, "größe")[0]] == [c]
        # Diacritics are folded both ways: "cafe" finds "Café", "café" finds it too.
        assert [r["asset_id"] for r in search_ocr_text(conn, "cafe")[0]] == [c]
        assert [r["asset_id"] for r in search_ocr_text(conn, "CAFÉ")[0]] == [c]
        # Non-diacritic decompositions are kept: Hangul syllables don't collapse.
        from takeout_rater.db.queries import fold_for_search

        assert fold_for_search("가나다") == "가나다"
        assert fold_for_search("Ångström") == "Angstrom"
        # casefold maps ß → ss, and the highlighter marks the ß spelling.
        hits = search_ocr_text(conn, "grosse")[0]
        assert [h["asset_id"] for h in hits] == [c]
        assert f"{OCR_SNIPPET_START}Größe{OCR_SNIPPET_END}" in hits[0]["snippet"]
        hit = search_ocr_text(conn, "große")[0][0]
        assert f"{OCR_SNIPPET_START}Größe{OCR_SNIPPET_END}" in hit["snippet"]
        # Short terms are applied as LIKE filters.
        assert [r["asset_id"] for r in search_ocr_text(conn, "to")[0]] == [a]
        assert search_ocr_text(conn, "berlin xy") == ([], 0)
        assert search_ocr_text(conn, '"to berlin"')[1] == 1
        assert search_ocr_text(conn, '"berlin to"')[1] == 0
        # FTS operators are treated as text.
        assert search_ocr_text(conn, "OR -") == ([], 0)

        # Updating a row replaces its index entry.
        upsert_ocr_texts(conn, [_ocr_row(a, "Munich")])
        assert search_ocr_text(conn, "berlin") == ([], 0)
        assert search_ocr_text(conn, "munich")[1] == 1

        assert get_ocr_stats(conn) == {"total_assets": 3, "processed": 3, "with_text": 2}
        assert list_asset_ids_without_ocr(conn) == []
        assert get_ocr_text(conn, b)["text"] == ""

    def test_asset_delete_cascades_to_fts(self, conn) -> None:
        a = _add_asset(conn, "a.jpg")
        upsert_ocr_texts(conn, [_ocr_row(a, "receipt total")])
        conn.execute("DELETE FROM assets WHERE id = ?", (a,))
        conn.commit()
        assert search_ocr_text(conn, "receipt") == ([], 0)

    def test_clear(self, conn) -> None:
        a = _add_asset(conn, "a.jpg")
        upsert_ocr_texts(conn, [_ocr_row(a, "stop")])
        clear_ocr_text(conn)
        assert search_ocr_text(conn, "stop") == ([], 0)
        assert list_asset_ids_without_ocr(conn) == [a]

    def test_ranking_and_pagination(self, conn) -> None:
        ids = [_add_asset(conn, f"{i}.jpg") for i in range(5)]
        rows = [_ocr_row(aid, "sale " * (i + 1) + "filler words here") for i, aid in enumerate(ids)]
        upsert_ocr_texts(conn, rows)
        page1, total = search_ocr_text(conn, "sale", limit=2, offset=0)
        page2, _ = search_ocr_text(conn, "sale", limit=2, offset=2)
        assert total == 5
        assert len(page1) == 2 and len(page2) == 2
        assert not {r["asset_id"] for r in page1} & {r["asset_id"] for r in page2}
        # BM25 favours the row that repeats the term most.
        assert page1[0]["asset_id"] == ids[-1]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestOcrEngine:
    def test_filters_low_confidence_and_blank_lines(self, monkeypatch) -> None:
        import sys
        import types

        captured: dict[str, object] = {}

        class _FakeRapidOCR:
            def __init__(self, params: dict) -> None:
                captured.update(params)

            def __call__(self, img):
                return SimpleNamespace(
                    txts=("STOP", "  ", "blurry", "Main St"),
                    scores=(0.98, 0.99, 0.3, 0.8),
                )

        monkeypatch.setitem(sys.modules, "rapidocr", types.SimpleNamespace(RapidOCR=_FakeRapidOCR))
        monkeypatch.setattr("takeout_rater.ocr.engine.cuda_available", lambda: False)
        from takeout_rater.ocr.engine import OcrEngine

        engine = OcrEngine(accelerator="gpu", min_score=0.6)
        assert engine.active_accelerator == "cpu"
        assert "EngineConfig.onnxruntime.use_cuda" not in captured
        result = engine.recognize(object())
        assert result.text == "STOP\nMain St"
        assert result.mean_score == pytest.approx(0.89)

    def test_empty_output(self, monkeypatch) -> None:
        import sys
        import types

        class _FakeRapidOCR:
            def __init__(self, params: dict) -> None:
                pass

            def __call__(self, img):
                return SimpleNamespace(txts=None, scores=None)

        monkeypatch.setitem(sys.modules, "rapidocr", types.SimpleNamespace(RapidOCR=_FakeRapidOCR))
        from takeout_rater.ocr.engine import OcrEngine

        result = OcrEngine(accelerator="cpu").recognize(object())
        assert result.text == "" and result.mean_score is None

    def test_load_image_bgr_downscales(self, tmp_path: Path) -> None:
        from takeout_rater.ocr.engine import load_image_bgr

        path = tmp_path / "big.png"
        Image.new("RGB", (4000, 1000), color=(255, 0, 0)).save(path)
        arr = load_image_bgr(path, max_side=2048)
        assert arr.shape == (512, 2048, 3)
        assert tuple(arr[0, 0]) == (0, 0, 255)  # BGR
        assert load_image_bgr(tmp_path / "missing.jpg") is None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture()
def sync_threads(monkeypatch):
    import takeout_rater.api.jobs as jobs_mod

    original_thread = threading.Thread

    class _SyncThread(original_thread):
        def start(self) -> None:
            self.run()

    monkeypatch.setattr(jobs_mod.threading, "Thread", _SyncThread)


def _make_client(tmp_path: Path) -> tuple[TestClient, object]:
    app = create_app(tmp_path, open_library_db(tmp_path), db_root=tmp_path)
    return TestClient(app, follow_redirects=False), app


def _write_thumb(tmp_path: Path, aid: int) -> None:
    from takeout_rater.indexing.thumbnailer import thumb_path_for_id

    thumb = thumb_path_for_id(tmp_path / "takeout-rater" / "thumbs", aid)
    thumb.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(200, 200, 200)).save(thumb, "JPEG")


class TestOcrJob:
    def test_rejects_bad_params(self, tmp_path: Path) -> None:
        client, _ = _make_client(tmp_path)
        assert client.post("/api/jobs/ocr/start", json={"source": "x"}).status_code == 400
        assert client.post("/api/jobs/ocr/start", json={"accelerator": "tpu"}).status_code == 400
        assert client.post("/api/jobs/ocr/start", json={"min_score": 2}).status_code == 400

    def test_cancel_without_job_is_404(self, tmp_path: Path) -> None:
        client, _ = _make_client(tmp_path)
        assert client.post("/api/jobs/ocr/cancel").status_code == 404

    def test_job_stores_text_and_skips_processed(
        self, tmp_path: Path, monkeypatch, sync_threads
    ) -> None:
        conn = open_library_db(tmp_path)
        a = _add_asset(conn, "sign.jpg")
        b = _add_asset(conn, "beach.jpg")
        c = _add_asset(conn, "nothumb.jpg")
        conn.close()
        _write_thumb(tmp_path, a)
        _write_thumb(tmp_path, b)

        from takeout_rater.ocr.engine import OcrLine, OcrResult

        calls: list[int] = []
        texts = {a: [OcrLine("Hauptbahnhof", 0.95)], b: []}

        class _FakeEngine:
            engine_id = "fake-ocr"

            def __init__(self, *, accelerator: str, min_score: float) -> None:
                self.active_accelerator = accelerator
                self._order = iter([a, b])

            def recognize(self, img) -> OcrResult:
                aid = next(self._order)
                calls.append(aid)
                return OcrResult(lines=texts[aid])

        monkeypatch.setattr("takeout_rater.ocr.engine.OcrEngine", _FakeEngine)
        client, app = _make_client(tmp_path)

        resp = client.post("/api/jobs/ocr/start", json={"accelerator": "cpu"})
        assert resp.status_code == 200
        progress = app.state.jobs["ocr"]
        assert progress.error is None, progress.message
        assert progress.done and "OCR complete" in progress.message
        assert progress.diagnostics["with_text"] == 1
        assert progress.diagnostics["unreadable"] == 1
        assert calls == [a, b]

        status = client.get("/api/ocr/status").json()
        assert status["processed"] == 2
        assert status["with_text"] == 1
        assert status["total_assets"] == 3

        # The asset without a thumbnail stays unprocessed for a later run.
        check = open_library_db(tmp_path)
        assert list_asset_ids_without_ocr(check) == [c]
        assert get_ocr_text(check, a)["engine"] == "fake-ocr"
        check.close()

        # Search finds it.
        data = client.get("/api/search", params={"q": "hauptbahn", "mode": "text"}).json()
        assert data["total"] == 1
        assert data["results"][0]["asset_id"] == a
        assert data["results"][0]["snippet"] == "Hauptbahnhof"

    def test_job_skips_when_everything_processed(
        self, tmp_path: Path, monkeypatch, sync_threads
    ) -> None:
        conn = open_library_db(tmp_path)
        a = _add_asset(conn, "a.jpg")
        upsert_ocr_texts(conn, [_ocr_row(a, "done")])
        conn.close()

        def _boom(**kwargs):
            raise AssertionError("engine should not load")

        monkeypatch.setattr("takeout_rater.ocr.engine.OcrEngine", _boom)
        client, app = _make_client(tmp_path)
        assert client.post("/api/jobs/ocr/start", json={}).status_code == 200
        assert app.state.jobs["ocr"].message == "All assets already have OCR text."


class TestTextSearchApi:
    def test_partial_escapes_and_highlights(self, tmp_path: Path) -> None:
        conn = open_library_db(tmp_path)
        a = _add_asset(conn, "a.jpg")
        upsert_ocr_texts(conn, [_ocr_row(a, "<script>alert(1)</script> OPEN")])
        conn.close()
        client, _ = _make_client(tmp_path)
        resp = client.get("/api/search", params={"q": "open", "mode": "text", "partial": "1"})
        assert resp.status_code == 200
        html = resp.text
        assert "<mark>OPEN</mark>" in html
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
        assert 'data-total-results="1"' in html

    def test_no_ocr_yet_flag(self, tmp_path: Path) -> None:
        client, _ = _make_client(tmp_path)
        resp = client.get("/api/search", params={"q": "x", "mode": "text", "partial": "1"})
        assert 'data-no-ocr="true"' in resp.text

    def test_bad_mode(self, tmp_path: Path) -> None:
        client, _ = _make_client(tmp_path)
        assert client.get("/api/search", params={"q": "x", "mode": "nope"}).status_code == 400

    def test_search_page_text_mode(self, tmp_path: Path) -> None:
        client, _ = _make_client(tmp_path)
        resp = client.get("/search", params={"mode": "text", "q": "menu"})
        assert resp.status_code == 200
        assert 'id="card-ocr"' in resp.text
        assert 'value="menu"' in resp.text

    def test_detail_page_shows_text_tab(self, tmp_path: Path) -> None:
        conn = open_library_db(tmp_path)
        a = _add_asset(conn, "a.jpg")
        b = _add_asset(conn, "b.jpg")
        upsert_ocr_texts(conn, [_ocr_row(a, "Gleis 7")])
        conn.close()
        client, _ = _make_client(tmp_path)
        html = client.get(f"/assets/{a}", params={"partial": "1"}).text
        assert 'data-tab="ocr"' in html and "Gleis 7" in html
        html_b = client.get(f"/assets/{b}", params={"partial": "1"}).text
        assert "Not read yet" in html_b


class TestReviewFixes:
    def test_unicode_case_folding_for_short_and_long_terms(self, conn) -> None:
        a = _add_asset(conn, "a.jpg")
        b = _add_asset(conn, "b.jpg")
        upsert_ocr_texts(conn, [_ocr_row(a, "ЖД вокзал"), _ocr_row(b, "Hauptstraße 5")])
        # Short (<3 chars) Cyrillic term goes through LIKE: must still ignore case.
        assert [r["asset_id"] for r in search_ocr_text(conn, "жд")[0]] == [a]
        assert [r["asset_id"] for r in search_ocr_text(conn, "ВОКЗАЛ")[0]] == [a]
        # casefold maps ß → ss in both directions.
        assert [r["asset_id"] for r in search_ocr_text(conn, "strasse")[0]] == [b]
        hits = search_ocr_text(conn, "STRASSE 5")[0]
        assert [r["asset_id"] for r in hits] == [b]
        assert f"{OCR_SNIPPET_START}straße{OCR_SNIPPET_END}" in hits[0]["snippet"]
        assert [r["asset_id"] for r in search_ocr_text(conn, "straße")[0]] == [b]

    def test_redo_overwrites_instead_of_clearing(
        self, tmp_path: Path, monkeypatch, sync_threads
    ) -> None:
        conn = open_library_db(tmp_path)
        a = _add_asset(conn, "a.jpg")
        b = _add_asset(conn, "b.jpg")
        upsert_ocr_texts(conn, [_ocr_row(a, "old a"), _ocr_row(b, "old b")])
        conn.close()
        _write_thumb(tmp_path, a)
        _write_thumb(tmp_path, b)

        from takeout_rater.ocr.engine import OcrLine, OcrResult

        holder: dict[str, object] = {}

        class _CancellingEngine:
            engine_id = "fake"

            def __init__(self, *, accelerator: str, min_score: float) -> None:
                self.active_accelerator = accelerator

            def recognize(self, img) -> OcrResult:
                # Cancel after the first asset: the second must keep its old text.
                holder["app"].state.jobs["ocr"].cancel_event.set()
                return OcrResult(lines=[OcrLine("new a", 0.9)])

        monkeypatch.setattr("takeout_rater.ocr.engine.OcrEngine", _CancellingEngine)
        client, app = _make_client(tmp_path)
        holder["app"] = app
        assert client.post("/api/jobs/ocr/start", json={"redo": True}).status_code == 200
        assert "cancelled" in app.state.jobs["ocr"].message

        check = open_library_db(tmp_path)
        assert get_ocr_text(check, a)["text"] == "new a"
        assert get_ocr_text(check, b)["text"] == "old b"
        check.close()

    def test_failed_load_still_counts_as_processed(
        self, tmp_path: Path, monkeypatch, sync_threads
    ) -> None:
        conn = open_library_db(tmp_path)
        a = _add_asset(conn, "a.jpg")
        conn.close()
        _write_thumb(tmp_path, a)

        def _boom(*args, **kwargs):
            raise OSError("disk gone")

        class _Engine:
            engine_id = "fake"

            def __init__(self, *, accelerator: str, min_score: float) -> None:
                self.active_accelerator = accelerator

        monkeypatch.setattr("takeout_rater.ocr.engine.OcrEngine", _Engine)
        monkeypatch.setattr("takeout_rater.ocr.engine.load_image_bgr", _boom)
        client, app = _make_client(tmp_path)
        assert client.post("/api/jobs/ocr/start", json={}).status_code == 200
        progress = app.state.jobs["ocr"]
        assert progress.error is None
        assert progress.processed == 1 == progress.total
        assert progress.diagnostics["unreadable"] == 1
