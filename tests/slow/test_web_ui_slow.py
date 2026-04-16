"""Slow end-to-end web-UI test.

Excluded from the default test run (--ignore=tests/slow) because it calls
run_index() which hangs in environments without a display or GPU for thumbnail
generation.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from takeout_rater.ui.app import create_app


def _make_jpeg_with_exif(path: Path, make: str = "TestCamera", model: str = "TestModel X1") -> None:
    """Create a minimal JPEG at *path* with Make and Model EXIF tags."""
    from PIL import ExifTags  # noqa: PLC0415
    from PIL import Image as _Image

    img = _Image.new("RGB", (4, 4), color=(100, 150, 200))
    exif = img.getexif()
    exif[ExifTags.Base.Make] = make
    exif[ExifTags.Base.Model] = model
    img.save(path, "JPEG", exif=exif.tobytes())


# ── End-to-end: indexer → detail page ────────────────────────────────────────


def test_full_pipeline_sidecar_and_exif_shown_in_detail_page(tmp_path: Path) -> None:
    """End-to-end: after indexing a real takeout tree the detail page must display
    both the sidecar JSON panel and the EXIF data panel.

    This test exercises the complete stack:
        real takeout layout on disk
        → run_index()  (scan + upsert)
        → create_app() (resolves photos_root from library_root)
        → GET /assets/{id}
        → sidecar JSON content visible
        → EXIF Make/Model visible
    """
    from takeout_rater.db.connection import open_library_db  # noqa: PLC0415
    from takeout_rater.db.queries import list_assets  # noqa: PLC0415
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    # Build a minimal takeout tree with a real JPEG (EXIF) and a sidecar file.
    album = tmp_path / "Takeout" / "Photos from 2026"
    album.mkdir(parents=True)

    img_path = album / "photo.jpg"
    _make_jpeg_with_exif(img_path, make="PipelineCamera", model="FullStack 5000")

    (album / "photo.jpg.supplemental-metadata.json").write_text(
        '{"title":"photo.jpg","description":"pipeline test","url":"",'
        '"creationTime":{"timestamp":"1771361057"},'
        '"photoTakenTime":{"timestamp":"1771354888"},'
        '"imageViews":"42"}',
        encoding="utf-8",
    )

    # Run the real indexer — this is the full run_index pipeline.
    conn = open_library_db(tmp_path)
    run_index(tmp_path, conn)

    # create_app must resolve the correct photos_root so file reads succeed.
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)

    rows = list_assets(conn, limit=10)
    assert len(rows) == 1, f"Expected 1 indexed asset, got {len(rows)}"
    asset_id = rows[0].id

    resp = client.get(f"/assets/{asset_id}")
    assert resp.status_code == 200

    # Sidecar JSON panel: content from the .supplemental-metadata.json file.
    assert "pipeline test" in resp.text
    assert "imageViews" in resp.text
    assert "No sidecar JSON available" not in resp.text

    # EXIF panel: Make and Model tags read directly from the image file.
    assert "PipelineCamera" in resp.text
    assert "FullStack 5000" in resp.text
    assert "No EXIF data available" not in resp.text
