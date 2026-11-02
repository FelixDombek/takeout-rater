"""End-to-end OCR with the real RapidOCR models bundled in the rapidocr wheel."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

pytest.importorskip("rapidocr")


def test_rapidocr_reads_rendered_text(tmp_path: Path) -> None:
    from takeout_rater.ocr.engine import OcrEngine, load_image_bgr

    img = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
    except OSError:
        font = ImageFont.load_default(40)
    draw.text((30, 90), "Welcome to Berlin", fill="black", font=font)
    path = tmp_path / "sign.png"
    img.save(path)

    engine = OcrEngine(accelerator="cpu")
    result = engine.recognize(load_image_bgr(path))
    assert "berlin" in result.text.lower()
