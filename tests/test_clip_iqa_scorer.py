"""Tests for the CLIPIQAScorer adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from takeout_rater.scorers.adapters.clip_iqa import CLIPIQAScorer

# ---------------------------------------------------------------------------
# Spec tests — no dependencies needed
# ---------------------------------------------------------------------------


def test_spec_scorer_id() -> None:
    assert CLIPIQAScorer.spec().scorer_id == "clip_iqa"


def test_spec_has_clip_iqa_metric() -> None:
    spec = CLIPIQAScorer.spec()
    assert len(spec.metrics) == 1
    assert spec.metrics[0].key == "clip_iqa"


def test_spec_clip_iqa_range() -> None:
    m = CLIPIQAScorer.spec().metrics[0]
    assert m.min_value == 0.0
    assert m.max_value == 100.0


def test_spec_clip_iqa_higher_is_better() -> None:
    m = CLIPIQAScorer.spec().metrics[0]
    assert m.higher_is_better is True


def test_spec_has_variants() -> None:
    spec = CLIPIQAScorer.spec()
    variant_ids = {v.variant_id for v in spec.variants}
    assert "vitb32" in variant_ids
    assert "vitl14" in variant_ids


def test_spec_default_variant() -> None:
    assert CLIPIQAScorer.spec().default_variant_id == "vitb32"


def test_is_available_returns_bool() -> None:
    assert isinstance(CLIPIQAScorer.is_available(), bool)


# ---------------------------------------------------------------------------
# Helper: build a scorer whose CLIP model is fully mocked (no network)
# ---------------------------------------------------------------------------


def _make_mock_scorer(tmp_path: Path, fixed_high_quality_prob: float = 0.75) -> CLIPIQAScorer:
    """Return a CLIPIQAScorer with all heavy state mocked (no download)."""
    import torch  # noqa: PLC0415

    scorer = CLIPIQAScorer.create()

    # CLIP model mock: encode_image returns a unit-norm embedding.
    fake_clip = MagicMock()
    embed_dim = 512  # ViT-B/32 output dim

    def fake_encode_image(batch):  # type: ignore[no-untyped-def]
        emb = torch.zeros(batch.shape[0], embed_dim)
        emb[:, 0] = 1.0
        return emb

    fake_clip.encode_image.side_effect = fake_encode_image

    # Preprocess mock: returns a zero tensor of the expected spatial shape.
    def fake_preprocess(img):  # type: ignore[no-untyped-def]
        return torch.zeros(3, 224, 224)

    # Pre-computed text features: shape (2, embed_dim).
    # The dot product of our unit-image-embedding with these text features
    # determines the softmax probabilities.  We choose values so that
    # P(high quality) = fixed_high_quality_prob after softmax.
    import math  # noqa: PLC0415

    # logit_high - logit_low = log(p / (1-p))
    log_ratio = math.log(fixed_high_quality_prob / (1.0 - fixed_high_quality_prob))
    text_features = torch.zeros(2, embed_dim)
    # image embedding = [1, 0, …0] → dot products are text_features[:, 0]
    # set high-quality logit = log_ratio/2, low-quality = -log_ratio/2
    text_features[0, 0] = log_ratio / 2.0
    text_features[1, 0] = -log_ratio / 2.0

    device = torch.device("cpu")
    scorer._clip_model = fake_clip
    scorer._preprocess = fake_preprocess
    scorer._text_features = text_features.to(device)
    scorer._device = device

    return scorer


# ---------------------------------------------------------------------------
# score_batch — empty input (no model load)
# ---------------------------------------------------------------------------


def test_score_batch_empty_returns_empty(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    scorer = _make_mock_scorer(tmp_path)
    assert scorer.score_batch([]) == []


# ---------------------------------------------------------------------------
# score_batch — missing file
# ---------------------------------------------------------------------------


def test_score_batch_missing_file_returns_zero(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    scorer = _make_mock_scorer(tmp_path)
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["clip_iqa"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_batch — real image files (mocked model)
# ---------------------------------------------------------------------------


def test_score_batch_returns_clip_iqa_key(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(p, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch([p])
    assert len(results) == 1
    assert "clip_iqa" in results[0]


def test_score_batch_score_in_range(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(p, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch([p])
    assert 0.0 <= results[0]["clip_iqa"] <= 100.0


def test_score_batch_mock_returns_expected_value(tmp_path: Path) -> None:
    """Mock scorer with fixed_high_quality_prob=0.75 should produce ≈75.0."""
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(p, "JPEG")

    scorer = _make_mock_scorer(tmp_path, fixed_high_quality_prob=0.75)
    results = scorer.score_batch([p])
    assert results[0]["clip_iqa"] == pytest.approx(75.0, abs=1.0)


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 80, 100, 150)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)
    for r in results:
        assert "clip_iqa" in r
        assert 0.0 <= r["clip_iqa"] <= 100.0
