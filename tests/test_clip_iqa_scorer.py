"""Tests for the CLIPIQAScorer adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from takeout_rater.scorers.adapters.clip_iqa import CLIPIQAScorer

# ---------------------------------------------------------------------------
# Spec tests — no dependencies needed
# ---------------------------------------------------------------------------


def test_spec_scorer_id() -> None:
    assert CLIPIQAScorer.spec().scorer_id == "clip_iqa"


def test_spec_has_clip_quality_metric() -> None:
    spec = CLIPIQAScorer.spec()
    assert len(spec.metrics) == 1
    assert spec.metrics[0].key == "clip_quality"


def test_spec_range() -> None:
    m = CLIPIQAScorer.spec().metrics[0]
    assert m.min_value == 0.0
    assert m.max_value == 1.0
    assert m.higher_is_better is True


def test_spec_has_variant() -> None:
    spec = CLIPIQAScorer.spec()
    assert any(v.variant_id == "vit_l14_openai" for v in spec.variants)
    assert spec.default_variant_id == "vit_l14_openai"


def test_spec_requires_no_extras() -> None:
    assert CLIPIQAScorer.spec().requires_extras == ()


def test_spec_display_name_not_empty() -> None:
    spec = CLIPIQAScorer.spec()
    assert spec.display_name
    assert spec.description


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def test_is_available_returns_bool() -> None:
    result = CLIPIQAScorer.is_available()
    assert isinstance(result, bool)


def test_is_available_false_when_open_clip_missing() -> None:
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "open_clip":
            raise ImportError("mocked missing open_clip")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        assert CLIPIQAScorer.is_available() is False


# ---------------------------------------------------------------------------
# score_batch edge cases (no model required)
# ---------------------------------------------------------------------------


def test_score_batch_empty_returns_empty() -> None:
    scorer = CLIPIQAScorer.create()
    assert scorer.score_batch([]) == []


# ---------------------------------------------------------------------------
# score_batch with mocked CLIP model
# ---------------------------------------------------------------------------


def _make_mock_scorer(good_prob: float = 0.75) -> CLIPIQAScorer:
    """Return a CLIPIQAScorer with a fully mocked CLIP backbone.

    Args:
        good_prob: The softmax probability to assign to the "good" prompt.
    """
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    scorer = CLIPIQAScorer.create()

    # The text features are shape (2, D); image features are (1, D).
    # We set them up so that softmax(sims)[0] == good_prob.
    D = 16

    # We'll mock encode_image to return something, and softmax to return our value.
    # Simpler: wire up the internal state directly.
    fake_clip = MagicMock()

    # Make encode_image return a unit vector
    def fake_encode_image(tensor):  # type: ignore[no-untyped-def]
        emb = torch.zeros(tensor.shape[0], D)
        emb[:, 0] = 1.0
        return emb

    fake_clip.encode_image.side_effect = fake_encode_image

    # text_features: good=(1,0,...), bad=(0,1,...) → sims will differ by feature direction
    # Instead, bypass the whole sim computation by mocking _text_features so that
    # the resulting softmax gives us the desired good_prob.
    # sims[i] = img_feat @ text_feat[i].T
    # We set text_features such that:
    #   sims[0] = logit that produces good_prob after softmax with sims[1]
    # With sims = [log(good_prob) - log(bad_prob), 0]:
    bad_prob = 1.0 - good_prob
    logit_good = (torch.log(torch.tensor(good_prob)) - torch.log(torch.tensor(bad_prob))).item()
    # img_feat will be unit vector along dim 0.
    # Set text_feat[0] = logit_good * e_0, text_feat[1] = 0
    text_features = torch.zeros(2, D)
    text_features[0, 0] = logit_good  # dot with img_feat[0]=1 → logit_good
    text_features[1, 0] = 0.0

    scorer._clip_model = fake_clip
    scorer._preprocess = lambda img: torch.zeros(3, 224, 224)
    scorer._text_features = text_features
    scorer._device = torch.device("cpu")

    return scorer


def test_score_batch_returns_clip_quality_key(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()
    results = scorer.score_batch([img_path])
    assert len(results) == 1
    assert "clip_quality" in results[0]


def test_score_batch_value_in_range(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(good_prob=0.75)
    results = scorer.score_batch([img_path])
    assert 0.0 <= results[0]["clip_quality"] <= 1.0


def test_score_batch_missing_file_returns_zero(tmp_path: Path) -> None:
    scorer = _make_mock_scorer()
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["clip_quality"] == pytest.approx(0.0)


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 80, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer()
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)


def test_score_one(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = _make_mock_scorer(good_prob=0.8)
    result = scorer.score_one(p)
    assert "clip_quality" in result
    assert 0.0 <= result["clip_quality"] <= 1.0
