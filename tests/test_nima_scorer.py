"""Tests for the NIMAScorer adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from takeout_rater.scorers.adapters.nima import _NUM_CLASSES, _VARIANT_FILENAMES, NIMAScorer

# ---------------------------------------------------------------------------
# Spec tests — no dependencies needed
# ---------------------------------------------------------------------------


def test_spec_scorer_id() -> None:
    assert NIMAScorer.spec().scorer_id == "nima"


def test_spec_has_nima_score_metric() -> None:
    spec = NIMAScorer.spec()
    assert len(spec.metrics) == 1
    assert spec.metrics[0].key == "nima_score"


def test_spec_range() -> None:
    m = NIMAScorer.spec().metrics[0]
    assert m.min_value == 1.0
    assert m.max_value == 10.0
    assert m.higher_is_better is True


def test_spec_has_aesthetic_and_technical_variants() -> None:
    spec = NIMAScorer.spec()
    variant_ids = {v.variant_id for v in spec.variants}
    assert "aesthetic" in variant_ids
    assert "technical" in variant_ids


def test_spec_default_variant_is_aesthetic() -> None:
    assert NIMAScorer.spec().default_variant_id == "aesthetic"


def test_spec_requires_no_extras() -> None:
    assert NIMAScorer.spec().requires_extras == ()


def test_spec_display_name_not_empty() -> None:
    spec = NIMAScorer.spec()
    assert spec.display_name
    assert spec.description


# ---------------------------------------------------------------------------
# Variant filenames
# ---------------------------------------------------------------------------


def test_variant_filenames_covers_both_variants() -> None:
    assert "aesthetic" in _VARIANT_FILENAMES
    assert "technical" in _VARIANT_FILENAMES


def test_variant_filenames_are_pth_files() -> None:
    for filename in _VARIANT_FILENAMES.values():
        assert filename.endswith(".pth")


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def test_is_available_returns_bool() -> None:
    result = NIMAScorer.is_available()
    assert isinstance(result, bool)


def test_is_available_false_when_torchvision_missing() -> None:
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torchvision":
            raise ImportError("mocked missing torchvision")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        assert NIMAScorer.is_available() is False


# ---------------------------------------------------------------------------
# _download_weights
# ---------------------------------------------------------------------------


def test_download_weights_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unknown NIMA variant"):
        NIMAScorer._download_weights("unknown_variant")


def test_download_weights_calls_hf_hub(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub  # noqa: PLC0415

    from takeout_rater.scorers.adapters import nima

    def fake_download(*, repo_id: str, filename: str) -> str:
        dest = tmp_path / filename
        dest.write_bytes(b"fake_weights")
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    path = nima.NIMAScorer._download_weights("aesthetic")
    assert path.exists()


# ---------------------------------------------------------------------------
# score_batch edge cases (no model required)
# ---------------------------------------------------------------------------


def test_score_batch_empty_returns_empty() -> None:
    scorer = NIMAScorer.create()
    assert scorer.score_batch([]) == []


def test_score_batch_missing_file_returns_one(tmp_path: Path) -> None:
    """A missing file should yield nima_score=1.0 (minimum), not raise."""
    scorer = NIMAScorer.create()
    # Inject a no-op model so we don't need to download weights
    import torch  # noqa: PLC0415
    import torchvision.transforms as T  # noqa: PLC0415

    fake_model = MagicMock()
    # Return a uniform distribution over 10 classes → expected score = 5.5
    fake_model.return_value = torch.full((1, _NUM_CLASSES), 1.0 / _NUM_CLASSES)
    scorer._model = fake_model
    scorer._preprocess = T.ToTensor()
    scorer._device = torch.device("cpu")

    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["nima_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_batch with mocked model
# ---------------------------------------------------------------------------


def _make_mock_scorer(fixed_score: float = 7.0, variant_id: str = "aesthetic") -> NIMAScorer:
    """Return a NIMAScorer whose model outputs a fixed expected score."""
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    import torchvision.transforms as T  # noqa: PLC0415

    scorer = NIMAScorer.create(variant_id=variant_id)

    # Build a distribution that concentrates all probability on the rating bin
    # nearest to fixed_score.  Since ratings are integers 1–10, we round.
    rating_idx = max(0, min(_NUM_CLASSES - 1, round(fixed_score) - 1))
    probs = torch.zeros(1, _NUM_CLASSES)
    probs[0, rating_idx] = 1.0  # one-hot → expected score = rating_idx + 1

    fake_model = MagicMock()
    fake_model.return_value = probs

    scorer._model = fake_model
    scorer._preprocess = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    scorer._device = torch.device("cpu")
    return scorer


def test_score_batch_returns_nima_score_key(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()
    results = scorer.score_batch([img_path])
    assert len(results) == 1
    assert "nima_score" in results[0]


def test_score_batch_value_in_range(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(fixed_score=7.0)
    results = scorer.score_batch([img_path])
    assert 1.0 <= results[0]["nima_score"] <= 10.0


def test_score_batch_expected_score(tmp_path: Path) -> None:
    """One-hot distribution on rating 7 should yield nima_score ≈ 7.0."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(fixed_score=7.0)
    results = scorer.score_batch([img_path])
    assert results[0]["nima_score"] == pytest.approx(7.0)


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer()
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)


def test_score_batch_technical_variant(tmp_path: Path) -> None:
    """Technical variant uses a separate model load; spec is the same."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(fixed_score=5.0, variant_id="technical")
    assert scorer.variant_id == "technical"
    results = scorer.score_batch([img_path])
    assert "nima_score" in results[0]


def test_score_one(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = _make_mock_scorer(fixed_score=6.0)
    result = scorer.score_one(p)
    assert "nima_score" in result
    assert 1.0 <= result["nima_score"] <= 10.0
