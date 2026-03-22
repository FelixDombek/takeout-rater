"""Tests for the AestheticScorer adapter (LAION Aesthetic Predictor v2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from takeout_rater.scorers.adapters.laion import AestheticScorer, _build_mlp, _EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Spec tests — no dependencies needed
# ---------------------------------------------------------------------------


def test_spec_scorer_id() -> None:
    assert AestheticScorer.spec().scorer_id == "aesthetic"


def test_spec_has_aesthetic_metric() -> None:
    spec = AestheticScorer.spec()
    assert len(spec.metrics) == 1
    assert spec.metrics[0].key == "aesthetic"


def test_spec_aesthetic_range() -> None:
    m = AestheticScorer.spec().metrics[0]
    assert m.min_value == 0.0
    assert m.max_value == 10.0
    assert m.higher_is_better is True


def test_spec_has_laion_v2_variant() -> None:
    spec = AestheticScorer.spec()
    assert any(v.variant_id == "laion_v2" for v in spec.variants)
    assert spec.default_variant_id == "laion_v2"


def test_spec_requires_aesthetic_extra() -> None:
    spec = AestheticScorer.spec()
    assert "aesthetic" in spec.requires_extras


def test_spec_display_name_not_empty() -> None:
    spec = AestheticScorer.spec()
    assert spec.display_name
    assert spec.description


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def test_is_available_returns_bool() -> None:
    result = AestheticScorer.is_available()
    assert isinstance(result, bool)


def test_is_available_false_when_open_clip_missing() -> None:
    """is_available() must return False when open_clip cannot be imported."""
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "open_clip":
            raise ImportError("mocked missing open_clip")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        assert AestheticScorer.is_available() is False


def test_is_available_false_when_torch_missing() -> None:
    """is_available() must return False when torch cannot be imported."""
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise ImportError("mocked missing torch")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        assert AestheticScorer.is_available() is False


# ---------------------------------------------------------------------------
# score_batch — empty input (no dependencies needed)
# ---------------------------------------------------------------------------


def test_score_batch_empty_returns_empty() -> None:
    """score_batch([]) must return [] without loading the model."""
    scorer = AestheticScorer.create()
    result = scorer.score_batch([])
    assert result == []


# ---------------------------------------------------------------------------
# score_batch — mocked model (avoids downloading anything)
# ---------------------------------------------------------------------------


def _make_mock_scorer(tmp_path: Path) -> AestheticScorer:
    """Return an AestheticScorer whose model is fully mocked (no network, no torch)."""
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    # Create a small test image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path, "JPEG")

    scorer = AestheticScorer.create()

    # Build a tiny torch tensor mock that behaves like a real embedding
    import torch  # noqa: PLC0415  (skip test if torch not available)

    # Fake CLIP model: returns a fixed unit embedding
    fake_clip = MagicMock()
    fake_embedding = torch.zeros(1, _EMBEDDING_DIM)
    fake_embedding[0, 0] = 1.0  # unit vector
    fake_clip.encode_image.return_value = fake_embedding

    # Fake preprocess: just returns a zero tensor of the right shape
    def fake_preprocess(img):  # type: ignore[no-untyped-def]
        return torch.zeros(3, 224, 224)

    # Fake MLP: returns a fixed aesthetic score of 7.5
    fake_mlp = MagicMock()
    fake_mlp.return_value = torch.tensor([[7.5]])

    import torch  # noqa: PLC0415, F811

    device = torch.device("cpu")
    scorer._clip_model = fake_clip
    scorer._preprocess = fake_preprocess
    scorer._mlp = fake_mlp
    scorer._device = device

    return scorer


@pytest.mark.skipif(
    not AestheticScorer.is_available(),
    reason="aesthetic scorer dependencies (torch + open_clip) not installed",
)
def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    """score_batch must return exactly one result per input path."""
    pytest.importorskip("torch")
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)


@pytest.mark.skipif(
    not AestheticScorer.is_available(),
    reason="aesthetic scorer dependencies (torch + open_clip) not installed",
)
def test_score_batch_returns_aesthetic_key(tmp_path: Path) -> None:
    """Each result dict must contain the 'aesthetic' key."""
    pytest.importorskip("torch")
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch([img_path])
    assert len(results) == 1
    assert "aesthetic" in results[0]


@pytest.mark.skipif(
    not AestheticScorer.is_available(),
    reason="aesthetic scorer dependencies (torch + open_clip) not installed",
)
def test_score_batch_score_in_range(tmp_path: Path) -> None:
    """Score must be in [0, 10]."""
    pytest.importorskip("torch")
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch([img_path])
    assert 0.0 <= results[0]["aesthetic"] <= 10.0


@pytest.mark.skipif(
    not AestheticScorer.is_available(),
    reason="aesthetic scorer dependencies (torch + open_clip) not installed",
)
def test_score_batch_mock_returns_expected_value(tmp_path: Path) -> None:
    """The mocked scorer should return the value injected by the fake MLP (7.5)."""
    pytest.importorskip("torch")
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch([img_path])
    assert results[0]["aesthetic"] == pytest.approx(7.5)


@pytest.mark.skipif(
    not AestheticScorer.is_available(),
    reason="aesthetic scorer dependencies (torch + open_clip) not installed",
)
def test_score_batch_missing_file_returns_zero(tmp_path: Path) -> None:
    """A missing image file should yield aesthetic = 0.0, not raise."""
    pytest.importorskip("torch")

    scorer = _make_mock_scorer(tmp_path)
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["aesthetic"] == pytest.approx(0.0)


@pytest.mark.skipif(
    not AestheticScorer.is_available(),
    reason="aesthetic scorer dependencies (torch + open_clip) not installed",
)
def test_score_batch_clamps_above_ten(tmp_path: Path) -> None:
    """Scores above 10 must be clamped to 10.0."""
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    # Override MLP to return a value above 10
    scorer._mlp.return_value = torch.tensor([[15.0]])
    results = scorer.score_batch([img_path])
    assert results[0]["aesthetic"] == pytest.approx(10.0)


@pytest.mark.skipif(
    not AestheticScorer.is_available(),
    reason="aesthetic scorer dependencies (torch + open_clip) not installed",
)
def test_score_batch_clamps_below_zero(tmp_path: Path) -> None:
    """Scores below 0 must be clamped to 0.0."""
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    # Override MLP to return a negative value
    scorer._mlp.return_value = torch.tensor([[-3.0]])
    results = scorer.score_batch([img_path])
    assert results[0]["aesthetic"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_one convenience wrapper
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not AestheticScorer.is_available(),
    reason="aesthetic scorer dependencies (torch + open_clip) not installed",
)
def test_score_one(tmp_path: Path) -> None:
    """score_one must return a dict with 'aesthetic' in [0, 10]."""
    pytest.importorskip("torch")
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    result = scorer.score_one(img_path)
    assert "aesthetic" in result
    assert 0.0 <= result["aesthetic"] <= 10.0


# ---------------------------------------------------------------------------
# _build_mlp architecture check (no torch required for structure test)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not AestheticScorer.is_available(),
    reason="aesthetic scorer dependencies (torch + open_clip) not installed",
)
def test_build_mlp_output_shape() -> None:
    """_build_mlp(768) must map a (1, 768) tensor to a (1, 1) output."""
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    mlp = _build_mlp(_EMBEDDING_DIM)
    mlp.eval()
    with torch.no_grad():
        x = torch.randn(1, _EMBEDDING_DIM)
        out = mlp(x)
    assert out.shape == (1, 1)
