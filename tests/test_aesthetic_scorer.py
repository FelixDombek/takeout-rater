"""Tests for the AestheticScorer adapter (LAION Aesthetic Predictor v2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from takeout_rater.scorers.adapters.laion import (
    _EMBEDDING_DIM,
    _SCORE_BATCH_SIZE,
    AestheticScorer,
    _build_mlp,
)

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


def test_spec_requires_no_extras() -> None:
    spec = AestheticScorer.spec()
    assert spec.requires_extras == ()


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
# HuggingFace repo resolution
# ---------------------------------------------------------------------------


def test_hf_repo_candidates_env_first(monkeypatch) -> None:
    from takeout_rater.scorers.adapters import laion

    monkeypatch.setenv("TAKEOUT_RATER_AESTHETIC_REPO", "custom/repo")
    candidates = laion.AestheticScorer._hf_repo_candidates()
    assert candidates[0] == "custom/repo"
    # No duplicates should be present
    assert len(candidates) == len(set(candidates))


def test_download_mlp_weights_uses_fallback(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub  # noqa: PLC0415

    from takeout_rater.scorers.adapters import laion

    repos = ("broken/repo", "working/repo")
    monkeypatch.setattr(
        laion.AestheticScorer,
        "_hf_repo_candidates",
        staticmethod(lambda: repos),
    )

    calls: list[str] = []

    def fake_download(*, repo_id: str, filename: str) -> str:
        calls.append(repo_id)
        if repo_id == "working/repo":
            dest = tmp_path / filename
            dest.write_bytes(b"ok")
            return str(dest)
        raise RuntimeError("not found")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    weights_path = laion.AestheticScorer._download_mlp_weights()
    assert weights_path == tmp_path / laion._HF_FILENAME
    assert calls == list(repos)


# ---------------------------------------------------------------------------
# Model loading — quick_gelu
# ---------------------------------------------------------------------------


def test_ensure_loaded_passes_quick_gelu(monkeypatch, tmp_path: Path) -> None:
    """_ensure_loaded must pass quick_gelu=True via the shared clip_backbone."""
    import torch  # noqa: PLC0415

    import takeout_rater.scorers.adapters.clip_backbone as backbone  # noqa: PLC0415

    fake_model = MagicMock()
    fake_model.encode_image.return_value = torch.zeros(1, _EMBEDDING_DIM)

    create_calls: list[dict] = []

    def fake_create(model_name, pretrained=None, **kwargs):  # type: ignore[no-untyped-def]
        create_calls.append({"model_name": model_name, "pretrained": pretrained, **kwargs})
        fake_transform = MagicMock()
        return fake_model, None, fake_transform

    fake_mlp = MagicMock()
    fake_mlp.return_value = torch.full((1, 1), 7.5)
    fake_mlp.eval = MagicMock(return_value=fake_mlp)
    fake_mlp.to = MagicMock(return_value=fake_mlp)

    # Reset the backbone singleton so _ensure_loaded triggers a fresh load
    monkeypatch.setattr(backbone, "_clip_model", None)
    monkeypatch.setattr(backbone, "_preprocess", None)
    monkeypatch.setattr(backbone, "_tokenizer", None)
    monkeypatch.setattr(backbone, "_device", None)

    import open_clip  # noqa: PLC0415

    monkeypatch.setattr(open_clip, "create_model_and_transforms", fake_create)
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda _name: MagicMock())

    from takeout_rater.scorers.adapters import laion

    monkeypatch.setattr(laion, "_build_mlp", lambda _dim: fake_mlp)
    monkeypatch.setattr(
        laion.AestheticScorer,
        "_download_mlp_weights",
        lambda self: tmp_path / "weights.pth",
    )

    import torch as torch_mod  # noqa: PLC0415

    monkeypatch.setattr(torch_mod, "load", lambda path, map_location=None, weights_only=False: {})
    monkeypatch.setattr(fake_mlp, "load_state_dict", lambda state_dict: None)

    scorer = AestheticScorer.create()
    scorer._ensure_loaded()

    assert len(create_calls) == 1
    assert create_calls[0].get("quick_gelu") is True

    # Clean up singleton state so other tests are not affected
    monkeypatch.setattr(backbone, "_clip_model", None)
    monkeypatch.setattr(backbone, "_preprocess", None)
    monkeypatch.setattr(backbone, "_tokenizer", None)
    monkeypatch.setattr(backbone, "_device", None)


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
    from PIL import Image  # noqa: PLC0415

    # Create a small test image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path, "JPEG")

    scorer = AestheticScorer.create()

    # Build a tiny torch tensor mock that behaves like a real embedding
    import torch  # noqa: PLC0415

    # Fake CLIP model: returns a unit-vector embedding for each image in the batch.
    fake_clip = MagicMock()

    def fake_encode_image(batch):  # type: ignore[no-untyped-def]
        emb = torch.zeros(batch.shape[0], _EMBEDDING_DIM)
        emb[:, 0] = 1.0  # unit vectors
        return emb

    fake_clip.encode_image.side_effect = fake_encode_image

    # Fake preprocess: just returns a zero tensor of the right shape
    def fake_preprocess(img):  # type: ignore[no-untyped-def]
        return torch.zeros(3, 224, 224)

    # Fake MLP: returns a fixed aesthetic score of 7.5, shaped for the batch.
    fake_mlp = MagicMock()
    fake_mlp.side_effect = lambda x: torch.full((x.shape[0], 1), 7.5)

    device = torch.device("cpu")
    scorer._clip_model = fake_clip
    scorer._preprocess = fake_preprocess
    scorer._mlp = fake_mlp
    scorer._device = device

    return scorer


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    """score_batch must return exactly one result per input path."""
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)


def test_score_batch_returns_aesthetic_key(tmp_path: Path) -> None:
    """Each result dict must contain the 'aesthetic' key."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch([img_path])
    assert len(results) == 1
    assert "aesthetic" in results[0]


def test_score_batch_score_in_range(tmp_path: Path) -> None:
    """Score must be in [0, 10]."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch([img_path])
    assert 0.0 <= results[0]["aesthetic"] <= 10.0


def test_score_batch_mock_returns_expected_value(tmp_path: Path) -> None:
    """The mocked scorer should return the value injected by the fake MLP (7.5)."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch([img_path])
    assert results[0]["aesthetic"] == pytest.approx(7.5)


def test_score_batch_missing_file_returns_zero(tmp_path: Path) -> None:
    """A missing image file should yield aesthetic = 0.0, not raise."""
    scorer = _make_mock_scorer(tmp_path)
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["aesthetic"] == pytest.approx(0.0)


def test_score_batch_clamps_above_ten(tmp_path: Path) -> None:
    """Scores above 10 must be clamped to 10.0."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    # Override MLP to return a value above 10
    scorer._mlp.side_effect = lambda x: torch.full((x.shape[0], 1), 15.0)
    results = scorer.score_batch([img_path])
    assert results[0]["aesthetic"] == pytest.approx(10.0)


def test_score_batch_clamps_below_zero(tmp_path: Path) -> None:
    """Scores below 0 must be clamped to 0.0."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(tmp_path)
    # Override MLP to return a negative value
    scorer._mlp.side_effect = lambda x: torch.full((x.shape[0], 1), -3.0)
    results = scorer.score_batch([img_path])
    assert results[0]["aesthetic"] == pytest.approx(0.0)


def test_score_batch_uses_batched_inference(tmp_path: Path) -> None:
    """CLIP encode_image must be called once for N images that fit in one chunk."""
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(5):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 40, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer(tmp_path)
    scorer.score_batch(paths)
    assert scorer._clip_model.encode_image.call_count == 1


def test_score_batch_multiple_chunks(tmp_path: Path) -> None:
    """With _SCORE_BATCH_SIZE + 1 images, encode_image must be called exactly twice."""
    from PIL import Image  # noqa: PLC0415

    n = _SCORE_BATCH_SIZE + 1
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (64, 64), color=(i % 256, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch(paths)

    assert len(results) == n
    assert scorer._clip_model.encode_image.call_count == 2


# ---------------------------------------------------------------------------
# score_one convenience wrapper
# ---------------------------------------------------------------------------


def test_score_one(tmp_path: Path) -> None:
    """score_one must return a dict with 'aesthetic' in [0, 10]."""
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


def test_build_mlp_output_shape() -> None:
    """_build_mlp(768) must map a (1, 768) tensor to a (1, 1) output."""
    import torch  # noqa: PLC0415

    mlp = _build_mlp(_EMBEDDING_DIM)
    mlp.eval()
    with torch.no_grad():
        x = torch.randn(1, _EMBEDDING_DIM)
        out = mlp(x)
    assert out.shape == (1, 1)


def test_build_mlp_state_dict_keys_use_layers_prefix() -> None:
    """All MLP state_dict keys must start with 'layers.' to match the published checkpoint.

    The checkpoint (sac+logos+ava1-l14-linearMSE.pth) was saved from a model with
    ``self.layers = nn.Sequential(...)``, so every parameter key is prefixed with
    ``layers.`` (e.g. ``layers.0.weight``).  A bare ``nn.Sequential`` would instead
    produce keys like ``0.weight``, causing ``load_state_dict`` to fail.
    """
    mlp = _build_mlp(_EMBEDDING_DIM)
    keys = list(mlp.state_dict().keys())

    assert keys, "_build_mlp returned a model with no parameters"
    assert all(k.startswith("layers.") for k in keys), (
        f"Expected all state_dict keys to start with 'layers.', got: {keys}"
    )


def test_build_mlp_state_dict_round_trip(tmp_path: Path) -> None:
    """The MLP state_dict must be loadable back into a fresh instance without errors.

    This guards against architecture mismatches where saving and reloading the
    weights (as the real checkpoint loading does) would raise a RuntimeError.
    """
    import torch  # noqa: PLC0415

    mlp = _build_mlp(_EMBEDDING_DIM)
    weights_path = tmp_path / "weights.pth"
    torch.save(mlp.state_dict(), str(weights_path))

    mlp2 = _build_mlp(_EMBEDDING_DIM)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    mlp2.load_state_dict(state)  # Must not raise


# ---------------------------------------------------------------------------
# Prefetch / pipelining tests
# ---------------------------------------------------------------------------


def test_score_batch_encode_image_called_once_per_chunk(tmp_path: Path) -> None:
    """encode_image must be called exactly once per chunk (batching preserved after prefetch)."""
    from PIL import Image  # noqa: PLC0415

    n = _SCORE_BATCH_SIZE + 1  # forces two chunks
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (64, 64), color=(i % 256, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer(tmp_path)
    results = scorer.score_batch(paths)

    assert len(results) == n
    # Two chunks → encode_image must have been called exactly twice.
    assert scorer._clip_model.encode_image.call_count == 2
