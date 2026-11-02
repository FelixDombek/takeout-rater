"""Tests for the Aesthetic Predictor V2.5 scorer.

The real model (~1.7 GB) is never loaded here: a fake model is injected so the
batching, preprocessing, and failure handling can be tested offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from takeout_rater.scoring.scorers import aesthetic_v25
from takeout_rater.scoring.scorers.aesthetic_v25 import AestheticV25Scorer
from takeout_rater.scoring.scorers.base import ScorerSpec
from takeout_rater.scoring.scorers.registry import list_scorers

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

# ── spec ─────────────────────────────────────────────────────────────────────


def test_spec_basics() -> None:
    spec = AestheticV25Scorer.spec()
    assert isinstance(spec, ScorerSpec)
    assert spec.scorer_id == "aesthetic_v25"
    assert spec.default_variant_id == "siglip_so400m"
    assert [v.variant_id for v in spec.variants] == ["siglip_so400m"]
    assert spec.requires_extras == ()


def test_metric_is_aesthetic_higher_is_better() -> None:
    metrics = AestheticV25Scorer.spec().metrics_for_variant("siglip_so400m")
    assert [m.key for m in metrics] == ["aesthetic"]
    assert metrics[0].higher_is_better is True
    assert (metrics[0].min_value, metrics[0].max_value) == (1.0, 10.0)


def test_registered() -> None:
    assert AestheticV25Scorer in list_scorers()


def test_is_available_returns_bool() -> None:
    assert isinstance(AestheticV25Scorer.is_available(), bool)


def test_score_batch_empty_does_not_load_model() -> None:
    scorer = AestheticV25Scorer()
    assert scorer.score_batch([]) == []
    assert scorer._model is None


# ── fake model helpers ───────────────────────────────────────────────────────


class _FakeOutput:
    def __init__(self, logits: Any) -> None:
        self.logits = logits


class _FakeModel:
    """Returns the mean normalised pixel value of each image + 5 as its score."""

    def __init__(self, max_batch: int | None = None) -> None:
        self.max_batch = max_batch
        self.batch_sizes: list[int] = []

    def __call__(self, pixel_values: Any, return_dict: bool = True) -> _FakeOutput:
        n = pixel_values.shape[0]
        if self.max_batch is not None and n > self.max_batch:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        self.batch_sizes.append(n)
        assert pixel_values.shape[1:] == (3, aesthetic_v25.INPUT_SIZE, aesthetic_v25.INPUT_SIZE)
        return _FakeOutput(pixel_values.mean(dim=(1, 2, 3)).unsqueeze(1) + 5.0)


def _install_fake(monkeypatch: pytest.MonkeyPatch, model: _FakeModel) -> None:
    monkeypatch.setattr(
        aesthetic_v25, "_load_model", lambda: (model, torch.device("cpu"), torch.float32)
    )


def _solid(tmp_path: Path, name: str, value: int) -> Path:
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (512, 384), color=(value, value, value)).save(path)
    return path


# ── scoring ──────────────────────────────────────────────────────────────────


def test_scores_in_order_with_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel()
    _install_fake(monkeypatch, model)
    monkeypatch.setenv("TAKEOUT_RATER_AESTHETIC_V25_BATCH_SIZE", "2")

    black = _solid(tmp_path, "black.jpg", 0)
    white = _solid(tmp_path, "white.png", 255)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    missing = tmp_path / "missing.jpg"

    scorer = AestheticV25Scorer()
    try:
        results = scorer.score_batch([black, broken, white, missing, black])
    finally:
        scorer.close()

    assert len(results) == 5
    assert all(set(r) == {"aesthetic"} for r in results)
    assert results[0]["aesthetic"] == pytest.approx(4.0)  # black → -1 normalised
    assert results[1]["aesthetic"] == 0.0
    assert results[2]["aesthetic"] == pytest.approx(6.0)  # white → +1 normalised
    assert results[3]["aesthetic"] == 0.0
    assert results[4]["aesthetic"] == pytest.approx(4.0)
    # Chunks of 2 paths; failed decodes are not sent to the model.
    assert model.batch_sizes == [1, 1, 1]


def test_all_failed_chunk_skips_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel()
    _install_fake(monkeypatch, model)
    scorer = AestheticV25Scorer()
    results = scorer.score_batch([tmp_path / "a.jpg", tmp_path / "b.jpg"])
    scorer.close()
    assert results == [{"aesthetic": 0.0}, {"aesthetic": 0.0}]
    assert model.batch_sizes == []


def test_oom_halves_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(max_batch=2)
    _install_fake(monkeypatch, model)
    monkeypatch.setenv("TAKEOUT_RATER_AESTHETIC_V25_BATCH_SIZE", "8")
    paths = [_solid(tmp_path, f"{i}.jpg", 0) for i in range(7)]

    scorer = AestheticV25Scorer()
    results = scorer.score_batch(paths)
    scorer.close()

    assert [r["aesthetic"] for r in results] == pytest.approx([4.0] * 7)
    assert sum(model.batch_sizes) == 7
    assert max(model.batch_sizes) <= 2


def test_invalid_batch_size_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAKEOUT_RATER_AESTHETIC_V25_BATCH_SIZE", "nope")
    assert aesthetic_v25._infer_batch_size() == aesthetic_v25._DEFAULT_INFER_BATCH_SIZE


# ── preprocessing parity with the upstream SigLIP processor ──────────────────


def test_preprocessing_matches_siglip_image_processor(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    from PIL import Image

    rng = np.random.default_rng(0)
    path = tmp_path / "noise.png"
    Image.fromarray(rng.integers(0, 256, (300, 512, 3), dtype=np.uint8)).save(path)

    # Same settings as google/siglip-so400m-patch14-384's preprocessor_config.json.
    processor = transformers.SiglipImageProcessor(
        do_resize=True,
        size={"height": 384, "width": 384},
        resample=3,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    expected = processor(images=Image.open(path), return_tensors="pt").pixel_values

    arr = aesthetic_v25.preprocess_image(path)
    assert arr.shape == (384, 384, 3)
    assert arr.dtype == np.uint8
    ours = aesthetic_v25.normalize_pixels(torch.from_numpy(arr[None]), torch.float32)

    assert ours.shape == expected.shape
    assert torch.allclose(ours, expected, atol=1e-5)


# ── integration with the real (tiny, randomly initialised) predictor model ───


def test_real_model_class_with_tiny_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Our forward call matches the upstream model's API (no weights downloaded)."""
    apv = pytest.importorskip("aesthetic_predictor_v2_5")
    transformers = pytest.importorskip("transformers")

    config = transformers.SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=aesthetic_v25.INPUT_SIZE,
        patch_size=48,
    )
    model = apv.AestheticPredictorV2_5Model(config).eval()
    monkeypatch.setattr(
        aesthetic_v25, "_load_model", lambda: (model, torch.device("cpu"), torch.float32)
    )
    paths = [_solid(tmp_path, f"{i}.jpg", 40 * i) for i in range(3)]

    scorer = AestheticV25Scorer()
    results = scorer.score_batch(paths)
    scorer.close()

    with torch.inference_mode():
        batch = torch.from_numpy(np.stack([aesthetic_v25.preprocess_image(p) for p in paths]))
        expected = model(aesthetic_v25.normalize_pixels(batch, torch.float32)).logits.reshape(-1)
    assert [r["aesthetic"] for r in results] == pytest.approx(expected.tolist(), abs=1e-5)
