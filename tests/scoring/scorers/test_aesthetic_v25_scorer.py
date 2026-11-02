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


class _FakeVisionOutput:
    def __init__(self, pooler_output: Any) -> None:
        self.pooler_output = pooler_output


class _FakeModel:
    """Fake predictor: score = mean normalised pixel value + *offset*.

    The pooled embedding is ``[mean, 1]``; after the scorer's L2-normalisation
    the head recovers ``mean`` as the ratio of the two components.
    """

    def __init__(self, max_batch: int | None = None, offset: float = 5.0) -> None:
        self.max_batch = max_batch
        self.offset = offset
        self.batch_sizes: list[int] = []
        self.head_dtypes: list[Any] = []

    def vision_model(self, pixel_values: Any) -> _FakeVisionOutput:
        n = pixel_values.shape[0]
        if self.max_batch is not None and n > self.max_batch:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        self.batch_sizes.append(n)
        assert pixel_values.shape[1:] == (3, aesthetic_v25.INPUT_SIZE, aesthetic_v25.INPUT_SIZE)
        mean = pixel_values.mean(dim=(1, 2, 3))
        return _FakeVisionOutput(torch.stack([mean, torch.ones_like(mean)], dim=1))

    def layers(self, embeds: Any) -> Any:
        self.head_dtypes.append(embeds.dtype)
        return (embeds[:, 0] / embeds[:, 1] + self.offset).unsqueeze(1)


def _install_fake(monkeypatch: pytest.MonkeyPatch, model: _FakeModel) -> None:
    monkeypatch.setattr(
        aesthetic_v25, "_load_model", lambda: (model, torch.device("cpu"), torch.float32)
    )
    monkeypatch.setattr(
        aesthetic_v25, "pooled_embeddings", lambda m, px: m.vision_model(px).pooler_output
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
    assert results[0]["aesthetic"] == pytest.approx(4.0)  # black → -1 normalised
    # Undecodable images get no score at all, so they stay unscored and are
    # retried later instead of being stored below the 1–10 range.
    assert results[1] == {}
    assert results[2]["aesthetic"] == pytest.approx(6.0)  # white → +1 normalised
    assert results[3] == {}
    assert results[4]["aesthetic"] == pytest.approx(4.0)
    assert all(d == torch.float32 for d in model.head_dtypes)
    # Chunks of 2 paths; failed decodes are not sent to the model.
    assert model.batch_sizes == [1, 1, 1]


def test_all_failed_chunk_skips_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel()
    _install_fake(monkeypatch, model)
    scorer = AestheticV25Scorer()
    results = scorer.score_batch([tmp_path / "a.jpg", tmp_path / "b.jpg"])
    scorer.close()
    assert results == [{}, {}]
    assert model.batch_sizes == []


@pytest.mark.parametrize(("offset", "expected"), [(20.0, 10.0), (-20.0, 1.0)])
def test_scores_are_clamped_to_metric_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offset: float, expected: float
) -> None:
    _install_fake(monkeypatch, _FakeModel(offset=offset))
    scorer = AestheticV25Scorer()
    results = scorer.score_batch([_solid(tmp_path, "a.jpg", 128)])
    scorer.close()
    assert results == [{"aesthetic": expected}]


def test_decode_errors_outside_oserror_are_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E.g. PIL's DecompressionBombError must not fail the whole batch."""
    from PIL import Image

    _install_fake(monkeypatch, _FakeModel())
    good = _solid(tmp_path, "good.jpg", 0)
    bomb = _solid(tmp_path, "bomb.jpg", 0)
    real_preprocess = aesthetic_v25.preprocess_image

    def _preprocess(path: Path) -> Any:
        if path == bomb:
            raise Image.DecompressionBombError("too big")
        return real_preprocess(path)

    monkeypatch.setattr(aesthetic_v25, "preprocess_image", _preprocess)
    scorer = AestheticV25Scorer()
    results = scorer.score_batch([good, bomb])
    scorer.close()
    assert results[0]["aesthetic"] == pytest.approx(4.0)
    assert results[1] == {}


def test_close_releases_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, _FakeModel())
    scorer = AestheticV25Scorer()
    scorer.score_batch([_solid(tmp_path, "a.jpg", 0)])
    assert scorer._model is not None
    scorer.close()
    assert scorer._model is None
    assert scorer._pool is None


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

    # The head was trained with the PIL-based processor (transformers 4.x).
    # transformers 5 defaults to a torchvision backend whose bicubic resize
    # differs slightly, so compare against the PIL backend explicitly.
    processor_cls = getattr(transformers, "SiglipImageProcessorPil", None)
    if processor_cls is None:
        processor_cls = transformers.SiglipImageProcessor
    # Same settings as google/siglip-so400m-patch14-384's preprocessor_config.json.
    processor = processor_cls(
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


@pytest.mark.parametrize("tower_dtype", ["float32", "bfloat16"])
def test_real_model_class_with_tiny_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tower_dtype: str
) -> None:
    """Our forward matches the upstream model's own forward (no weights downloaded).

    With a bf16 tower the head still runs in fp32, as _load_model sets it up.
    """
    apv = pytest.importorskip("aesthetic_predictor_v2_5")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(0)
    config = transformers.SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=aesthetic_v25.INPUT_SIZE,
        patch_size=48,
    )
    reference = apv.AestheticPredictorV2_5Model(config).eval()
    # Move the random head's output into the 1–10 range so clamping is a no-op.
    with torch.no_grad():
        reference.layers.scoring_head[-1].bias.fill_(5.0)

    dtype = getattr(torch, tower_dtype)
    model = apv.AestheticPredictorV2_5Model(config).eval()
    model.load_state_dict(reference.state_dict())
    model.to(dtype=dtype)
    model.layers.to(dtype=torch.float32)
    monkeypatch.setattr(aesthetic_v25, "_load_model", lambda: (model, torch.device("cpu"), dtype))
    paths = [_solid(tmp_path, f"{i}.jpg", 40 * i) for i in range(3)]

    scorer = AestheticV25Scorer()
    results = scorer.score_batch(paths)
    scorer.close()

    with torch.inference_mode():
        batch = torch.from_numpy(np.stack([aesthetic_v25.preprocess_image(p) for p in paths]))
        pixels = aesthetic_v25.normalize_pixels(batch, torch.float32)
        expected = reference(pixel_values=pixels).logits.reshape(-1).tolist()
    tol = 1e-5 if tower_dtype == "float32" else 5e-2
    assert all(1.0 <= e <= 10.0 for e in expected)
    assert [r["aesthetic"] for r in results] == pytest.approx(expected, abs=tol)


def test_load_model_passes_vision_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full SigLIP checkpoint must be loaded with its *vision* config.

    transformers 5 fails with "'SiglipConfig' object has no attribute
    'hidden_size'" when the predictor is built from the full SiglipConfig.
    """
    import sys
    import types

    vision_config = object()
    calls: dict[str, Any] = {}

    class _FakeVisionConfig:
        @staticmethod
        def from_pretrained(name: str) -> object:
            calls["config_name"] = name
            return vision_config

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Linear(2, 1)

        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> _FakeModel:
            calls["model_name"] = name
            calls["kwargs"] = kwargs
            return cls()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.SiglipVisionConfig = _FakeVisionConfig  # type: ignore[attr-defined]
    fake_apv = types.ModuleType("aesthetic_predictor_v2_5")
    fake_apv.AestheticPredictorV2_5Model = _FakeModel  # type: ignore[attr-defined]
    fake_apv_sub = types.ModuleType("aesthetic_predictor_v2_5.siglip_v2_5")
    fake_apv_sub.URL = "https://example.invalid/head.pth"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "aesthetic_predictor_v2_5", fake_apv)
    monkeypatch.setitem(sys.modules, "aesthetic_predictor_v2_5.siglip_v2_5", fake_apv_sub)
    monkeypatch.setattr(
        torch.hub,
        "load_state_dict_from_url",
        lambda *_a, **_k: torch.nn.Linear(2, 1).state_dict(),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model, device, dtype = aesthetic_v25._load_model()

    assert calls["config_name"] == aesthetic_v25._ENCODER_MODEL
    assert calls["model_name"] == aesthetic_v25._ENCODER_MODEL
    assert calls["kwargs"]["config"] is vision_config
    assert device == torch.device("cpu") and dtype == torch.float32
    assert isinstance(model, _FakeModel)
