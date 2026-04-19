"""Tests for the BRISQUEScorer heuristic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from takeout_rater.scoring.scorers.brisque import BRISQUEScorer

# ---------------------------------------------------------------------------
# Spec tests — no dependencies needed
# ---------------------------------------------------------------------------


def test_spec_scorer_id() -> None:
    assert BRISQUEScorer.spec().scorer_id == "brisque"


def test_spec_has_brisque_quality_metric() -> None:
    spec = BRISQUEScorer.spec()
    assert len(spec.all_metrics()) == 1
    assert spec.all_metrics()[0].key == "brisque_quality"


def test_spec_range() -> None:
    m = BRISQUEScorer.spec().all_metrics()[0]
    assert m.min_value == 0.0
    assert m.max_value == 100.0
    assert m.higher_is_better is True


def test_spec_has_default_variant() -> None:
    spec = BRISQUEScorer.spec()
    assert any(v.variant_id == "default" for v in spec.variants)
    assert spec.default_variant_id == "default"


def test_spec_display_name_not_empty() -> None:
    spec = BRISQUEScorer.spec()
    assert spec.display_name
    assert spec.description


def test_spec_requires_no_extras() -> None:
    assert BRISQUEScorer.spec().requires_extras == ()


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def test_is_available_returns_bool() -> None:
    result = BRISQUEScorer.is_available()
    assert isinstance(result, bool)


def test_is_available_false_when_piq_missing() -> None:
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "piq":
            raise ImportError("mocked missing piq")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        assert BRISQUEScorer.is_available() is False


# ---------------------------------------------------------------------------
# score_batch edge cases (no model required)
# ---------------------------------------------------------------------------


def test_score_batch_empty_returns_empty() -> None:
    if not BRISQUEScorer.is_available():
        pytest.skip("piq not available")
    scorer = BRISQUEScorer.create()
    assert scorer.score_batch([]) == []


def test_score_batch_missing_file_returns_zero(tmp_path: Path) -> None:
    if not BRISQUEScorer.is_available():
        pytest.skip("piq not available")
    scorer = BRISQUEScorer.create()
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["brisque_quality"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_batch with mocked piq (avoids heavy computation in unit tests)
# ---------------------------------------------------------------------------


def _make_mock_scorer() -> BRISQUEScorer:
    """Return a BRISQUEScorer with piq.brisque mocked to return a fixed raw score."""
    scorer = BRISQUEScorer.create()
    return scorer


def test_score_batch_inverts_raw_score(tmp_path: Path) -> None:
    """Raw BRISQUE=20 should yield quality=80."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()

    import torch  # noqa: PLC0415

    mock_tensor = MagicMock(return_value=torch.tensor(20.0))
    with patch("piq.brisque", mock_tensor):
        results = scorer.score_batch([img_path])

    assert len(results) == 1
    assert results[0]["brisque_quality"] == pytest.approx(80.0)


def test_score_batch_clamps_raw_above_100(tmp_path: Path) -> None:
    """Raw BRISQUE > 100 should be clamped, yielding quality=0."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()

    import torch  # noqa: PLC0415

    with patch("piq.brisque", return_value=torch.tensor(150.0)):
        results = scorer.score_batch([img_path])

    assert results[0]["brisque_quality"] == pytest.approx(0.0)


def test_score_batch_raw_zero_yields_quality_100(tmp_path: Path) -> None:
    """Raw BRISQUE=0 (perfect) should invert to quality=100."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()

    import torch  # noqa: PLC0415

    with patch("piq.brisque", return_value=torch.tensor(0.0)):
        results = scorer.score_batch([img_path])

    assert results[0]["brisque_quality"] == pytest.approx(100.0)


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    """score_batch must return one result per input path."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer()
    with patch("piq.brisque", return_value=torch.tensor(50.0)):
        results = scorer.score_batch(paths)

    assert len(results) == len(paths)
    for r in results:
        assert "brisque_quality" in r
        assert 0.0 <= r["brisque_quality"] <= 100.0


def test_score_batch_assertion_error_returns_zero(tmp_path: Path) -> None:
    """AssertionError from piq (e.g. AGGD assertion on uniform images) is caught gracefully."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "uniform.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()

    with patch(
        "piq.brisque",
        side_effect=AssertionError(
            "Expected input tensor (pairwise products of neighboring MSCN coefficients)"
            "  with values below zero to compute parameters of AGGD"
        ),
    ):
        results = scorer.score_batch([img_path])

    assert len(results) == 1
    assert results[0]["brisque_quality"] == pytest.approx(0.0)


def test_score_batch_assertion_error_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """AssertionError from piq is logged at WARNING with scorer id and asset path."""
    import logging  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "uniform.jpg"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()

    with (
        caplog.at_level(logging.WARNING, logger="takeout_rater.scoring.scorers.brisque"),
        patch("piq.brisque", side_effect=AssertionError("AGGD assertion")),
    ):
        scorer.score_batch([img_path])

    assert any("brisque" in r.message and str(img_path) in r.message for r in caplog.records)


def test_score_one(tmp_path: Path) -> None:
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = _make_mock_scorer()
    with patch("piq.brisque", return_value=torch.tensor(30.0)):
        result = scorer.score_one(p)
    assert "brisque_quality" in result
    assert 0.0 <= result["brisque_quality"] <= 100.0
