"""Semantic outlier detection from stored image embeddings (k nearest neighbours).

For every photo the cosine similarity to its *k* nearest neighbours in
embedding space is averaged.  Photos whose neighbours are all far away (a low
mean similarity) do not look like anything else in the library: one-off
subjects, accidental pocket shots, screenshots among holiday photos, scans,
etc.  They are worth a look, either as hidden gems or as candidates for
deletion.

The search is exact brute force: the embedding matrix is multiplied with
itself in row blocks, so memory stays bounded (one ``block x N`` similarity
slab at a time).  Torch is used when importable (CUDA when available, which
makes 200k photos take seconds); otherwise a numpy fallback runs on the CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

#: Default number of neighbours averaged per photo.  Larger than a typical
#: burst, so a handful of near-identical shots of an odd subject still count
#: as unusual, while small enough to stay local.
DEFAULT_K = 10

#: Upper bound on the elements of one ``block x N`` similarity slab
#: (100M floats = 400 MB in float32, 200 MB in float16).
_MAX_SLAB_ELEMENTS = 100_000_000


def _block_rows(n: int, requested: int | None) -> int:
    if requested is not None:
        return max(1, requested)
    return max(64, min(4096, _MAX_SLAB_ELEMENTS // max(n, 1)))


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def _torch_device() -> tuple[Any, Any] | None:
    """Return ``(torch, device)`` or ``None`` when torch is not importable."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    try:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    except Exception:  # noqa: BLE001
        device = torch.device("cpu")
    return torch, device


def knn_mean_similarity(
    matrix: np.ndarray,
    *,
    k: int = DEFAULT_K,
    block_rows: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    use_torch: bool = True,
) -> np.ndarray | None:
    """Return each row's mean cosine similarity to its *k* nearest other rows.

    Args:
        matrix: ``(N, dim)`` embeddings (need not be normalised).
        k: Neighbours to average.  Capped at ``N - 1``.
        block_rows: Query rows per block; default sized to bound memory.
        on_progress: Called with ``(rows_done, N)`` after each block.
        cancel_check: Returning ``True`` aborts; the function then returns ``None``.
        use_torch: Use torch when importable (CUDA if available).

    Returns:
        ``(N,)`` float32 array, or ``None`` when cancelled.  With fewer than
        two rows there are no neighbours and every value is ``nan``.
    """
    n = int(matrix.shape[0])
    if n < 2:
        return np.full(n, np.nan, dtype=np.float32)
    k_eff = max(1, min(int(k), n - 1))
    x = _normalise(np.asarray(matrix, dtype=np.float32))
    block = _block_rows(n, block_rows)
    out = np.empty(n, dtype=np.float32)

    backend = _torch_device() if use_torch else None
    if backend is not None:
        torch, device = backend
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        with torch.inference_mode():
            xt = torch.from_numpy(x).to(device=device, dtype=dtype)
            for start in range(0, n, block):
                if cancel_check is not None and cancel_check():
                    return None
                stop = min(start + block, n)
                sims = xt[start:stop] @ xt.T
                rows = torch.arange(stop - start, device=device)
                sims[rows, rows + start] = float("-inf")  # exclude self
                top = torch.topk(sims, k_eff, dim=1, largest=True, sorted=False).values
                out[start:stop] = top.float().mean(dim=1).cpu().numpy()
                if on_progress is not None:
                    on_progress(stop, n)
        return out

    for start in range(0, n, block):
        if cancel_check is not None and cancel_check():
            return None
        stop = min(start + block, n)
        sims = x[start:stop] @ x.T
        rows = np.arange(stop - start)
        sims[rows, rows + start] = -np.inf
        # partition puts the k largest in the last k columns (unordered).
        top = np.partition(sims, n - k_eff, axis=1)[:, n - k_eff :]
        out[start:stop] = top.mean(axis=1)
        if on_progress is not None:
            on_progress(stop, n)
    return out


def outlier_percentiles(mean_similarity: np.ndarray) -> np.ndarray:
    """Turn kNN similarities into a 0–1 outlier rank (1 = most unusual).

    The rank is the fraction of photos that are *less* unusual, so a value of
    0.99 means "more unusual than 99 % of the library".  Ties share a rank.
    """
    n = int(mean_similarity.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.float32)
    if n == 1:
        return np.zeros(1, dtype=np.float32)
    # Lower similarity = more unusual.  Rank ascending by similarity descending.
    sims = np.asarray(mean_similarity, dtype=np.float64)
    order = np.argsort(-sims, kind="stable")
    sorted_sims = -sims[order]
    # Average ranks for ties: first index of each value in the sorted array.
    first = np.searchsorted(sorted_sims, sorted_sims, side="left")
    last = np.searchsorted(sorted_sims, sorted_sims, side="right") - 1
    ranks_sorted = (first + last) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return (ranks / (n - 1)).astype(np.float32)
