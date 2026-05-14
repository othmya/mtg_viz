"""Shared helpers for EnCodec npy layout (T, 128) and grain construction."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

GrainMode = Literal["mean", "concat"]


def grain_fingerprint(window_size: int, stride: int, grain_mode: str) -> str:
    """Stable id for global embedding artifacts (matches bundle grain construction)."""
    return f"w{int(window_size)}_s{int(stride)}_{grain_mode}"


def load_embedding(path: str, assume_row_is_time: bool = True) -> np.ndarray:
    x = np.load(path)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"{path}: expected 2D array, got shape {x.shape}")
    t, d = x.shape
    if not assume_row_is_time:
        x = x.T
        t, d = x.shape
    if d != 128:
        raise ValueError(f"{path}: expected 128-d columns after layout fix, got {d}")
    if t < 2:
        raise ValueError(f"{path}: need T>=2, got T={t}")
    return np.ascontiguousarray(x)


def make_grains(x: np.ndarray, window_size: int, stride: int, mode: GrainMode) -> np.ndarray:
    """x: (T, 128) -> (N, D) grains; last partial window dropped."""
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be >= 1")
    n = (x.shape[0] - window_size) // stride + 1
    if n < 2:
        raise ValueError(
            f"not enough frames for window_size={window_size}, stride={stride}: T={x.shape[0]}"
        )
    out: list[np.ndarray] = []
    for i in range(n):
        s = i * stride
        w = x[s : s + window_size]
        if mode == "mean":
            out.append(w.mean(axis=0))
        else:
            out.append(w.reshape(-1))
    g = np.stack(out, axis=0)
    return np.ascontiguousarray(g.astype(np.float32, copy=False))


def collect_source_codebook(
    enc_root: str,
    source_folder: str,
    glob_pattern: str = "*.npy",
    max_files: int | None = None,
    window_size: int = 1,
    stride: int | None = None,
    grain_mode: GrainMode = "mean",
    assume_row_is_time: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Stack grains from all npy files in source_folder into (N_cb, D)."""
    import fnmatch

    stride = stride if stride is not None else window_size
    dpath = os.path.join(enc_root, source_folder)
    if not os.path.isdir(dpath):
        raise FileNotFoundError(dpath)
    files = sorted(
        f for f in fnmatch.filter(os.listdir(dpath), glob_pattern) if f.endswith(".npy")
    )
    if max_files is not None:
        files = files[: max_files]
    chunks: list[np.ndarray] = []
    stems: list[str] = []
    for fname in files:
        path = os.path.join(dpath, fname)
        x = load_embedding(path, assume_row_is_time=assume_row_is_time)
        if window_size == 1 and stride == 1:
            g = x
        else:
            g = make_grains(x, window_size, stride, grain_mode)
        chunks.append(g)
        stems.append(os.path.splitext(fname)[0])
    if not chunks:
        raise RuntimeError(f"no source files matched in {dpath}")
    return np.concatenate(chunks, axis=0), stems


def l2_normalize_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return (a / n).astype(np.float32, copy=False)
