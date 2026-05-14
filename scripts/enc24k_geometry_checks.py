#!/usr/bin/env python3
"""
Per-file EnCodec geometry checks (128-D, time along rows):

1) Temporal smoothness: mean cosine similarity adjacent vs random non-adjacent pairs
   (same temporal lag distribution).
2) Chord vs path length: ‖z_0−z_{T-1}‖ vs Σ‖z_{t+1}−z_t‖ in native space (and optional PCA-3).
3) Per-file PCA variance: cumulative explained variance for first K components.

Use this to validate time-ordered trajectory structure before heavier latent-granular plots.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from enc24k_embedding_utils import load_embedding


def _adjacent_cosine(z: np.ndarray) -> float:
    """Mean cosine between consecutive rows."""
    if z.shape[0] < 2:
        return float("nan")
    a = z[:-1]
    b = z[1:]
    na = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    c = np.sum(a * b, axis=1) / na
    return float(np.mean(np.clip(c, -1.0, 1.0)))


def _random_lag_cosine(z: np.ndarray, n_samples: int, rng: np.random.Generator) -> float:
    """Mean cosine for pairs (t, t+1) with random t, excluding actual adjacency? 
    We use random t and t' = t+1 only when t+1 < T — that's still adjacent in definition.
    Better: compare pairs (t, t+k) for random t and random k in [2, min(32, T-1)]."""
    t_max, _ = z.shape
    if t_max < 4:
        return float("nan")
    k_hi = min(32, t_max - 1)
    if k_hi < 2:
        return float("nan")
    cos_vals = []
    for _ in range(n_samples):
        k = int(rng.integers(2, k_hi + 1))
        t0 = int(rng.integers(0, t_max - k))
        a = z[t0]
        b = z[t0 + k]
        na = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        cos_vals.append(float(np.dot(a, b) / na))
    return float(np.mean(np.clip(cos_vals, -1.0, 1.0)))


def _chord_path_ratio(z: np.ndarray, space: str, rng: np.random.Generator) -> dict[str, float]:
    t = z.shape[0]
    if t < 2:
        return {"chord": float("nan"), "path": float("nan"), "ratio": float("nan")}
    work = z.astype(np.float64, copy=True)
    if space == "pca3":
        p = PCA(n_components=min(3, z.shape[0], z.shape[1]), random_state=42)
        work = p.fit_transform(work)
    diffs = np.diff(work, axis=0)
    path = float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))
    chord = float(np.linalg.norm(work[-1] - work[0]))
    ratio = path / (chord + 1e-12)
    return {"chord": chord, "path": path, "ratio": ratio}


def _pca_variance_curve(z: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(max_k, z.shape[0], z.shape[1])
    if k < 1:
        return np.array([]), np.array([])
    p = PCA(n_components=k, random_state=42)
    p.fit(z)
    return np.arange(1, k + 1), np.cumsum(p.explained_variance_ratio_)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npy", action="append", default=[], help="Path to a .npy file (repeatable).")
    p.add_argument(
        "--glob-dir",
        default="",
        help="If set, glob *.npy under this directory (in addition to --npy).",
    )
    p.add_argument(
        "--assume-layout",
        choices=("row_is_time", "col_is_time"),
        default="row_is_time",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lag-samples", type=int, default=4000)
    p.add_argument("--pca-top-k", type=int, default=24)
    p.add_argument("--out-csv", default="", help="If set, append one row per file.")
    args = p.parse_args()

    assume_row = args.assume_layout == "row_is_time"
    rng = np.random.default_rng(args.seed)

    paths: list[str] = list(args.npy)
    if args.glob_dir:
        import glob

        paths.extend(sorted(glob.glob(os.path.join(args.glob_dir, "**/*.npy"), recursive=True)))
    if not paths:
        raise SystemExit("pass --npy and/or --glob-dir")

    rows_out: list[dict[str, float | str]] = []

    for path in paths:
        z = load_embedding(path, assume_row_is_time=assume_row)
        adj = _adjacent_cosine(z)
        rnd = _random_lag_cosine(z, args.lag_samples, rng)
        c128 = _chord_path_ratio(z, "native", rng)
        c3 = _chord_path_ratio(z, "pca3", rng)
        ks, cum = _pca_variance_curve(z, args.pca_top_k)
        frac3 = float(cum[2]) if cum.size > 2 else float("nan")
        frac10 = float(cum[9]) if cum.size > 9 else (float(cum[-1]) if cum.size else float("nan"))

        row = {
            "path": path,
            "T": z.shape[0],
            "mean_cos_adjacent": adj,
            "mean_cos_lag2_32": rnd,
            "smoothness_margin": adj - rnd,
            "chord_l2_128": c128["chord"],
            "path_l2_128": c128["path"],
            "ratio_path_chord_128": c128["ratio"],
            "chord_pca3": c3["chord"],
            "path_pca3": c3["path"],
            "ratio_path_chord_pca3": c3["ratio"],
            "pca_cumvar_k3": frac3,
            "pca_cumvar_k10": frac10,
        }
        rows_out.append(row)
        print(
            f"{path}\n"
            f"  T={z.shape[0]}  adjacent_cos={adj:.4f}  random_lag_cos={rnd:.4f}  "
            f"margin={adj - rnd:.4f}\n"
            f"  path/chord (128D)={c128['ratio']:.3f}  (PCA3)={c3['ratio']:.3f}\n"
            f"  PCA cum. var k≤3, k≤10: {frac3:.3f}, {frac10:.3f}\n"
        )

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            if write_header:
                w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"Appended {len(rows_out)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
