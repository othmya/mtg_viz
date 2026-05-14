#!/usr/bin/env python3
"""
Ask: do similar sounds (e.g. everything under data/enc24k/soundscapes) share
similar *internal* geometry in EnCodec space?

We operationalise "internal geometry" without a single global UMAP per file:
for each recording we draw many random pairs of grains (or frames), measure
Euclidean distance in feature space, and summarise the resulting distribution
with a fixed-bin histogram. Similar histograms (cosine similarity) mean similar
typical spreads / relational structure inside that sound, independent of
global placement in a joint embedding.

Pass --folder-b to build one combined similarity matrix across two category
folders (within-A, within-B, and cross pairs share histogram bin edges).

Expected layout: each .npy is float32 with time along rows and 128-d EnCodec
vectors along columns, shape (T, 128). If a file is stored as (128, T), pass
--assume-layout row_is_time=false to transpose once.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from enc24k_embedding_utils import load_embedding, make_grains


def _load_embedding(path: str, assume_row_is_time: bool) -> np.ndarray:
    return load_embedding(path, assume_row_is_time=assume_row_is_time)


def _pairwise_dists_sample(
    feats: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random distinct index pairs (i, j), i < j; Euclidean distance in rows."""
    n = feats.shape[0]
    if n < 2:
        raise ValueError("need at least two points")
    out = np.empty(n_pairs, dtype=np.float32)
    # rejection sample pairs (fast enough for audio-scale n)
    for k in range(n_pairs):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        di = feats[i] - feats[j]
        out[k] = float(np.sqrt(np.dot(di, di)))
    return out


def _histogram_signature(
    dists: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    hist, _ = np.histogram(dists, bins=edges)
    h = hist.astype(np.float64)
    s = h.sum()
    if s <= 0:
        return h
    return h / s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--enc-root",
        default="data/enc24k",
        help="Root that contains one subfolder per category (e.g. soundscapes).",
    )
    p.add_argument(
        "--folder",
        default="soundscapes",
        help="Subfolder of enc-root to compare (only these .npy files).",
    )
    p.add_argument(
        "--folder-b",
        default="",
        help="Optional second subfolder; builds one combined histogram-similarity matrix (within A, within B, and cross).",
    )
    p.add_argument("--window-size", type=int, default=1, help="Grain length in frames (1 = use raw frames).")
    p.add_argument("--stride", type=int, default=None, help="Grain stride; default equals window-size.")
    p.add_argument(
        "--grain-mode",
        choices=("mean", "concat"),
        default="mean",
        help="How each grain vector is built from its window.",
    )
    p.add_argument(
        "--assume-layout",
        choices=("row_is_time", "col_is_time"),
        default="row_is_time",
        help="row_is_time: (T,128). col_is_time: (128,T) on disk, transposed after load.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-pairs", type=int, default=8000, help="Random pairs per file for distance samples.")
    p.add_argument("--n-bins", type=int, default=48, help="Histogram bins for the distance distribution.")
    p.add_argument(
        "--out-dir",
        default="results/internal_geometry",
        help="Directory for heatmap + CSV exports.",
    )
    args = p.parse_args()

    stride = args.stride if args.stride is not None else args.window_size
    assume_row_is_time = args.assume_layout == "row_is_time"
    rng = np.random.default_rng(args.seed)

    def load_folder(folder_name: str) -> tuple[list[str], list[np.ndarray]]:
        folder_path = os.path.join(args.enc_root, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"missing folder: {folder_path}")
        files = sorted(f for f in os.listdir(folder_path) if f.endswith(".npy"))
        stems_l: list[str] = []
        feats_l: list[np.ndarray] = []
        for fname in files:
            path = os.path.join(folder_path, fname)
            x = _load_embedding(path, assume_row_is_time=assume_row_is_time)
            if args.window_size == 1 and stride == 1:
                g = x
            else:
                g = _make_grains(x, args.window_size, stride, args.grain_mode)  # type: ignore[arg-type]
            stems_l.append(os.path.splitext(fname)[0])
            feats_l.append(g)
        return stems_l, feats_l

    stems: list[str]
    feats_list: list[np.ndarray]
    labels: list[str]

    if args.folder_b:
        stems_a, feats_a = load_folder(args.folder)
        stems_b, feats_b = load_folder(args.folder_b)
        if len(feats_a) < 1 or len(feats_b) < 1:
            raise RuntimeError("each folder must have at least one .npy when using --folder-b")
        feats_list = feats_a + feats_b
        labels = [f"{args.folder}/{s}" for s in stems_a] + [f"{args.folder_b}/{s}" for s in stems_b]
        stems = labels
    else:
        stems, feats_list = load_folder(args.folder)
        if len(feats_list) < 2:
            fp = os.path.join(args.enc_root, args.folder)
            raise RuntimeError(f"need at least two .npy files in {fp}, found {len(feats_list)}")
        labels = stems

    # Pooled distances to set shared bin edges (robust range).
    pooled: list[np.ndarray] = []
    for g in feats_list:
        pooled.append(_pairwise_dists_sample(g, min(args.n_pairs, 2000), rng))
    all_d = np.concatenate(pooled, axis=0)
    lo, hi = np.percentile(all_d, [1.0, 99.0])
    if hi <= lo:
        hi = float(lo + 1e-6)
    edges = np.linspace(lo, hi, args.n_bins + 1, dtype=np.float64)

    sigs: list[np.ndarray] = []
    for g in feats_list:
        d = _pairwise_dists_sample(g, args.n_pairs, rng)
        sigs.append(_histogram_signature(d, edges))

    S = np.stack(sigs, axis=0)  # (n_files, n_bins)
    # Cosine similarity matrix between histograms (treat as vectors).
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    sim = Sn @ Sn.T

    os.makedirs(args.out_dir, exist_ok=True)
    if args.folder_b:
        tag = f"{args.folder}_x_{args.folder_b}_w{args.window_size}_s{stride}_{args.grain_mode}"
    else:
        tag = f"{args.folder}_w{args.window_size}_s{stride}_{args.grain_mode}"
    csv_path = os.path.join(args.out_dir, f"{tag}_pair_hist_cosine_sim.csv")
    png_path = os.path.join(args.out_dir, f"{tag}_pair_hist_cosine_sim.png")

    # CSV: matrix with header
    header = "," + ",".join(labels)
    lines = [header]
    for i, row_name in enumerate(labels):
        lines.append(row_name + "," + ",".join(f"{sim[i, j]:.6f}" for j in range(len(labels))))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    plt.figure(figsize=(max(8, len(labels) * 0.35), max(7, len(labels) * 0.35)), dpi=200)
    sns.heatmap(
        sim,
        xticklabels=labels,
        yticklabels=labels,
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "cosine similarity of pairwise-distance histograms"},
    )
    title_folder = f"{args.folder} × {args.folder_b}" if args.folder_b else args.folder
    plt.title(
        f"Internal geometry similarity — {title_folder}\n"
        f"(random intra-sound pair distances in EnCodec space; "
        f"window={args.window_size}, stride={stride}, mode={args.grain_mode})"
    )
    plt.xticks(rotation=75, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(
        "Interpretation: values near 1 mean similar *internal* spread / relational structure "
        "(not necessarily similar absolute timbres or global UMAP position)."
    )


if __name__ == "__main__":
    main()
