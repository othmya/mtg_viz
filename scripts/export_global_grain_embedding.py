#!/usr/bin/env python3
"""
Fit global PCA(3) and UMAP(3) on pooled L2-normalized EnCodec *grains* (same construction as latent-granular bundles).

Writes:
  • enc24k_global_grain_{fingerprint}_pca.npz — mean, components, corpus normalization params, metadata
  • enc24k_global_grain_{fingerprint}_umap.joblib — fitted umap.UMAP for .transform() at bundle time
  • enc24k_global_grain_{fingerprint}_corpus_cloud.json — subsampled full-corpus 3D points (PCA + UMAP spaces)
    with per-category colour + marker type for the latent granular viewer background cloud.

Requires: numpy, scikit-learn, umap-learn, joblib.

Example:
  python3 scripts/export_global_grain_embedding.py \\
    --enc-root data/enc24k --out-dir results/PCA_plots/ \\
    --window-size 1 --stride 1 --grain-mode mean
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import joblib
import numpy as np
import umap
from sklearn.decomposition import PCA

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from enc24k_embedding_utils import (
    GrainMode,
    grain_fingerprint,
    load_embedding,
    l2_normalize_rows,
    make_grains,
)
from build_latent_granular_3d_viewer import (  # noqa: E402
    _apply_normalize_xyz,
    _b64_f32,
    _normalize_xyz_center_span,
    scan_enc_tree,
)


def _b64_u8(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")


def _folder_colours(n_folders: int) -> list[str]:
    palette = [
        "#c41e3a",
        "#2e7d32",
        "#1565c0",
        "#6a1b9a",
        "#ef6c00",
        "#00838f",
        "#5d4037",
        "#ad1457",
        "#37474f",
    ]
    return [palette[i % len(palette)] for i in range(n_folders)]


def collect_all_grains_labeled(
    enc_root: str,
    *,
    window_size: int,
    stride: int,
    grain_mode: GrainMode,
    assume_row_is_time: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack normalized grains with folder_idx per row (same order as scan_enc_tree folders)."""
    tree = scan_enc_tree(enc_root)
    folders = tree["folders"]
    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for fi, folder in enumerate(folders):
        dpath = os.path.join(enc_root, folder)
        for fname in sorted(f for f in os.listdir(dpath) if f.endswith(".npy")):
            path = os.path.join(dpath, fname)
            x = load_embedding(path, assume_row_is_time=assume_row_is_time)
            if window_size == 1 and stride == 1:
                g = x
            else:
                if x.shape[0] < window_size:
                    continue
                g = make_grains(x, window_size, stride, grain_mode)
            g = l2_normalize_rows(g.astype(np.float32))
            chunks.append(g)
            labels.append(np.full(g.shape[0], fi, dtype=np.int32))
    if not chunks:
        raise RuntimeError(f"no grains collected under {enc_root}")
    G = np.concatenate(chunks, axis=0)
    folder_idx = np.concatenate(labels, axis=0)
    return G, folder_idx, folders


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enc-root", default="data/enc24k")
    p.add_argument("--out-dir", default="results/PCA_plots")
    p.add_argument("--window-size", type=int, default=1)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--grain-mode", choices=("mean", "concat"), default="mean")
    p.add_argument("--assume-layout", choices=("row_is_time", "col_is_time"), default="row_is_time")
    p.add_argument(
        "--max-umap-grains",
        type=int,
        default=120_000,
        help="Max rows used to fit UMAP (subsample if larger); PCA uses all unless --max-pca-grains set",
    )
    p.add_argument("--max-pca-grains", type=int, default=0, help="If >0, subsample this many rows for PCA fit")
    p.add_argument(
        "--max-corpus-cloud",
        type=int,
        default=48_000,
        help="Max points in *_corpus_cloud.json per space (uniform subsample)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    args = p.parse_args()

    stride = args.window_size if args.stride is None else args.stride
    assume_row = args.assume_layout == "row_is_time"
    rng = np.random.default_rng(args.seed)

    print("Collecting grains…")
    G, folder_idx, folders = collect_all_grains_labeled(
        args.enc_root,
        window_size=args.window_size,
        stride=stride,
        grain_mode=args.grain_mode,  # type: ignore[arg-type]
        assume_row_is_time=assume_row,
    )
    n = G.shape[0]
    d = G.shape[1]
    fingerprint = grain_fingerprint(args.window_size, stride, args.grain_mode)
    print(f"  stacked {n} grains, D={d}, fingerprint={fingerprint}, folders={len(folders)}")

    G64 = G.astype(np.float64)
    if args.max_pca_grains and n > args.max_pca_grains:
        pick_fit = rng.choice(n, size=args.max_pca_grains, replace=False)
        G_pca_fit = G64[pick_fit]
        n_pca_fit = args.max_pca_grains
    else:
        G_pca_fit = G64
        n_pca_fit = n

    print("Fitting PCA(3)…")
    pca = PCA(n_components=3, random_state=args.seed)
    pca.fit(G_pca_fit)

    umap_n = min(n, args.max_umap_grains)
    if umap_n < n:
        pick_u = rng.choice(n, size=umap_n, replace=False)
        G_umap_fit = G64[pick_u]
    else:
        G_umap_fit = G64

    print(f"Fitting UMAP(3) on {G_umap_fit.shape[0]} rows…")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    reducer.fit(G_umap_fit.astype(np.float32))

    print("Projecting full corpus to PCA(3) and UMAP(3)…")
    xyz_pca_all = (G64 - pca.mean_) @ pca.components_.T
    xyz_umap_all = np.asarray(reducer.transform(G.astype(np.float32)), dtype=np.float64)

    pca_ctr, pca_span = _normalize_xyz_center_span(xyz_pca_all)
    xyz_pca_norm = _apply_normalize_xyz(xyz_pca_all, pca_ctr, pca_span)
    umap_ctr, umap_span = _normalize_xyz_center_span(xyz_umap_all)
    xyz_umap_norm = _apply_normalize_xyz(xyz_umap_all, umap_ctr, umap_span)

    max_cloud = max(1000, min(args.max_corpus_cloud, n))
    if n <= max_cloud:
        pick = np.arange(n, dtype=np.int64)
    else:
        pick = rng.choice(n, size=max_cloud, replace=False)
    pick.sort()

    cols = _folder_colours(len(folders))
    folder_styles = [{"color": cols[i], "marker": i % 7} for i in range(len(folders))]

    fi_sub = folder_idx[pick].astype(np.uint8)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"enc24k_global_grain_{fingerprint}"
    npz_path = os.path.join(args.out_dir, f"{stem}_pca.npz")
    job_path = os.path.join(args.out_dir, f"{stem}_umap.joblib")
    cloud_json_path = os.path.join(args.out_dir, f"{stem}_corpus_cloud.json")

    np.savez_compressed(
        npz_path,
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        fingerprint=np.array(fingerprint),
        d=np.int32(d),
        n_grains_total=np.int64(n),
        n_grains_pca_fit=np.int64(n_pca_fit),
        n_grains_umap_fit=np.int64(G_umap_fit.shape[0]),
        window_size=np.int32(args.window_size),
        stride=np.int32(stride),
        grain_mode=np.array(args.grain_mode),
        corpus_norm_pca_center=pca_ctr.astype(np.float32),
        corpus_norm_pca_span=np.float64(pca_span),
        corpus_norm_umap_center=umap_ctr.astype(np.float32),
        corpus_norm_umap_span=np.float64(umap_span),
        version=np.int32(2),
    )
    joblib.dump(reducer, job_path)

    corpus_payload = {
        "version": 1,
        "fingerprint": fingerprint,
        "folders": folders,
        "folder_styles": folder_styles,
        "spaces": {
            "global_pca": {
                "count": int(pick.shape[0]),
                "pos_b64": _b64_f32(xyz_pca_norm[pick].astype(np.float32).reshape(-1)),
                "folder_idx_b64": _b64_u8(fi_sub),
            },
            "global_umap": {
                "count": int(pick.shape[0]),
                "pos_b64": _b64_f32(xyz_umap_norm[pick].astype(np.float32).reshape(-1)),
                "folder_idx_b64": _b64_u8(fi_sub),
            },
        },
    }
    with open(cloud_json_path, "w", encoding="utf-8") as jf:
        json.dump(corpus_payload, jf, indent=0)

    print(f"Wrote {npz_path}")
    print(f"Wrote {job_path}")
    print(f"Wrote {cloud_json_path}")
    print("Pass to bundle build, e.g.:")
    print(
        f"  --global-embed-npz {npz_path} \\\n"
        f"  --global-umap-model {job_path}"
    )


if __name__ == "__main__":
    main()
