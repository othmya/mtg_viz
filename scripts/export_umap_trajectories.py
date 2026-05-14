#!/usr/bin/env python3
"""
Export per-sound sequential 3D UMAP coordinates (time order) from enc24k_umap_3d_export.npz.

Writes results/PCA_plots/umap_trajectories_by_sound.npz with:
  sound_labels: object array of shape (n_sounds,)
  traj_N: int32 (n_sounds,) — number of points per sound in stacked array
  traj_xyz: float32 (sum_N, 3) — concatenated rows in global row order (same as pca_enc24k_all.py)

Optional: --normalize matches build_umap_structures_viewer.py hull scaling.

Use for external tools; the HTML viewer embeds the same data when you run build_umap_structures_viewer.py.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    c = 0.5 * (lo + hi)
    span = np.max(hi - lo)
    if span <= 1e-12:
        span = 1.0
    return (xyz - c) * (2.2 / span)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="results/PCA_plots/enc24k_umap_3d_export.npz")
    p.add_argument(
        "--out",
        default="results/PCA_plots/umap_trajectories_by_sound.npz",
        help="Output .npz with stacked trajectories + counts",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Apply same normalization as enc24k_umap_structures_viewer hulls",
    )
    args = p.parse_args()

    data = np.load(args.npz)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    sound_id = np.asarray(data["sound_id"], dtype=np.int32)
    sounds = np.asarray(data["sounds"])

    if args.normalize:
        xyz = _normalize_xyz(xyz).astype(np.float32)

    n_sounds = int(sound_id.max()) + 1 if sound_id.size else 0
    counts = np.bincount(sound_id, minlength=n_sounds)
    labels = np.array([str(sounds[i]) for i in range(n_sounds)], dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        sound_labels=labels,
        traj_N=counts.astype(np.int32),
        traj_xyz=xyz,
        normalized=np.array(args.normalize),
    )
    print(f"Wrote {args.out}")
    print(f"  sounds={n_sounds} stacked_rows={xyz.shape[0]} normalize={args.normalize}")


if __name__ == "__main__":
    main()
